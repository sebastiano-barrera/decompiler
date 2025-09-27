use std::collections::HashMap;

use tracing::{event, instrument, span, Level};

use crate::{
    cfg::BlockID,
    mil::{ArithOp, Insn, Reg, RegType},
    ssa, ty,
    util::Bytes,
    x86_to_mil,
};

mod mem;

fn fold_constants(insn: Insn, prog: &mut ssa::OpenProgram, bid: BlockID) -> Insn {
    use crate::mil::ArithOp;

    /// Evaluate expression (ak (op) bk)
    fn eval_const(op: ArithOp, ak: i64, bk: i64) -> i64 {
        match op {
            ArithOp::Add => ak + bk,
            ArithOp::Sub => ak - bk,
            ArithOp::Mul => ak * bk,
            ArithOp::Shl => ak << bk,
            ArithOp::Shr => ak >> bk,
            ArithOp::BitXor => ak ^ bk,
            ArithOp::BitAnd => ak & bk,
            ArithOp::BitOr => ak | bk,
        }
    }

    /// Compute (op_res, rk) such that, for all x:
    ///   (x <op> ak) <op> bk <===> x <op> rk
    /// or, equivalently:
    ///   (x <op> ak) <op> (y <op> bk)<===> (x <op> y) <op> rk
    ///
    /// Returns None for non-associative operators.
    fn assoc_const(op_in: ArithOp, ak: i64, bk: i64) -> Option<i64> {
        match op_in {
            ArithOp::Add => Some(ak + bk),
            ArithOp::Mul => Some(ak * bk),
            ArithOp::Shl => Some(ak + bk),
            _ => None,
        }
    }

    /*
        \math-container{\frac{a𝛽,𝛽a|a\index{𝛽}}

        \text{Convert sub to add, to get more associative nodes:}
        \frac{a\power-index{𝛽|-}|a\power-index{+|-𝛽}},\frac{a-b\power-index{+|𝛾}|a+b\power-index{+|-𝛾}}

        \text{if op is associative: }\frac{a\index{𝛽}b\index{𝛾}|(ab)\index{\underline{𝛽𝛾}}},\frac{(a\index{𝛽})\index{𝛾}|a\index{\underline{𝛽𝛾}}}

        \fracslashed{𝛽\index{𝛾}|\underline{𝛽𝛾}}

        [⋅=+]⟹\frac{a\index{0}|a}
        [⋅=×]⟹\frac{a\index{1}|a}}
    */

    let (mut op, mut lr, mut li, mut ri) = match insn {
        Insn::Arith(op, lr, rr) => {
            let li = prog.get(lr).unwrap();
            let ri = prog.get(rr).unwrap();

            // ensure the const is on the right
            if let Insn::Int { .. } = li {
                (op, rr, ri, li)
            } else {
                (op, lr, li, ri)
            }
        }
        Insn::ArithK(op, a, bk) => (
            op,
            a,
            prog.get(a).unwrap(),
            Insn::Int { value: bk, size: 8 },
        ),
        _ => return insn,
    };

    // if there is a Const, it's on the right (or they are both Const)
    assert!(matches!(ri, Insn::Int { .. }) || !matches!(li, Insn::Int { .. }));

    // convert sub to add to increase the probability of applying the following rules
    if op == ArithOp::Sub {
        if let Insn::Int { value, .. } = &mut ri {
            op = ArithOp::Add;
            *value = -*value;
        } else if let Insn::ArithK(ArithOp::Add, _, r_k) = &mut ri {
            op = ArithOp::Add;
            *r_k = -*r_k;
        }
    }

    match (li, ri) {
        // (a op ka) op (b op kb) === (a op b) op (ka op kb)  (if op is associative)
        (Insn::ArithK(l_op, llr, lk), Insn::ArithK(r_op, rlr, rk))
            if l_op == r_op && l_op == op =>
        {
            if let Some(k) = assoc_const(op, lk, rk) {
                li = fold_constants(Insn::Arith(op, llr, rlr), prog, bid);
                lr = prog.append_new(bid, li);
                ri = Insn::Int { value: k, size: 8 };
            }
        }
        // (a op ka) op kb === a op (ka op kb)  (if op is associative)
        (Insn::ArithK(l_op, llr, lk), Insn::Int { value: rk, .. }) if l_op == op => {
            if let Some(k) = assoc_const(op, lk, rk) {
                li = prog.get(llr).unwrap();
                lr = llr;
                ri = Insn::Int { value: k, size: 8 };
            }
        }
        _ => {}
    }

    match (op, li, ri) {
        (op, Insn::Int { value: ka, .. }, Insn::Int { value: kb, .. }) => Insn::Int {
            value: eval_const(op, ka, kb),
            size: 8,
        },
        (ArithOp::Add, _, Insn::Int { value: 0, .. }) => Insn::Get(lr),
        (ArithOp::Mul, _, Insn::Int { value: 1, .. }) => Insn::Get(lr),

        (op, _, Insn::Int { value: kr, .. }) => Insn::ArithK(op, lr, kr),
        _ => {
            // dang it, we couldn't hack it
            insn
        }
    }
}

fn fold_subregs(insn: Insn, prog: &ssa::Program) -> Insn {
    // operators that matter here are:
    // - subrange: src[a..b]
    //      b > a; b <= 8; a, b >= 0
    // - concatenation: hi :: lo
    //
    // two optimizations in one, where the argument may "skip over" the Concat,
    // possibly shifting the range:
    // - Part(Concat(...), ...)
    // - Part(Part(...), ...)

    let Insn::Part { src, offset, size } = insn else {
        return insn;
    };

    let end = offset + size;

    let src_sz = prog.reg_type(src).bytes_size().unwrap();
    assert!(end as usize <= src_sz);

    let src_insn = prog.get(src).unwrap();
    match src_insn {
        Insn::Part {
            src: up_src,
            offset: up_offset,
            size: up_size,
        } => {
            let up_end = up_offset + up_size;
            let up_src_sz = prog.reg_type(up_src).bytes_size().unwrap();
            assert!(up_end as usize <= up_src_sz);

            Insn::Part {
                src: up_src,
                offset: offset + up_offset,
                size,
            }
        }

        Insn::Concat { lo, hi } => {
            let lo_sz = prog
                .reg_type(lo)
                .bytes_size()
                .unwrap()
                .try_into()
                .expect("size is too large for Concat");

            if end <= lo_sz {
                // offset..size falls entirely within lo
                Insn::Part {
                    src: lo,
                    offset,
                    size,
                }
            } else if offset >= lo_sz {
                // offset..size falls entirely within hi
                Insn::Part {
                    src: hi,
                    offset: offset - lo_sz,
                    size,
                }
            } else {
                // offset..size covers (at least part of) both lo and hi
                insn
            }
        }

        _ => insn,
    }
}

fn fold_concat_void(insn: Insn, prog: &ssa::Program) -> Insn {
    let Insn::Concat { lo, hi } = insn else {
        return insn;
    };

    match (prog.reg_type(lo), prog.reg_type(hi)) {
        (RegType::Bytes(0), RegType::Bytes(0)) => Insn::Void,
        (RegType::Bytes(0), _) => Insn::Get(hi),
        (_, RegType::Bytes(0)) => Insn::Get(lo),
        (_, _) => insn,
    }
}

fn fold_bitops(insn: Insn, prog: &ssa::Program) -> Insn {
    match insn {
        // TODO put the appropriate size
        Insn::Arith(ArithOp::BitXor, a, b) if a == b => Insn::Int {
            value: 0,
            size: prog.reg_type(a).bytes_size().unwrap().try_into().unwrap(),
        },
        Insn::Arith(ArithOp::BitAnd, a, b) if a == b => Insn::Get(a),
        Insn::Arith(ArithOp::BitOr, a, b) if a == b => Insn::Get(a),
        _ => insn,
    }
}

fn fold_part_part(insn: Insn, prog: &ssa::Program) -> Insn {
    // the pattern is
    //  r0 <- (any, of size s0)
    //  r1 <- Widen r0 to size s1,  s1 > s0
    //  r2 <- Part of r1, 0..plen
    //
    // if s1 > s0 && plen < s1
    //
    //  r0 <- (any, of size s0)
    //  r2 <- Widen r0 to size plen
    // (skip the r1 Widen, and transform Part to a shorter Widen)

    if let Insn::Part {
        src: out_src,
        offset: out_offset,
        size: out_size,
    } = insn
    {
        if let Insn::Part {
            src: in_src,
            offset: in_offset,
            size: in_size,
        } = prog.get(out_src).unwrap()
        {
            assert!(out_size <= in_size);
            return Insn::Part {
                src: in_src,
                offset: out_offset + in_offset,
                size: out_size,
            };
        }
    }

    insn
}

fn fold_part_concat(insn: Insn, prog: &ssa::Program) -> Insn {
    // the pattern is
    //  r0 <- (any, of size s0)
    //  r1 <- Widen r0 to size s1,  s1 > s0
    //  r2 <- Part of r1, 0..plen
    //
    // if s1 > s0 && plen < s1
    //
    //  r0 <- (any, of size s0)
    //  r2 <- Widen r0 to size plen
    // (skip the r1 Widen, and transform Part to a shorter Widen)

    if let Insn::Part {
        src: p_src,
        offset: p_offset,
        size: p_size,
    } = insn
    {
        if let Insn::Concat { lo, hi } = prog.get(p_src).unwrap() {
            let lo_size = prog.reg_type(lo).bytes_size().unwrap().try_into().unwrap();

            if p_offset + p_size <= lo_size {
                return Insn::Part {
                    src: lo,
                    offset: p_offset,
                    size: p_size,
                };
            } else if p_offset >= lo_size {
                return Insn::Part {
                    src: hi,
                    offset: p_offset - lo_size,
                    size: p_size,
                };
            }
        }
    }

    insn
}

fn fold_part_widen(insn: Insn, prog: &ssa::Program) -> Insn {
    // the pattern is
    //  r0 <- (any, of size s0)
    //  r1 <- Widen r0 to size s1,  s1 > s0
    //  r2 <- Part of r1, 0..plen
    //
    // if s1 > s0 && plen < s1
    //
    //  r0 <- (any, of size s0)
    //  r2 <- Widen r0 to size plen
    // (skip the r1 Widen, and transform Part to a shorter Widen)

    if let Insn::Part {
        src: part_src,
        offset: 0,
        size: part_size,
    } = insn
    {
        if let Insn::Widen {
            reg,
            target_size: _,
            sign,
        } = prog.get(part_src).unwrap()
        {
            // TODO convert to error
            let orig_size = prog.reg_type(reg).bytes_size().unwrap().try_into().unwrap();

            if part_size < orig_size {
                // skip the widen, as it does nothing; directly Part the orig. reg
                return Insn::Part {
                    src: reg,
                    offset: 0,
                    size: part_size,
                };
            } else if part_size == orig_size {
                // skip the widen, as it does nothing; use the reg directly,
                // we're not even really Part'ing it
                return Insn::Get(reg);
            } else {
                // widen to a smaller size
                return Insn::Widen {
                    reg,
                    target_size: part_size,
                    sign,
                };
            }
        }
    }

    insn
}

fn fold_widen_const(insn: Insn, prog: &ssa::Program) -> Insn {
    // TODO add signedness to Const as well? then we could check if they match
    if let Insn::Widen {
        reg,
        target_size,
        sign: _,
    } = insn
    {
        if let Insn::Int { value, size } = prog.get(reg).unwrap() {
            assert!(target_size > size);
            return Insn::Int {
                value,
                size: target_size,
            };
        }
    }
    insn
}

fn fold_widen_null(insn: Insn, prog: &ssa::Program) -> Insn {
    if let Insn::Widen {
        reg,
        target_size,
        sign: _,
    } = insn
    {
        if let RegType::Bytes(sz) = prog.reg_type(reg) {
            if target_size as usize == sz {
                return Insn::Get(reg);
            }
        }
    }

    insn
}

fn fold_part_null(insn: Insn, prog: &ssa::Program) -> Insn {
    if let Insn::Part {
        src,
        offset: 0,
        size,
    } = insn
    {
        if let RegType::Bytes(src_size) = prog.reg_type(src) {
            if src_size == size as usize {
                return Insn::Get(src);
            }
        }
    }

    insn
}

fn fold_shr_part(insn: Insn, prog: &mut ssa::OpenProgram, bid: BlockID) -> Insn {
    if let Insn::ArithK(ArithOp::Shr, reg, shift_len) = insn {
        if shift_len >= 0 && shift_len % 8 == 0 {
            if let Insn::Part { src, offset, size } = prog.get(reg).unwrap() {
                let shift_bytes: u16 = (shift_len / 8).try_into().unwrap();
                let smaller_part = prog.append_new(
                    bid,
                    Insn::Part {
                        src,
                        offset: offset + shift_bytes,
                        size: size - shift_bytes,
                    },
                );

                // widen back to original size
                return Insn::Widen {
                    reg: smaller_part,
                    target_size: size,
                    sign: false,
                };
            }
        }
    }

    insn
}

fn fold_get(mut insn: Insn, prog: &ssa::Program) -> Insn {
    for input in insn.input_regs_iter() {
        loop {
            let input_def = prog.get(*input).unwrap();
            if let Insn::Get(arg) = input_def {
                *input = arg;
            } else {
                break;
            }
        }
    }

    insn
}

fn fold_part_void(insn: Insn) -> Insn {
    if let Insn::Part { size: 0, .. } = insn {
        return Insn::Void;
    }
    insn
}

fn fold_part_const(insn: Insn, prog: &ssa::Program) -> Insn {
    if let Insn::Part { src, offset, size } = insn {
        if let Insn::Int {
            value: src_value,
            size: src_size,
        } = prog.get(src).unwrap()
        {
            let src_bytes = prog.int_bytes(src_value, src_size);

            let offset = offset as usize;
            let size = size as usize;
            let cut = &src_bytes.as_slice()[offset..offset + size];
            return Insn::Bytes(Bytes::from_slice(cut).unwrap());
        }
    }

    insn
}

/// If the given instruction is `Insn::Part { src, offset, size }`, and we know
/// the type of `src`, then use the type information to replace it with a chain
/// of instructions that extract the relevant part of `src` (e.g. named struct
/// members, array element access by index).
///
/// Doesn't do anything for other types of instructions or when no type info is
/// available.
fn apply_type_selection(
    insn: Insn,
    prog: &mut ssa::OpenProgram,
    bid: BlockID,
    types: &ty::TypeSet,
) -> Insn {
    let Insn::Part { src, offset, size } = insn else {
        return insn;
    };

    let offset = offset as usize;
    let size = size as usize;

    // desist if there is no type information for the src value
    let Some(src_tyid) = prog.value_type(src) else {
        return insn;
    };
    let ty::Selection {
        tyid: selected_tyid,
        path,
    } = match types.select(src_tyid, ty::ByteRange::new(offset, offset + size)) {
        Ok(part_tyid) => part_tyid,
        Err(err) => {
            event!(
                Level::INFO,
                ?src_tyid,
                ?offset,
                ?size,
                ?err,
                "unable to select in type"
            );
            return insn;
        }
    };

    event!(Level::TRACE, ?offset, ?size, "type selected");

    if path.len() == 0 {
        // this is what TypeSet::select returns when the required byte range precisely matches src
        return Insn::Get(src);
    }

    // in the most generic case, it contains a single SelectStep::RawBytes
    let mut last_reg = src;
    for step in path {
        // add insns for intermediate steps
        last_reg = prog.append_new(bid, step.to_insn(last_reg));
    }

    // for now, type IDs are not assigned to the intermediate steps
    prog.set_value_type(last_reg, Some(selected_tyid));
    // because we currently only return an Insn, we can't call set_value_type
    // on the register corresponding to its result (`append_new` is done by
    // `canonical`). so, Insn::Get it is
    Insn::Get(last_reg)
}

/// Add cast instructions wherever high level type information indicates that it might be valuable to do so.
///
/// Typical example is: a value with HLL ty::Ty::Float but RegType::Bytes. In
/// this case, `Insn::ReinterpretFloat` (RegType::Float) can be inserted, which
/// may (in combination with constant folding), produce a proper float literal
/// in the final decompiled output.
fn add_cast_for_type(
    insn: Insn,
    reg: Reg,
    prog: &mut ssa::OpenProgram,
    bid: BlockID,
    types: &ty::TypeSet,
) -> Insn {
    let Some(tyid) = prog.value_type(reg) else {
        return insn;
    };
    let Some(ty) = types.get_through_alias(tyid) else {
        return insn;
    };

    let rt = prog.reg_type(reg);

    match (rt, &*ty) {
        (RegType::Bytes(_), ty::Ty::Float(float_ty)) => {
            // move the original insn to a different register, without copying
            // type info! (or we go in an infinite loop)
            let new_reg = prog.append_new(bid, insn);

            match float_ty.size {
                4 => Insn::ReinterpretFloat32(new_reg),
                8 => Insn::ReinterpretFloat64(new_reg),
                _ => {
                    event!(
                        Level::WARN,
                        ?reg,
                        ?tyid,
                        ?float_ty,
                        "cannot add cast for float type of unsupported size"
                    );
                    return insn;
                }
            }
        }
        _ => insn,
    }
}

/// Perform the standard chain of transformations that we intend to generally apply to programs
#[tracing::instrument(skip_all)]
pub fn canonical(prog: &mut ssa::Program, types: &ty::TypeSet) {
    prog.assert_invariants();

    // apply transforms in lockstep
    //
    // in this setup, we scan the program once; for each instruction, we apply
    // every transformation (in a fixed order). despite the fixed order, every
    // transform "sees" the effect of *all* transforms when it "looks back" at
    // the instruction's dependencies.
    //
    // Note that `fold_get` is the first in the chain, so any `Insn::Get`
    // introduced by earlier transforms is going to be "dereferenced" and will
    // most likely end up dead and eliminated by the time this function returns.

    // TODO make this architecture agnostic
    let mem_ref_reg = find_mem_ref(prog);
    let mut prog = ssa::OpenProgram::wrap(prog);
    let mut deduper = Deduper::new();

    propagate_call_types(&mut prog, types);

    let mut any_change = true;
    while any_change {
        any_change = false;

        let bids: Vec<_> = prog.cfg().block_ids_rpo().collect();
        for bid in bids {
            // clear the block's schedule, then reconstruct it.
            // existing instruction are processed and replaced to keep using the memory they already occupy
            for reg in prog.clear_block_schedule(bid) {
                let span = span!(Level::TRACE, "processing", ?reg);
                let _enter = span.enter();

                let orig_block_len = prog.block_len(bid);
                let orig_insn = prog.get(reg).unwrap();
                let orig_has_fx = orig_insn.has_side_effects();

                let mut insn = orig_insn;
                if let Some(mem_ref_reg) = mem_ref_reg {
                    insn = mem::fold_load_store(&mut prog, mem_ref_reg, bid, insn);
                }
                insn = pack_aggregates(reg, insn, &mut prog, bid, types);
                insn = fold_get(insn, &prog);
                insn = fold_subregs(insn, &prog);
                insn = fold_concat_void(insn, &prog);
                insn = fold_part_part(insn, &prog);
                insn = fold_part_widen(insn, &prog);
                insn = fold_part_concat(insn, &prog);
                insn = fold_part_null(insn, &prog);
                insn = fold_part_void(insn);
                insn = fold_part_const(insn, &prog);
                insn = fold_widen_null(insn, &prog);
                insn = fold_widen_const(insn, &prog);
                insn = fold_bitops(insn, &prog);
                insn = fold_constants(insn, &mut prog, bid);
                insn = fold_shr_part(insn, &mut prog, bid);
                insn = apply_type_selection(insn, &mut prog, bid, types);
                insn = add_cast_for_type(insn, reg, &mut prog, bid, types);
                if !insn.is_replaceable_with_get() {
                    // replacing a side-effecting instruction with a non-side-effecting
                    // Insn::Get is currently wrong (would be quite complicated to handle)
                    insn = deduper.try_dedup(reg, insn);
                }
                prog.set(reg, insn);
                prog.append_existing(bid, reg);

                if insn != orig_insn {
                    event!(Level::TRACE, ?insn, ?orig_insn, "insn set");
                }

                let final_has_fx = insn.has_side_effects();
                if final_has_fx != orig_has_fx {
                    panic!(
                        " --- bug:\n\
                        orig: side fx: {orig_has_fx:?} insn: {orig_insn:?}\n\
                        final: side fx: {final_has_fx:?} insn: {insn:?}"
                    );
                }

                if insn != orig_insn || prog.block_len(bid) != orig_block_len + 1 {
                    any_change = true;
                }
            }
        }

        event!(Level::TRACE, prog = ? &*prog, "ssa after xform cycle");
    }
}

/// For each Insn::Call instruction, use the callee's type to assign types to
/// the argument values.
///
/// # Example
///
/// If we have the following SSA:
///
/// ```ignore
/// r10 <- Call {value: r5, first_arg: None}
/// r11 <- Call {value: r6, first_arg: Some(r10)}
/// r12 <- Call {value: r7, first_arg: Some(r11)}
/// r13 <- Call {value: r8, first_arg: Some(r12)}
/// r14 <- Call {callee: F, first_arg: Some(r13)}
/// ```
///
/// and we have type information for `F`, such that it is a subroutine/function type like
/// `TRet F(T1, T2, T3, T4)`, then the following type assignments are made:
///
/// * r5: T1
/// * r6: T2
/// * r7: T3
/// * r8: T4
/// * r14: TRet
#[instrument(skip_all)]
fn propagate_call_types(prog: &mut ssa::OpenProgram, types: &ty::TypeSet) {
    let mut tyids_to_set = ssa::RegMap::for_program(prog, None);
    for (_, reg) in prog.insns_rpo() {
        let Insn::Call { callee, first_arg } = prog.get(reg).unwrap() else {
            continue;
        };
        let Some(callee_tyid) = prog.value_type(callee) else {
            continue;
        };
        let Some(callee_ty) = types.get_through_alias(callee_tyid) else {
            event!(
                Level::ERROR,
                ?callee,
                ?callee_tyid,
                "callee's type ID is invalid"
            );
            continue;
        };
        let ty::Ty::Subroutine(subr_ty) = callee_ty.as_ref() else {
            event!(
                Level::ERROR,
                ?callee,
                ?callee_tyid,
                "callee's type not a subroutine"
            );
            continue;
        };

        if prog.value_type(reg).is_none() {
            tyids_to_set[reg] = Some(subr_ty.return_tyid);
        }

        let param_count = subr_ty.param_tyids.len();

        let mut arg_count = 0;
        if let Some(first_arg) = first_arg {
            for (arg, arg_tyid) in prog.get_call_args(first_arg).zip(&subr_ty.param_tyids) {
                if prog.value_type(arg).is_none() {
                    tyids_to_set[arg] = Some(*arg_tyid);
                }
                arg_count += 1;
            }
        }

        if arg_count != param_count {
            event!(
                Level::ERROR,
                ?reg,
                ?callee,
                ?arg_count,
                ?param_count,
                "call site has fewer arguments than subroutine type has parameters"
            );
        }
    }

    let mut count = 0;
    for (reg, tyid) in tyids_to_set.items() {
        if let &Some(tyid) = tyid {
            event!(Level::DEBUG, ?reg, ?tyid, "set type around call site");
            prog.set_value_type(reg, Some(tyid));
            count += 1;
        }
    }
    event!(
        Level::DEBUG,
        count,
        "finished type propagation at call sites"
    );
}

#[instrument(skip(prog, types))]
fn pack_aggregates(
    reg: Reg,
    mut insn: Insn,
    prog: &mut ssa::OpenProgram,
    bid: BlockID,
    types: &ty::TypeSet,
) -> Insn {
    let Some(tyid) = prog.value_type(reg) else {
        return insn;
    };
    let Some(tycow) = types.get_through_alias(tyid) else {
        return insn;
    };

    if let ty::Ty::Struct(struct_ty) = tycow.as_ref() {
        // this is a heuristic.
        //
        // the idea is that sometimes we want to explicitly spell out how
        // the struct value is composed (e.g., for a call argument), whereas
        // sometimes there is no value in doing so (e.g., if the struct value is
        // a FuncArgument).
        //
        // so for now the approximate implementation of this idea is:
        //
        //  - spell out the struct (with an Insn::Struct) if the insn has
        //    arguments, as that would indicate that we're composing a struct
        //    out of some currently-opaque components (e.g. a Concat of some
        //    "raw parts").
        //
        //  - keep it implicit for insns with no inputs, as that probably
        //    indicates that the value is an ancestral or some prepackaged value
        //    that has clear meaning from context (e.g., FuncArgument).
        //
        let has_inputs = insn.input_regs().len() > 0;
        if !matches!(insn, Insn::Struct { .. }) && has_inputs {
            // copy the instruction to a new value -- it will act as the "raw"
            // value of the struct, to be formally sectioned into fields
            let src = prog.append_new(bid, insn);

            let mut first_member_reg = None;

            for member in struct_ty.members.iter().rev() {
                let Some(size) = types.bytes_size(member.tyid) else {
                    event!(
                        Level::ERROR,
                        struct_tyid = ?tyid,
                        name = member.name.to_string(),
                        member_tyid = ?member.tyid,
                        "struct member type has no size; skipped"
                    );
                    continue;
                };
                // TODO figure out proper memory management
                let name = member.name.to_string().leak();
                let value = prog.append_new(
                    bid,
                    Insn::Part {
                        src,
                        offset: member.offset.try_into().unwrap(),
                        size: size.try_into().unwrap(),
                    },
                );
                prog.set_value_type(value, Some(member.tyid));

                first_member_reg = Some(prog.append_new(
                    bid,
                    Insn::StructMember {
                        name,
                        value,
                        next: first_member_reg,
                    },
                ));
            }

            insn = Insn::Struct {
                type_name: types
                    .name(tyid)
                    .map(|s| s.to_string().leak() as &str)
                    .unwrap_or("?"),
                first_member: first_member_reg,
                size: struct_ty.size.try_into().unwrap(),
            };

            // NOTE that, although `src` points to a copy of Insn, it has no
            // type info; the type info is instead kept in the new Insn::Struct,
            // assigned to the original register (agg_reg)
        }
    }

    insn
}

struct Deduper {
    rev_lookup: HashMap<Insn, Reg>,
}

impl Deduper {
    fn new() -> Self {
        Deduper {
            rev_lookup: HashMap::new(),
        }
    }

    fn try_dedup(&mut self, reg: Reg, insn: Insn) -> Insn {
        // replacing a side-effecting instruction with a non-side-effecting
        // Insn::Get is currently wrong (would be quite complicated to handle)
        if !insn.is_replaceable_with_get() {
            return insn;
        }

        let prev_reg = self.rev_lookup.entry(insn).or_insert(reg);
        if *prev_reg != reg {
            Insn::Get(*prev_reg)
        } else {
            insn
        }
    }
}

fn find_mem_ref(prog: &ssa::Program) -> Option<Reg> {
    (0..prog.reg_count()).map(Reg).find(|&reg| {
        matches!(
            prog.get(reg).unwrap(),
            Insn::Ancestral {
                anc_name: x86_to_mil::ANC_RSP,
                ..
            }
        )
    })
}

#[cfg(test)]
mod tests {
    use crate::{
        mil::{self, Control, R},
        ssa, ty,
        util::Bytes,
    };
    use mil::{ArithOp, Insn, Reg};

    use super::canonical;

    mod constant_folding {
        use crate::{
            mil::{self, Control},
            ssa, ty, xform,
        };
        use mil::{ArithOp, Insn, Reg};

        #[test]
        fn addk() {
            let mut prog = mil::Program::new(Reg(0));
            prog.push(
                Reg(0),
                Insn::Ancestral {
                    anc_name: mil::ANC_STACK_BOTTOM,
                    reg_type: mil::RegType::Bytes(8),
                },
            );
            prog.push(Reg(1), Insn::Int { value: 5, size: 8 });
            prog.push(Reg(2), Insn::Int { value: 44, size: 8 });
            prog.push(Reg(0), Insn::Arith(ArithOp::Add, Reg(1), Reg(0)));
            prog.push(Reg(3), Insn::Arith(ArithOp::Add, Reg(0), Reg(1)));
            prog.push(Reg(4), Insn::Arith(ArithOp::Add, Reg(2), Reg(1)));
            prog.push(
                Reg(0),
                Insn::StoreMem {
                    addr: Reg(4),
                    value: Reg(3),
                },
            );
            prog.push(Reg(3), Insn::Int { value: 0, size: 8 });
            prog.push(
                Reg(4),
                Insn::Ancestral {
                    anc_name: mil::ANC_STACK_BOTTOM,
                    reg_type: mil::RegType::Bytes(8),
                },
            );
            prog.push(Reg(3), Insn::Arith(ArithOp::Add, Reg(3), Reg(4)));
            prog.push(Reg(0), Insn::SetReturnValue(Reg(3)));
            prog.push(Reg(0), Insn::Control(Control::Ret));

            let mut prog = ssa::Program::from_mil(prog);
            xform::canonical(&mut prog, &ty::TypeSet::new());

            assert_eq!(prog.cfg().block_count(), 1);
            assert_eq!(
                prog.get(Reg(4)).unwrap(),
                Insn::ArithK(ArithOp::Add, Reg(0), 10)
            );
            assert_eq!(prog.get(Reg(5)).unwrap(), Insn::Int { value: 49, size: 8 });
            assert_eq!(
                prog.get(Reg(8)).unwrap(),
                Insn::Ancestral {
                    anc_name: mil::ANC_STACK_BOTTOM,
                    reg_type: mil::RegType::Bytes(8),
                }
            );
            assert_eq!(prog.get(Reg(10)).unwrap(), Insn::SetReturnValue(Reg(8)));
        }

        #[test]
        fn mulk() {
            let mut prog = mil::Program::new(Reg(0));
            prog.push(
                Reg(0),
                Insn::Ancestral {
                    anc_name: mil::ANC_STACK_BOTTOM,
                    reg_type: mil::RegType::Bytes(8),
                },
            );
            prog.push(Reg(1), Insn::Int { value: 5, size: 8 });
            prog.push(Reg(2), Insn::Int { value: 44, size: 8 });
            prog.push(Reg(0), Insn::Arith(ArithOp::Mul, Reg(1), Reg(0)));
            prog.push(Reg(3), Insn::Arith(ArithOp::Mul, Reg(0), Reg(1)));
            prog.push(Reg(4), Insn::Arith(ArithOp::Mul, Reg(2), Reg(3)));
            prog.push(Reg(3), Insn::Int { value: 1, size: 8 });
            prog.push(
                Reg(0),
                Insn::StoreMem {
                    addr: Reg(3),
                    value: Reg(4),
                },
            );
            prog.push(
                Reg(4),
                Insn::Ancestral {
                    anc_name: mil::ANC_STACK_BOTTOM,
                    reg_type: mil::RegType::Bytes(8),
                },
            );
            prog.push(Reg(4), Insn::Arith(ArithOp::Mul, Reg(3), Reg(4)));
            prog.push(Reg(0), Insn::SetReturnValue(Reg(4)));
            prog.push(Reg(0), Insn::Control(Control::Ret));

            let mut prog = ssa::Program::from_mil(prog);
            xform::canonical(&mut prog, &ty::TypeSet::new());
            ssa::eliminate_dead_code(&mut prog);

            assert_eq!(prog.insns_rpo().count(), 6);
            assert_eq!(
                prog.get(Reg(5)).unwrap(),
                Insn::ArithK(ArithOp::Mul, Reg(0), 1100)
            );
            assert_eq!(prog.get(Reg(10)).unwrap(), Insn::SetReturnValue(Reg(8)));
        }
    }

    mod subreg_folding {
        use crate::{
            mil::{self, Control},
            ssa, ty, xform,
        };

        use test_log::test;

        define_ancestral_name!(ANC_A, "A");
        define_ancestral_name!(ANC_B, "B");

        #[test_log::test]
        fn part_of_concat() {
            use mil::{Insn, Reg};

            #[derive(Clone, Copy, Debug)]
            struct VariantParams {
                anc_a_sz: u16,
                anc_b_sz: u16,
                offset: u16,
                size: u16,
            }
            fn gen_prog(vp: VariantParams) -> mil::Program {
                let mut p = mil::Program::new(Reg(0));
                p.push(
                    Reg(0),
                    Insn::Ancestral {
                        anc_name: ANC_A,
                        reg_type: mil::RegType::Bytes(vp.anc_a_sz as _),
                    },
                );
                p.push(
                    Reg(1),
                    Insn::Ancestral {
                        anc_name: ANC_B,
                        reg_type: mil::RegType::Bytes(vp.anc_b_sz as _),
                    },
                );
                p.push(
                    Reg(2),
                    Insn::Concat {
                        lo: Reg(0),
                        hi: Reg(1),
                    },
                );
                p.push(
                    Reg(3),
                    Insn::Part {
                        src: Reg(2),
                        offset: vp.offset,
                        size: vp.size,
                    },
                );
                p.push(Reg(0), Insn::SetReturnValue(Reg(3)));
                p.push(Reg(0), Insn::Control(Control::Ret));
                p
            }

            let types = ty::TypeSet::new();

            for anc_a_sz in 1..=7 {
                for anc_b_sz in 1..=(8 - anc_a_sz) {
                    let concat_sz = anc_a_sz + anc_b_sz;

                    // case: fall within lo
                    for offset in 0..=(anc_a_sz - 1) {
                        for size in 1..=(anc_a_sz - offset) {
                            let variant_params = VariantParams {
                                anc_a_sz,
                                anc_b_sz,
                                offset,
                                size,
                            };
                            let prog = gen_prog(variant_params);
                            let mut prog = ssa::Program::from_mil(prog);
                            xform::canonical(&mut prog, &types);

                            assert_eq!(
                                prog.get(Reg(3)).unwrap(),
                                if offset == 0 && size == anc_a_sz {
                                    Insn::Get(Reg(0))
                                } else {
                                    Insn::Part {
                                        src: Reg(0),
                                        offset,
                                        size,
                                    }
                                }
                            );
                        }
                    }

                    // case: fall within hi
                    for offset in anc_a_sz..concat_sz {
                        for size in 1..=(concat_sz - offset) {
                            let prog = gen_prog(VariantParams {
                                anc_a_sz,
                                anc_b_sz,
                                offset,
                                size,
                            });
                            let mut prog = ssa::Program::from_mil(prog);
                            xform::canonical(&mut prog, &types);

                            assert_eq!(
                                prog.get(Reg(3)).unwrap(),
                                if offset == anc_a_sz && size == anc_b_sz {
                                    Insn::Get(Reg(1))
                                } else {
                                    Insn::Part {
                                        src: Reg(1),
                                        offset: offset - anc_a_sz,
                                        size,
                                    }
                                }
                            );
                        }
                    }

                    // case: crossing lo/hi
                    for offset in 0..anc_a_sz {
                        for end in (anc_a_sz + 1)..concat_sz {
                            let size = end - offset;
                            if size == 0 {
                                continue;
                            }

                            dbg!((anc_a_sz, anc_b_sz, offset, size));

                            let prog = gen_prog(VariantParams {
                                anc_a_sz,
                                anc_b_sz,
                                offset,
                                size,
                            });
                            let mut prog = ssa::Program::from_mil(prog);
                            let orig_insn = prog.get(Reg(3)).unwrap();

                            xform::canonical(&mut prog, &types);
                            assert_eq!(prog.get(Reg(3)).unwrap(), orig_insn);
                        }
                    }
                }
            }
        }

        #[test]
        fn part_of_part() {
            use mil::{Insn, Reg};

            #[derive(Clone, Copy)]
            struct VariantParams {
                src_sz: u16,
                offs0: u16,
                size0: u16,
                offs1: u16,
                size1: u16,
            }

            fn gen_prog(vp: VariantParams) -> mil::Program {
                let mut p = mil::Program::new(Reg(10_000));
                p.push(
                    Reg(0),
                    Insn::Ancestral {
                        anc_name: ANC_A,
                        reg_type: mil::RegType::Bytes(vp.src_sz as _),
                    },
                );
                p.push(
                    Reg(1),
                    Insn::Part {
                        src: Reg(0),
                        offset: vp.offs0,
                        size: vp.size0,
                    },
                );
                p.push(
                    Reg(2),
                    Insn::Part {
                        src: Reg(1),
                        offset: vp.offs1,
                        size: vp.size1,
                    },
                );
                p.push(Reg(0), Insn::SetReturnValue(Reg(2)));
                p.push(Reg(0), Insn::Control(Control::Ret));
                p
            }

            let types = ty::TypeSet::new();
            let sample_data = b"12345678";

            for src_sz in 1..=8 {
                for offs0 in 0..src_sz {
                    for size0 in 1..=(src_sz - offs0) {
                        for offs1 in 0..size0 {
                            for size1 in 1..=(size0 - offs1) {
                                let prog = gen_prog(VariantParams {
                                    src_sz,
                                    offs0,
                                    size0,
                                    offs1,
                                    size1,
                                });
                                let mut prog = ssa::Program::from_mil(prog);
                                xform::canonical(&mut prog, &types);

                                let exp_offset = offs0 + offs1;
                                let exp_size = size1;
                                assert_eq!(
                                    prog.get(Reg(2)).unwrap(),
                                    if offs1 == 0 && size1 == src_sz {
                                        Insn::Get(Reg(0))
                                    } else {
                                        Insn::Part {
                                            src: Reg(0),
                                            offset: exp_offset,
                                            size: exp_size,
                                        }
                                    }
                                );

                                let offs0 = offs0 as usize;
                                let size0 = size0 as usize;
                                let offs1 = offs1 as usize;
                                let size1 = size1 as usize;
                                let exp_offset = exp_offset as usize;
                                let exp_size = exp_size as usize;
                                let range0 = offs0..offs0 + size0;
                                let range1 = offs1..offs1 + size1;
                                let exp_range = exp_offset..exp_offset + exp_size;
                                assert_eq!(&sample_data[range0][range1], &sample_data[exp_range]);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn combined_with_fold_get() {
        // check that a transform "sees through" the Insn::Get introduced by an
        // earlier transform

        let mut prog = mil::Program::new(Reg(0));
        prog.push(Reg(1), Insn::Int { value: 5, size: 8 });
        prog.push(Reg(2), Insn::Int { value: 44, size: 8 });

        // removed by fold_bitops
        prog.push(Reg(1), Insn::Arith(ArithOp::BitAnd, Reg(1), Reg(1)));
        prog.push(Reg(2), Insn::Arith(ArithOp::BitAnd, Reg(2), Reg(2)));

        // removed by fold_constants IF the Insn::Get's added by fold_bitops
        // is dereferenced
        prog.push(Reg(0), Insn::Arith(ArithOp::Mul, Reg(1), Reg(2)));
        prog.push(Reg(0), Insn::SetReturnValue(Reg(0)));
        prog.push(Reg(0), Insn::Control(Control::Ret));

        let mut prog = ssa::Program::from_mil(prog);
        super::canonical(&mut prog, &ty::TypeSet::new());
        ssa::eliminate_dead_code(&mut prog);

        assert_eq!(prog.insns_rpo().count(), 2);
        assert_eq!(
            prog.get(Reg(4)).unwrap(),
            Insn::Int {
                value: 5 * 44,
                size: 8
            }
        );
    }

    #[test]
    fn add_cast_float() {
        let a_float_value: f64 = 3.14159;
        let float_as_bytes = Bytes::from_slice(a_float_value.to_le_bytes().as_slice()).unwrap();

        let mut types = ty::TypeSet::new();
        const TYID_F64: ty::TypeID = ty::TypeID(123);
        types.set(TYID_F64, ty::Ty::Float(ty::Float { size: 8 }));

        let mut prog = mil::Program::new(Reg(0));
        prog.push(Reg(0), Insn::Bytes(float_as_bytes));

        let mut prog = ssa::Program::from_mil(prog);
        let mut prog = ssa::OpenProgram::wrap(&mut prog);
        prog.set_value_type(Reg(0), Some(TYID_F64));

        let bid = prog.cfg().entry_block_id();
        prog.clear_block_schedule(bid);
        let orig_insn = prog.get(Reg(0)).unwrap();
        let insn = super::add_cast_for_type(orig_insn, R(0), &mut prog, bid, &types);

        let Insn::ReinterpretFloat64(src) = insn else {
            panic!("wrong insn: {insn:?}")
        };
        assert_eq!(prog.get(src).unwrap(), orig_insn);
    }
}
