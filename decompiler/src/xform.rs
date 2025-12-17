use std::collections::HashMap;

use tracing::{event, instrument, span, Level};

use crate::{
    cfg::BlockID,
    mil::{ArithOp, Endianness, Insn, Reg, RegType},
    ssa, ty,
    util::Bytes,
    x86_to_mil,
};

mod mem;
mod tests;

fn fold_constants(insn: Insn, prog: &mut ssa::OpenProgram, bid: BlockID) -> Insn {
    use crate::mil::ArithOp;

    /// Evaluate expression (ak (op) bk)
    fn eval_const(op: ArithOp, ak: i64, bk: i64) -> Option<i64> {
        match op {
            ArithOp::Add => Some(ak + bk),
            ArithOp::Sub => Some(ak - bk),
            ArithOp::Mul => Some(ak * bk),
            ArithOp::Shl => {
                let bk = bk.try_into().ok()?;
                ak.checked_shl(bk)
            }
            ArithOp::Shr => {
                let bk = bk.try_into().ok()?;
                ak.checked_shr(bk)
            }
            ArithOp::BitXor => Some(ak ^ bk),
            ArithOp::BitAnd => Some(ak & bk),
            ArithOp::BitOr => Some(ak | bk),
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
        (op, Insn::Int { value: ka, .. }, Insn::Int { value: kb, .. }) => {
            match eval_const(op, ka, kb) {
                Some(value) => Insn::Int { value, size: 8 },
                None => insn,
            }
        }
        (ArithOp::Add, _, Insn::Int { value: 0, .. }) => Insn::Get(lr),
        (ArithOp::Mul, _, Insn::Int { value: 1, .. }) => Insn::Get(lr),

        (op, _, Insn::Int { value: kr, .. }) => Insn::ArithK(op, lr, kr),
        _ => {
            // dang it, we couldn't hack it
            insn
        }
    }
}

#[tracing::instrument(skip_all)]
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

    let Some(src_sz) = prog.reg_type(src).bytes_size() else {
        return insn;
    };
    if end as usize > src_sz {
        event!(
            Level::ERROR,
            end,
            src_sz,
            ?insn,
            "insn invalid: end > src_sz"
        );
        return insn;
    }

    let src_insn = prog.get(src).unwrap();
    match src_insn {
        Insn::Part {
            src: up_src,
            offset: up_offset,
            size: up_size,
        } => {
            let up_end = up_offset + up_size;
            let Some(up_src_sz) = prog.reg_type(up_src).bytes_size() else {
                event!(
                    Level::ERROR,
                    ?insn,
                    ?src_insn,
                    ?up_src,
                    "bug in SSA code. up_src has no size",
                );
                return insn;
            };

            let up_insn = prog.get(up_src).unwrap();
            if up_end as usize > up_src_sz {
                event!(
                    Level::ERROR,
                    ?insn,
                    ?src_insn,
                    ?up_insn,
                    up_end,
                    up_src_sz,
                    "bug in SSA code. up_end > up_src_sz",
                );
            }

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
        Insn::Arith(ArithOp::BitXor, a, b) if a == b => {
            let size = prog.reg_type(a).bytes_size().unwrap().try_into().unwrap();
            if size <= 8 {
                Insn::Int { value: 0, size }
            } else {
                insn
            }
        }
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

    let Insn::Part {
        src: part_src,
        offset: 0,
        size: part_size,
    } = insn
    else {
        return insn;
    };
    let part_src_insn = prog.get(part_src).unwrap();

    let Insn::Widen {
        reg,
        target_size: _,
        sign,
    } = part_src_insn
    else {
        return insn;
    };

    // TODO convert to error
    let Some(orig_size) = prog.reg_type(reg).bytes_size() else {
        event!(
            Level::ERROR,
            ?insn,
            ?part_src_insn,
            ?reg,
            "bug. fold_part_widen: reg has no size"
        );
        return insn;
    };
    let orig_size = orig_size.try_into().unwrap();

    if part_size < orig_size {
        // skip the widen, as it does nothing; directly Part the orig. reg
        Insn::Part {
            src: reg,
            offset: 0,
            size: part_size,
        }
    } else if part_size == orig_size {
        // skip the widen, as it does nothing; use the reg directly,
        // we're not even really Part'ing it
        Insn::Get(reg)
    } else {
        // widen to a smaller size
        Insn::Widen {
            reg,
            target_size: part_size,
            sign,
        }
    }
}

fn fold_widen_const(insn: Insn, prog: &ssa::Program) -> Insn {
    // TODO add signedness to Const as well? then we could check if they match
    if let Insn::Widen {
        reg,
        target_size,
        sign: _,
    } = insn
    {
        if target_size <= 8 {
            if let Insn::Int { value, size } = prog.get(reg).unwrap() {
                assert!(target_size > size);
                return Insn::Int {
                    value,
                    size: target_size,
                };
            }
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
        match prog.get(src).unwrap() {
            Insn::Int {
                value: src_value,
                size: src_size,
            } => {
                let src_bytes = prog.int_bytes(src_value, src_size);
                let src_bytes = src_bytes.as_slice();

                let offset = offset as usize;
                let size = size as usize;
                let end = offset + size;

                if size > 8 {
                    // too large! can't represent this as a single Insn::Int
                    return insn;
                }
                if end > src_bytes.len() {
                    event!(
                        Level::ERROR,
                        ?insn,
                        src_value,
                        src_size,
                        "bug. Part: part offset/size invalid for source int"
                    );
                    return insn;
                }

                let part_bytes = &src_bytes[offset..end];
                let value = match prog.endianness() {
                    Endianness::Little => {
                        let mut result_bytes = [0u8; 8];
                        result_bytes[..size].copy_from_slice(part_bytes);
                        i64::from_le_bytes(result_bytes)
                    }
                    Endianness::Big => {
                        let mut result_bytes = [0u8; 8];
                        result_bytes[8 - size..].copy_from_slice(part_bytes);
                        i64::from_be_bytes(result_bytes)
                    }
                };
                return Insn::Int {
                    value,
                    size: size.try_into().unwrap(),
                };
            }
            Insn::Bytes(src_bytes) => {
                if src_bytes.len() < size as usize {
                    event!(
                        Level::ERROR,
                        ?src,
                        part_size = size,
                        src_size = src_bytes.len(),
                        "bug. Part: part size smaller than source size"
                    );
                    return insn;
                }

                let offset = offset as usize;
                let size = size as usize;
                let part_bytes = &src_bytes.as_slice()[offset..offset + size];

                return Insn::Bytes(Bytes::from_slice(part_bytes).unwrap());
            }
            _ => {}
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
fn select_type_on_part(
    insn: Insn,
    prog: &mut ssa::OpenProgram,
    bid: BlockID,
    types: ty::ReadTxRef,
) -> Insn {
    let Insn::Part { src, offset, size } = insn else {
        return insn;
    };

    let offset = offset as usize;
    let size = size as usize;
    let byte_range = ty::ByteRange::new(offset, offset + size);
    apply_type_selection(insn, prog, bid, types, src, byte_range)
}

fn apply_type_selection(
    insn: Insn,
    prog: &mut ssa::OpenProgram<'_>,
    bid: BlockID,
    types: ty::ReadTxRef<'_>,
    src: Reg,
    byte_range: ty::ByteRange,
) -> Insn {
    // desist if there is no type information for the src value
    let Some(src_tyid) = prog.value_type(src) else {
        event!(
            Level::DEBUG,
            ?src,
            "source has no type; skipping type selection"
        );
        return insn;
    };

    event!(
        Level::DEBUG,
        ?src_tyid,
        ?byte_range,
        "applying type selection"
    );
    let ty::Selection {
        tyid: selected_tyid,
        path,
    } = match types.select(src_tyid, byte_range) {
        Ok(part_tyid) => part_tyid,
        Err(err) => {
            event!(
                Level::INFO,
                ?src_tyid,
                ?byte_range,
                ?err,
                "unable to select in type"
            );
            return insn;
        }
    };

    event!(Level::TRACE, ?byte_range, "type selected");

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

#[tracing::instrument(skip_all)]
fn select_type_on_deref_member_read(
    insn: Insn,
    prog: &mut ssa::OpenProgram,
    bid: BlockID,
    types: ty::ReadTxRef,
) -> Insn {
    // example of pattern:
    //
    //  (for a struct type S)
    //
    //   r_ptr <- ArithK(Add, r_base, offset)  ;; of type Ptr(S)
    //   r_loaded <- LoadMem(r_ptr, size)
    //
    // becomes:
    //
    //  r_struct <- LoadMem(r_base, struct_size)
    //  r_loaded <- Part(r_struct, offset, size)

    let Insn::LoadMem {
        addr: reg_member_addr,
        size: member_size,
    } = insn
    else {
        return insn;
    };

    let (reg_base_ptr, member_offset) = match prog.get(reg_member_addr).unwrap() {
        Insn::ArithK(ArithOp::Add, reg_base_ptr, offset) => (reg_base_ptr, offset),
        _ => (reg_member_addr, 0),
    };

    if member_offset < 0 {
        return insn;
    }
    let member_offset: usize = member_offset.try_into().unwrap();
    let member_size: usize = member_size.try_into().unwrap();

    event!(
        Level::DEBUG,
        ?reg_base_ptr,
        ?member_offset,
        ?member_size,
        "ptr member read detected"
    );

    let Some(base_ptr_tyid) = prog.value_type(reg_base_ptr) else {
        return insn;
    };
    event!(Level::DEBUG, ?base_ptr_tyid, "base ptr type ID");
    let Some(base_ptr_ty) = types.get_through_alias(base_ptr_tyid).unwrap() else {
        return insn;
    };
    event!(Level::DEBUG, ?base_ptr_ty, "base ptr type");
    let &ty::Ty::Ptr(pointee_tyid) = base_ptr_ty.as_ref() else {
        return insn;
    };
    event!(Level::DEBUG, ?pointee_tyid, "pointee type ID");

    // add a LoadMem for the whole pointee,
    let Ok(Some(size)) = types.bytes_size(pointee_tyid) else {
        event!(Level::DEBUG, ?pointee_tyid, "pointee type has no byte size");
        return insn;
    };
    let reg_pointee_whole = prog.append_new(
        bid,
        Insn::LoadMem {
            addr: reg_base_ptr,
            size: size.try_into().unwrap(),
        },
    );
    prog.set_value_type(reg_pointee_whole, Some(pointee_tyid));

    // find the struct member that matches the offset/size
    let member_range = ty::ByteRange::new(member_offset, member_offset + member_size);
    apply_type_selection(insn, prog, bid, types, reg_pointee_whole, member_range)
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

    // TODO propagate this error
    let rtx = types.read_tx().unwrap();

    propagate_call_types(&mut prog, rtx.read());

    let bids: Vec<_> = prog.cfg().block_ids_rpo().collect();
    for bid in bids {
        // the block is passed 3 times so that:
        //
        // - instructions added by any of the transforms have a chance to be
        // seen/processed by the other passes;
        //
        // - the process does continue indefinitely, and instead always
        // terminates

        for _ in 0..3 {
            // clear the block's schedule, then reconstruct it.
            // existing instruction are processed and replaced to keep using the memory they already occupy
            for reg in prog.clear_block_schedule(bid) {
                let span = span!(Level::TRACE, "processing", ?reg);
                let _enter = span.enter();

                let orig_insn = prog.get(reg).unwrap();
                let orig_has_fx = orig_insn.has_side_effects();

                let mut insn = orig_insn;
                if let Some(mem_ref_reg) = mem_ref_reg {
                    insn = mem::fold_load_store(&mut prog, mem_ref_reg, bid, insn);
                }
                insn = pack_aggregates(reg, insn, &mut prog, bid, rtx.read());
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
                insn = select_type_on_deref_member_read(insn, &mut prog, bid, rtx.read());
                insn = select_type_on_part(insn, &mut prog, bid, rtx.read());
                if !insn.is_replaceable_with_get() {
                    // replacing a side-effecting instruction with a non-side-effecting
                    // Insn::Get is currently wrong (would be quite complicated to handle)
                    insn = deduper.try_dedup(reg, insn);
                }
                prog.set(reg, insn);
                prog.append_existing(bid, reg);

                assert_eq!(insn.has_side_effects(), orig_has_fx);

                if insn != orig_insn {
                    event!(Level::TRACE, ?insn, ?orig_insn, "insn set");
                }
            }
        }
    }

    event!(Level::TRACE, prog = ? &*prog, "ssa after xform cycle");
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
fn propagate_call_types<'t>(prog: &mut ssa::OpenProgram, types: ty::ReadTxRef) {
    let mut tyids_to_set = ssa::RegMap::for_program(prog, None);
    for (_, reg) in prog.insns_rpo() {
        let Insn::Call { callee, first_arg } = prog.get(reg).unwrap() else {
            continue;
        };
        let Some(callee_tyid) = prog.value_type(callee) else {
            continue;
        };
        let Some(callee_ty) = types.get_through_alias(callee_tyid).unwrap() else {
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
fn pack_aggregates<'t>(
    reg: Reg,
    mut insn: Insn,
    prog: &mut ssa::OpenProgram,
    bid: BlockID,
    types: ty::ReadTxRef,
) -> Insn {
    let Some(tyid) = prog.value_type(reg) else {
        return insn;
    };
    let Some(tycow) = types.get_through_alias(tyid).unwrap() else {
        return insn;
    };

    if let ty::Ty::Struct(struct_ty) = tycow.as_ref() {
        if matches!(insn, Insn::Concat { .. }) {
            // copy the instruction to a new value -- it will act as the "raw"
            // value of the struct, to be formally sectioned into fields
            let src = prog.append_new(bid, insn);

            let mut first_member_reg = None;

            for member in struct_ty.members.iter().rev() {
                let Ok(Some(size)) = types.bytes_size(member.tyid) else {
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
                    .unwrap()
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
