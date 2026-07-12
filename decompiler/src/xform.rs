use std::collections::HashMap;

use tracing::{event, span, Level};

use crate::{
    cfg::BlockID,
    mil::{ArithOp, Endianness, Insn, LLType, Reg, StructMemberValue},
    ssa, ty,
    util::Bytes,
    x86_to_mil,
};

mod mem;
mod tests;

struct Cursor<'a: 'b, 'b, 'c> {
    reg: Reg,
    prog: &'b mut ssa::OpenProgram<'a>,
    bid: BlockID,
    types: ty::ReadTxRef<'c>,
}

trait Transform {
    fn apply(&self, cursor: &mut Cursor);
}

struct FoldConstants;

impl Transform for FoldConstants {
    fn apply(&self, cursor: &mut Cursor) {
        use crate::mil::ArithOp;

        let reg = cursor.reg;
        let bid = cursor.bid;

        /// Evaluate expression (ak (op) bk)
        ///
        /// Overflow/underflow results in `None`.
        fn eval_const(op: ArithOp, ak: i64, bk: i64) -> Option<i64> {
            match op {
                ArithOp::Add => ak.checked_add(bk),
                ArithOp::Sub => ak.checked_sub(bk),
                ArithOp::Mul => ak.checked_mul(bk),
                ArithOp::Shl => {
                    let bk = bk.try_into().ok()?;
                    ak.checked_shl(bk)
                }
                ArithOp::Shr | ArithOp::Sar => {
                    let bk = bk.try_into().ok()?;
                    // Both logical (Shr) and arithmetic (Sar) right shift use
                    // i64::checked_shr, which is sign-extending on signed types.
                    // The MIL-level distinction between them affects downstream
                    // type inference (unsigned vs signed context), not the bit
                    // pattern at the i64 constant-folding level.
                    ak.checked_shr(bk)
                }
                ArithOp::Rol => {
                    let bk: u32 = bk.try_into().ok()?;
                    Some(((ak as u64).rotate_left(bk)) as i64)
                }
                ArithOp::Ror => {
                    let bk: u32 = bk.try_into().ok()?;
                    Some(((ak as u64).rotate_right(bk)) as i64)
                }
                ArithOp::BitXor => Some(ak ^ bk),
                ArithOp::BitAnd => Some(ak & bk),
                ArithOp::BitOr => Some(ak | bk),
                ArithOp::DivU => (ak as u64).checked_div(bk as u64).map(|v| v as i64),
                ArithOp::DivS => ak.checked_div(bk),
                ArithOp::ModU => (ak as u64).checked_rem(bk as u64).map(|v| v as i64),
                ArithOp::ModS => ak.checked_rem(bk),
            }
        }

        /// Compute rk such that, for all x:
        ///   (x <op> ak) <op> bk <===> x <op> rk
        /// or, equivalently:
        ///   (x <op> ak) <op> (y <op> bk)<===> (x <op> y) <op> rk
        ///
        /// Returns None for non-associative operators.
        fn assoc_const(op_in: ArithOp, ak: i64, bk: i64) -> Option<i64> {
            // arithmetic overflow seems to happen in practice
            match op_in {
                ArithOp::Add => ak.checked_add(bk),
                ArithOp::Mul => ak.checked_mul(bk),
                ArithOp::Shl => ak.checked_add(bk),
                // Rotation is associative: (x rol A) rol B = x rol (A + B)
                ArithOp::Rol | ArithOp::Ror => ak.checked_add(bk),
                _ => None,
            }
        }

        /*
            \math-container{\frac{a𝛽,𝛽a|a\index{𝛽}}

            \text{Convert sub to add, to get more associative nodes:}
            \frac{a\power-index{𝛽|-}|a\power-index{+|-𝛽}},\frac{a-b\power-index{+|𝛾}|a+b\power-index{+|-𝛾}}

            \text{if op is associative: }\frac{a\index{0}|a}
            [⋅=+]⟹\frac{a\index{0}|a}
            [⋅=×]⟹\frac{a\index{1}|a}}
        */

        let orig_llt = cursor.prog.ll_type(reg);
        let insn = cursor.prog.get(reg).unwrap().clone();
        let (mut op, mut lr, mut li, mut ri) = match &insn {
            Insn::Arith(op, lr, rr) => {
                let li = cursor.prog.get(*lr).unwrap().clone();
                let ri = cursor.prog.get(*rr).unwrap().clone();

                // ensure the const is on the right
                if let Insn::Int { .. } = li {
                    (*op, *rr, ri, li)
                } else {
                    (*op, *lr, li, ri)
                }
            }
            Insn::ArithK(op, a, bk) => (
                *op,
                *a,
                cursor.prog.get(*a).unwrap().clone(),
                Insn::Int {
                    value: *bk,
                    size: 8,
                },
            ),
            _ => return,
        };

        // if there is a Const, it's on the right (or they are both Const)
        assert!(matches!(ri, Insn::Int { .. }) || !matches!(li, Insn::Int { .. }));

        // convert sub to add to increase the probability of applying the following rules
        if op == ArithOp::Sub {
            if let Insn::Int { value, .. } = &mut ri {
                op = ArithOp::Add;
                let Some(value_pos) = value.checked_neg() else {
                    return;
                };
                *value = value_pos;
            } else if let Insn::ArithK(ArithOp::Add, _, r_k) = &mut ri {
                op = ArithOp::Add;
                let Some(value_pos) = r_k.checked_neg() else {
                    return;
                };
                *r_k = value_pos;
            }
        }

        match (li.clone(), ri.clone()) {
            // (a op ka) op (b op kb) === (a op b) op (ka op kb)  (if op is associative)
            (Insn::ArithK(l_op, llr, lk), Insn::ArithK(r_op, rlr, rk))
                if l_op == r_op && l_op == op =>
            {
                if let Some(k) = assoc_const(op, lk, rk) {
                    let new_insn = Insn::Arith(op, llr, rlr);
                    let new_reg = cursor.prog.append_new(bid, new_insn);
                    cursor.reg = new_reg;
                    self.apply(cursor);

                    lr = new_reg;
                    ri = Insn::Int { value: k, size: 8 };
                }
            }
            // (a op ka) op kb === a op (ka op kb)  (if op is associative)
            (Insn::ArithK(l_op, llr, lk), Insn::Int { value: rk, .. }) if l_op == op => {
                if let Some(k) = assoc_const(op, lk, rk) {
                    li = cursor.prog.get(llr).unwrap().clone();
                    lr = llr;
                    ri = Insn::Int { value: k, size: 8 };
                }
            }
            _ => {}
        }

        let result_insn = match (op, li.clone(), ri.clone()) {
            (op, Insn::Int { value: ka, .. }, Insn::Int { value: kb, .. }) => {
                let size = orig_llt.bytes_size().and_then(|sz| sz.try_into().ok());
                match (size, eval_const(op, ka, kb)) {
                    (Some(size), Some(value)) => Insn::Int { value, size },
                    _ => insn,
                }
            }
            (ArithOp::Add, _, Insn::Int { value: 0, .. }) => Insn::Get(lr),
            (ArithOp::Mul, _, Insn::Int { value: 1, .. }) => Insn::Get(lr),

            (op, _, Insn::Int { value: kr, .. }) => Insn::ArithK(op, lr, kr),
            _ => {
                // dang it, we couldn't hack it
                insn
            }
        };

        cursor.prog.set(reg, result_insn);
    }
}

struct FoldSubregs;

impl Transform for FoldSubregs {
    fn apply(&self, c: &mut Cursor) {
        // operators that matter here are:
        // - subrange: src[a..b]
        //      b > a; b <= 8; a, b >= 0
        // - concatenation: hi :: lo
        //
        // two optimizations in one, where the argument may "skip over" the Concat,
        // possibly shifting the range:
        // - Part(Concat(...), ...)
        // - Part(Part(...), ...)

        let insn = c.prog.get(c.reg).unwrap().clone();
        let Insn::Part { src, offset, size } = insn else {
            return;
        };

        let end = offset + size;

        let Some(src_sz) = c.prog.ll_type(src).bytes_size() else {
            return;
        };
        if end as usize > src_sz {
            event!(
                Level::ERROR,
                end,
                src_sz,
                ?insn,
                "insn invalid: end > src_sz"
            );
            return;
        }

        let src_insn = c.prog.get(src).unwrap().clone();
        let result_insn = match src_insn {
            Insn::Part {
                src: up_src,
                offset: up_offset,
                size: up_size,
            } => {
                let up_end = up_offset + up_size;
                let Some(up_src_sz) = c.prog.ll_type(up_src).bytes_size() else {
                    event!(
                        Level::ERROR,
                        ?insn,
                        ?src_insn,
                        ?up_src,
                        "bug in SSA code. up_src has no size",
                    );
                    return;
                };

                let up_insn = c.prog.get(up_src).unwrap().clone();
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
                let lo_size = c
                    .prog
                    .ll_type(lo)
                    .bytes_size()
                    .unwrap()
                    .try_into()
                    .expect("size is too large for Concat");

                if end <= lo_size {
                    // offset..size falls entirely within lo
                    Insn::Part {
                        src: lo,
                        offset,
                        size,
                    }
                } else if offset >= lo_size {
                    // offset..size falls entirely within hi
                    Insn::Part {
                        src: hi,
                        offset: offset - lo_size,
                        size,
                    }
                } else {
                    // offset..size covers (at least part of) both lo and hi
                    insn
                }
            }

            _ => insn,
        };

        c.prog.set(c.reg, result_insn);
    }
}

struct FoldConcatVoid;

impl Transform for FoldConcatVoid {
    fn apply(&self, c: &mut Cursor) {
        let insn = c.prog.get(c.reg).unwrap().clone();
        let Insn::Concat { lo, hi } = insn else {
            return;
        };

        let result_insn = match (c.prog.ll_type(lo), c.prog.ll_type(hi)) {
            (LLType::Bytes(0), LLType::Bytes(0)) => Insn::Void,
            (LLType::Bytes(0), _) => Insn::Get(hi),
            (_, LLType::Bytes(0)) => Insn::Get(lo),
            (_, _) => insn,
        };

        c.prog.set(c.reg, result_insn);
    }
}

struct FoldBitops;

impl Transform for FoldBitops {
    fn apply(&self, cursor: &mut Cursor) {
        let insn = cursor.prog.get(cursor.reg).unwrap().clone();
        let result_insn = match insn {
            // TODO put the appropriate size
            Insn::Arith(ArithOp::BitXor, a, b) if a == b => {
                let size = cursor
                    .prog
                    .ll_type(a)
                    .bytes_size()
                    .unwrap()
                    .try_into()
                    .unwrap();
                if size <= 8 {
                    Insn::Int { value: 0, size }
                } else {
                    insn
                }
            }
            Insn::Arith(ArithOp::BitAnd, a, b) if a == b => Insn::Get(a),
            Insn::Arith(ArithOp::BitOr, a, b) if a == b => Insn::Get(a),
            _ => insn,
        };

        cursor.prog.set(cursor.reg, result_insn);
    }
}

struct FoldPartPart;

impl Transform for FoldPartPart {
    fn apply(&self, c: &mut Cursor) {
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

        let insn = c.prog.get(c.reg).unwrap().clone();
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
            } = c.prog.get(out_src).unwrap().clone()
            {
                assert!(out_size <= in_size);
                let result_insn = Insn::Part {
                    src: in_src,
                    offset: out_offset + in_offset,
                    size: out_size,
                };
                c.prog.set(c.reg, result_insn);
            }
        }
    }
}

struct FoldPartConcat;

impl Transform for FoldPartConcat {
    fn apply(&self, c: &mut Cursor) {
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

        let insn = c.prog.get(c.reg).unwrap().clone();
        if let Insn::Part {
            src: p_src,
            offset: p_offset,
            size: p_size,
        } = insn
        {
            if let Insn::Concat { lo, hi } = c.prog.get(p_src).unwrap().clone() {
                let lo_size = c.prog.ll_type(lo).bytes_size().unwrap().try_into().unwrap();

                let result_insn = if p_offset + p_size <= lo_size {
                    Insn::Part {
                        src: lo,
                        offset: p_offset,
                        size: p_size,
                    }
                } else if p_offset >= lo_size {
                    Insn::Part {
                        src: hi,
                        offset: p_offset - lo_size,
                        size: p_size,
                    }
                } else {
                    insn
                };
                c.prog.set(c.reg, result_insn);
            }
        }
    }
}

struct FoldPartWiden;

impl Transform for FoldPartWiden {
    fn apply(&self, c: &mut Cursor) {
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

        let insn = c.prog.get(c.reg).unwrap().clone();
        let Insn::Part {
            src: part_src,
            offset: 0,
            size: part_size,
        } = insn
        else {
            return;
        };
        let part_src_insn = c.prog.get(part_src).unwrap().clone();

        let Insn::Widen {
            reg: widen_reg,
            target_size: _,
            sign,
        } = part_src_insn
        else {
            return;
        };

        // TODO convert to error
        let Some(orig_size) = c.prog.ll_type(widen_reg).bytes_size() else {
            event!(
                Level::ERROR,
                ?insn,
                ?part_src_insn,
                ?widen_reg,
                "bug. fold_part_widen: reg has no size"
            );
            return;
        };
        let orig_size = orig_size as u16;

        let result_insn = if part_size < orig_size {
            // skip the widen, as it does nothing; directly Part the orig. reg
            Insn::Part {
                src: widen_reg,
                offset: 0,
                size: part_size,
            }
        } else if part_size == orig_size {
            // skip the widen, as it does nothing; use the reg directly,
            // we're not even really Part'ing it
            Insn::Get(widen_reg)
        } else {
            // widen to a smaller size
            Insn::Widen {
                reg: widen_reg,
                target_size: part_size,
                sign,
            }
        };

        c.prog.set(c.reg, result_insn);
    }
}

struct FoldWidenConst;

impl Transform for FoldWidenConst {
    fn apply(&self, c: &mut Cursor) {
        // TODO add signedness to Const as well? then we could check if they match
        let insn = c.prog.get(c.reg).unwrap().clone();
        if let Insn::Widen {
            reg: widen_reg,
            target_size,
            sign: _,
        } = insn
        {
            if target_size <= 8 {
                let widen_insn = c.prog.get(widen_reg).unwrap().clone();
                if let Insn::Int { value, size } = widen_insn {
                    assert!(target_size > size);
                    let result_insn = Insn::Int {
                        value,
                        size: target_size,
                    };
                    c.prog.set(c.reg, result_insn);
                }
            }
        }
    }
}

struct FoldWidenNull;

impl Transform for FoldWidenNull {
    fn apply(&self, c: &mut Cursor) {
        let insn = c.prog.get(c.reg).unwrap().clone();
        if let Insn::Widen {
            reg: widen_reg,
            target_size,
            sign: _,
        } = insn
        {
            if let LLType::Bytes(sz) = c.prog.ll_type(widen_reg) {
                if target_size as usize == sz {
                    let result_insn = Insn::Get(widen_reg);
                    c.prog.set(c.reg, result_insn);
                }
            }
        }
    }
}

struct FoldPartNull;

impl Transform for FoldPartNull {
    fn apply(&self, c: &mut Cursor) {
        let insn = c.prog.get(c.reg).unwrap().clone();
        if let Insn::Part {
            src,
            offset: 0,
            size,
        } = insn
        {
            if let LLType::Bytes(src_size) = c.prog.ll_type(src) {
                if src_size == size as usize {
                    let result_insn = Insn::Get(src);
                    c.prog.set(c.reg, result_insn);
                }
            }
        }
    }
}

struct FoldShrPart;

impl Transform for FoldShrPart {
    fn apply(&self, c: &mut Cursor) {
        let insn = c.prog.get(c.reg).unwrap().clone();
        if let Insn::ArithK(ArithOp::Shr, shr_reg, shift_len) = insn {
            if shift_len >= 0 && shift_len % 8 == 0 {
                if let Insn::Part { src, offset, size } = c.prog.get(shr_reg).unwrap().clone() {
                    let shift_bytes: u16 = (shift_len / 8).try_into().unwrap();
                    let smaller_part = c.prog.append_new(
                        c.bid,
                        Insn::Part {
                            src,
                            offset: offset + shift_bytes,
                            size: size - shift_bytes,
                        },
                    );

                    // widen back to original size
                    let result_insn = Insn::Widen {
                        reg: smaller_part,
                        target_size: size,
                        sign: false,
                    };
                    c.prog.set(c.reg, result_insn);
                }
            }
        }
    }
}

struct FoldGet;

impl Transform for FoldGet {
    fn apply(&self, c: &mut Cursor) {
        let mut insn = c.prog.get(c.reg).unwrap().clone();
        for input in insn.input_regs_iter_mut() {
            loop {
                let input_def = c.prog.get(*input).unwrap().clone();
                if let Insn::Get(arg) = input_def {
                    *input = arg;
                } else {
                    break;
                }
            }
        }

        c.prog.set(c.reg, insn);
    }
}

struct FoldPartVoid;

impl Transform for FoldPartVoid {
    fn apply(&self, c: &mut Cursor) {
        let insn = c.prog.get(c.reg).unwrap().clone();
        if let Insn::Part { size: 0, .. } = insn {
            c.prog.set(c.reg, Insn::Void);
        }
    }
}

struct FoldPartConst;

impl Transform for FoldPartConst {
    fn apply(&self, c: &mut Cursor) {
        let insn = c.prog.get(c.reg).unwrap().clone();
        if let Insn::Part { src, offset, size } = insn {
            let result_insn = match c.prog.get(src).unwrap() {
                Insn::Int {
                    value: src_value,
                    size: src_size,
                } => {
                    let src_bytes = c.prog.int_bytes(*src_value, *src_size);
                    let src_bytes = src_bytes.as_slice();

                    let offset = offset as usize;
                    let size = size as usize;
                    let end = offset + size;

                    if size > 8 {
                        // too large! can't represent this as a single Insn::Int
                        return;
                    }
                    if end > src_bytes.len() {
                        event!(
                            Level::ERROR,
                            ?insn,
                            src_value,
                            src_size,
                            "bug. Part: part offset/size invalid for source int"
                        );
                        return;
                    }

                    let part_bytes = &src_bytes[offset..end];
                    let value = match c.prog.endianness() {
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
                    Insn::Int {
                        value,
                        size: size.try_into().unwrap(),
                    }
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
                        return;
                    }

                    let offset = offset as usize;
                    let size = size as usize;
                    let part_bytes = &src_bytes.as_slice()[offset..offset + size];

                    Insn::Bytes(Bytes::from_slice(part_bytes).unwrap())
                }
                _ => return,
            };

            c.prog.set(c.reg, result_insn);
        }
    }
}

struct AddNegativeToSub;

impl Transform for AddNegativeToSub {
    fn apply(&self, c: &mut Cursor) {
        let insn = c.prog.get(c.reg).unwrap().clone();
        if let Insn::ArithK(ArithOp::Add, lr, rk) = insn {
            if rk < 0 {
                c.prog.set(c.reg, Insn::ArithK(ArithOp::Sub, lr, -rk));
            }
        }
    }
}

/// If the given instruction is `Insn::Part { src, offset, size }`, and we know
/// the type of `src`, then use the type information to replace it with a chain
/// of instructions that extract the relevant part of `src` (e.g. named struct
/// members, array element access by index).
///
/// Doesn't do anything for other types of instructions or when no type info is
/// available.
struct SelectTypeOnPart;

impl Transform for SelectTypeOnPart {
    fn apply(&self, c: &mut Cursor) {
        let insn = c.prog.get(c.reg).unwrap().clone();
        let Insn::Part { src, offset, size } = insn else {
            return;
        };

        let byte_range = ty::ByteRange::new_offset_size(offset as usize, size as usize);
        let tip = select_type_and_read(c.prog, c.bid, c.types, src, byte_range);
        if tip != src {
            c.prog.set(c.reg, Insn::Get(tip));
        }
    }
}

struct PickCalleeName;

impl Transform for PickCalleeName {
    fn apply(&self, c: &mut Cursor) {
        let insn = c.prog.get(c.reg).unwrap().clone();
        if let Insn::Call { callee, .. } = insn {
            // only do this when the callee is a const int. this is the common case when
            // calling a globally defined address, and NOT when you do an indirect call.
            if let Insn::Int { .. } = c.prog.get(callee).unwrap() {
                if let Some(callee_tyid) = c.prog.value_type(callee) {
                    if let Ok(Some(name)) = c.types.name(callee_tyid) {
                        c.prog.set(callee, Insn::Global(name.leak()));
                    }
                }
            }
        }
    }
}

struct PickGlobalName;

impl Transform for PickGlobalName {
    fn apply(&self, c: &mut Cursor) {
        let insn = c.prog.get(c.reg).unwrap();
        let &Insn::ArithK(ArithOp::Add, base_reg, offset) = insn else {
            return;
        };
        let Insn::Ancestral {
            anc_name: x86_to_mil::ANC_RIP,
            ..
        } = c.prog.get(base_reg).unwrap()
        else {
            return;
        };

        let Some(rip_value) = c.prog.machine_addr(c.reg) else {
            event!(
                Level::DEBUG,
                ?c.reg,
                ?insn,
                "unable to get machine address for RIP-relative access"
            );
            return;
        };

        let global_addr = (rip_value as i64 + offset) as u64;
        let Ok(Some((known_obj, offset))) = c.types.get_known_object(global_addr) else {
            event!(
                Level::DEBUG,
                ?c.reg,
                ?insn,
                global_addr,
                "unable to get known object type for RIP-relative access"
            );
            return;
        };

        let Ok(Some(name)) = c.types.name(known_obj) else {
            event!(
                Level::DEBUG,
                ?c.reg,
                ?insn,
                global_addr,
                ?known_obj,
                "unable to get name for known object type for RIP-relative access"
            );
            return;
        };

        if offset == 0 {
            c.prog.set(c.reg, Insn::Global(name.leak()));
            c.prog.set_value_type(c.reg, Some(known_obj));
        } else {
            let global = c.prog.append_new(c.bid, Insn::Global(name.leak()));
            c.prog.set_value_type(global, Some(known_obj));
            c.prog
                .set(c.reg, Insn::ArithK(ArithOp::Add, global, offset));
        }
    }
}

fn select_type_and_read(
    prog: &mut ssa::OpenProgram<'_>,
    bid: BlockID,
    types: ty::ReadTxRef<'_>,
    src: Reg,
    byte_range: ty::ByteRange,
) -> Reg {
    let Some((tip_tyid, path)) = select_type(prog, types, src, byte_range) else {
        return src;
    };

    // TypeSet::select returns an empty path when the required byte range
    // precisely matches src

    // in the most generic case, it contains a single SelectStep::RawBytes
    let mut tip = src;
    for step in path {
        // add insns for intermediate steps
        tip = prog.append_new(bid, step.to_insn(tip));
    }

    // for now, type IDs are not assigned to the intermediate steps
    prog.set_value_type(tip, Some(tip_tyid));
    tip
}

fn select_type(
    prog: &mut ssa::OpenProgram<'_>,
    types: ty::ReadTxRef<'_>,
    src: Reg,
    byte_range: ty::ByteRange,
) -> Option<(ty::TypeID, smallvec::SmallVec<[ty::SelectStep; 4]>)> {
    let Some(src_tyid) = prog.value_type(src) else {
        event!(
            Level::DEBUG,
            ?src,
            "source has no type; skipping type selection"
        );
        return None;
    };
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
            return None;
        }
    };
    event!(Level::TRACE, ?src_tyid, ?byte_range, "type selected");
    Some((selected_tyid, path))
}

struct SelectTypeOnDerefMemberRead;

impl Transform for SelectTypeOnDerefMemberRead {
    fn apply(&self, c: &mut Cursor) {
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

        let insn = c.prog.get(c.reg).unwrap().clone();
        let Insn::LoadMem {
            addr: load_addr,
            size: load_size,
        } = insn
        else {
            return;
        };

        let Some(deref_access) = select_type_deref(c.prog, c.types, load_addr, load_size) else {
            return;
        };

        // Load the aggregate instead of the member, and then do the part access
        let reg_aggregate = c.prog.append_new(
            c.bid,
            Insn::LoadMem {
                addr: deref_access.aggregate_addr_reg,
                size: deref_access.aggregate_size,
            },
        );
        c.prog
            .set_value_type(reg_aggregate, Some(deref_access.aggregate_tyid));

        // find the struct member that matches the offset/size
        let tip = select_type_and_read(
            c.prog,
            c.bid,
            c.types,
            reg_aggregate,
            deref_access.member_range,
        );
        if tip != reg_aggregate {
            c.prog.set(c.reg, Insn::Get(tip));
        }
    }
}

#[derive(Debug)]
struct DerefAccess {
    aggregate_addr_reg: Reg,
    aggregate_size: u32,
    aggregate_tyid: ty::TypeID,
    member_range: ty::ByteRange,
}

fn select_type_deref(
    prog: &ssa::Program,
    types: ty::ReadTxRef<'_>,
    reg_addr: Reg,
    size: u32,
) -> Option<DerefAccess> {
    let (aggregate_addr_reg, offset) = match prog.get(reg_addr).unwrap() {
        &Insn::ArithK(ArithOp::Add, reg_base, offset) => (reg_base, offset),
        // assume the access is directly to the whole
        _ => (reg_addr, 0),
    };

    if offset < 0 {
        return None;
    }

    // let's see if we can actually characterize this as a member access via deref...
    let member_range = {
        let offset = offset.try_into().unwrap();
        let size = size.try_into().unwrap();
        ty::ByteRange::new_offset_size(offset, size)
    };
    event!(
        Level::TRACE,
        ?aggregate_addr_reg,
        ?member_range,
        "pointee range read detected"
    );

    let aggregate_ptr_tyid = prog.value_type(aggregate_addr_reg)?;
    let aggregate_ptr_ty = types.get_through_alias(aggregate_ptr_tyid).ok().flatten()?;
    let &ty::Ty::Ptr(aggregate_tyid) = aggregate_ptr_ty.as_ref() else {
        return None;
    };
    let aggregate_size = types.bytes_size(aggregate_tyid).ok().flatten()?;

    let deref_access = DerefAccess {
        aggregate_addr_reg,
        aggregate_size: aggregate_size.try_into().unwrap(),
        aggregate_tyid,
        member_range,
    };
    event!(Level::TRACE, ?deref_access, "deref access selected");
    Some(deref_access)
}

struct LabeledTransform<'a> {
    label: &'static str,
    xform: &'a dyn Transform,
}
struct TransformSet<'a> {
    transforms: &'a [LabeledTransform<'a>],
}
impl<'a> TransformSet<'a> {
    fn apply(&self, cursor: &mut Cursor) {
        for labeled in self.transforms.iter() {
            span!(Level::TRACE, "applying transform", ?labeled.label).in_scope(|| {
                labeled.xform.apply(cursor);
            });
        }
    }
}

/// Perform the standard chain of transformations that we intend to generally apply to programs
pub fn canonical(prog: &mut ssa::Program, types: &ty::TypeSet) {
    let mut transforms: Vec<LabeledTransform> = Vec::new();

    let mut mem_fold_load_store =
        find_mem_ref(prog).map(|ref_reg| mem::FoldLoadStoreReg { ref_reg });
    if let Some(xform) = &mut mem_fold_load_store {
        transforms.push(LabeledTransform {
            label: "FoldLoadStoreReg",
            xform,
        });
    }

    macro_rules! push_xform {
        ($xform:ident) => {
            let mut step = $xform;
            transforms.push(LabeledTransform {
                label: stringify!($xform),
                xform: &mut step,
            });
        };
    }
    push_xform!(PropagateCallTypes);
    push_xform!(PackAggregates);
    push_xform!(FoldGet);
    push_xform!(FoldSubregs);
    push_xform!(FoldConcatVoid);
    push_xform!(FoldPartPart);
    push_xform!(FoldPartWiden);
    push_xform!(FoldPartConcat);
    push_xform!(FoldPartNull);
    push_xform!(FoldPartVoid);
    push_xform!(FoldPartConst);
    push_xform!(FoldWidenNull);
    push_xform!(FoldWidenConst);
    push_xform!(FoldBitops);
    push_xform!(FoldConstants);
    push_xform!(FoldShrPart);
    push_xform!(SelectTypeOnDerefMemberRead);
    push_xform!(SelectTypeOnPart);
    push_xform!(PickGlobalName);
    push_xform!(PickCalleeName);

    let xformset = TransformSet {
        transforms: &mut transforms[..],
    };
    peephole(prog, types, &xformset);

    // apply some transfomrs at the end, for "cosmetic purposes" or readability
    transforms.clear();
    push_xform!(AddNegativeToSub);
    let xformset = TransformSet {
        transforms: &mut transforms[..],
    };
    peephole(prog, types, &xformset);
}

/// Perform the standard chain of transformations that we intend to generally apply to programs
#[tracing::instrument(skip_all)]
fn peephole(prog: &mut ssa::Program, types: &ty::TypeSet, xformset: &TransformSet) {
    prog.check_invariants();

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

    prog.mutate(|mut prog| {
        // TODO propagate this error
        let rtx = types.read_tx().unwrap();
        let bids: Vec<_> = prog.cfg().block_ids_rpo().collect();
        for bid in bids {
            // the deduper is block-local
            let mut deduper = Deduper::new();

            // the block is passed 3 times so that:
            for _ in 0..3 {
                // clear the block's schedule, then reconstruct it.
                // existing instruction are processed and replaced to keep using the memory they already occupy
                for reg in prog.clear_block_schedule(bid) {
                    let span = tracing::span!(Level::TRACE, "processing", ?reg);
                    let _enter = span.enter();

                    let orig_has_fx = prog.get(reg).unwrap().has_side_effects();

                    let mut cursor = Cursor {
                        reg,
                        prog: &mut prog,
                        bid,
                        types: rtx.read(),
                    };
                    xformset.apply(&mut cursor);
                    deduper.try_dedup(&mut cursor);
                    cursor.prog.append_existing(cursor.bid, cursor.reg);

                    assert_eq!(
                        cursor.prog.get(cursor.reg).unwrap().has_side_effects(),
                        orig_has_fx
                    );
                }
            }
        }

        event!(Level::TRACE, prog = ? &*prog, "ssa after xform cycle");
    });
}

struct PropagateCallTypes;

impl Transform for PropagateCallTypes {
    fn apply(&self, c: &mut Cursor) {
        // clone makes me a bit sad but it makes the code much cleaner. if this
        // becomes a bottleneck we can always optimize it later.
        let Insn::Call {
            callee,
            args,
            ret_ll_type: _,
        } = c.prog.get(c.reg).unwrap().clone()
        else {
            return;
        };

        let Some(callee_tyid) = c.prog.value_type(callee) else {
            return;
        };
        let Some(callee_ty) = c.types.get_through_alias(callee_tyid).unwrap() else {
            event!(
                Level::ERROR,
                ?callee,
                ?callee_tyid,
                "callee's type ID is invalid"
            );
            return;
        };
        let ty::Ty::Subroutine(subr_ty) = callee_ty.as_ref() else {
            event!(
                Level::ERROR,
                ?callee,
                ?callee_tyid,
                "callee's type not a subroutine"
            );
            return;
        };

        if c.prog.value_type(c.reg).is_none() {
            c.prog.set_value_type(c.reg, Some(subr_ty.return_tyid));
        }

        let param_count = subr_ty.param_tyids.len();
        let mut arg_count = 0;
        for (arg, arg_tyid) in args.iter().copied().zip(&subr_ty.param_tyids) {
            if c.prog.value_type(arg).is_none() {
                c.prog.set_value_type(arg, Some(*arg_tyid));
            }
            arg_count += 1;
        }

        if arg_count != param_count {
            event!(
                Level::ERROR,
                ?c.reg,
                ?callee,
                ?arg_count,
                ?param_count,
                "call site has fewer arguments than subroutine type has parameters"
            );
        }
    }
}

struct PackAggregates;

impl Transform for PackAggregates {
    fn apply(&self, c: &mut Cursor) {
        let Some(tyid) = c.prog.value_type(c.reg) else {
            return;
        };
        let Some(tycow) = c.types.get_through_alias(tyid).unwrap() else {
            return;
        };

        if let ty::Ty::Struct(struct_ty) = tycow.as_ref() {
            let insn = c.prog.get(c.reg).unwrap();

            if matches!(insn, Insn::Concat { .. }) {
                // copy the instruction to a new value -- it will act as the "raw"
                // value of the struct, to be formally sectioned into fields and
                // reassembled into a single `Insn::Struct` carrying all members
                // directly.
                let src = c.prog.append_new(c.bid, Insn::Void);
                c.prog.swap(src, c.reg);

                // reborrow
                let insn = c.prog.get(c.reg).unwrap();

                // now: src -> orig insn;  reg -> Insn::Void (which we're going to replace)
                debug_assert_eq!(c.prog.get(src).unwrap(), insn);

                let mut members = Vec::new();

                for member in &struct_ty.members {
                    let Ok(Some(size)) = c.types.bytes_size(member.tyid) else {
                        event!(
                            Level::ERROR,
                            struct_tyid = ?tyid,
                            name = member.name.to_string(),
                            member_tyid = ?member.tyid,
                            "struct member type has no size; skipped"
                        );
                        continue;
                    };
                    // TODO figure out proper memory management for leaked member names.
                    let name = member.name.to_string().leak();
                    let value = c.prog.append_new(
                        c.bid,
                        Insn::Part {
                            src,
                            offset: member.offset.try_into().unwrap(),
                            size: size.try_into().unwrap(),
                        },
                    );
                    c.prog.set_value_type(value, Some(member.tyid));

                    members.push(StructMemberValue { name, value });
                }

                c.prog.set(
                    c.reg,
                    Insn::Struct {
                        type_name: c
                            .types
                            .name(tyid)
                            .unwrap()
                            .map(|s| s.to_string().leak() as &str)
                            .unwrap_or("?"),
                        members,
                        size: struct_ty.size.try_into().unwrap(),
                    },
                );

                // NOTE that, although `src` points to a copy of Insn, it has no
                // type info; the type info is instead kept in the new Insn::Struct,
                // assigned to the original register (agg_reg)
            }
        }
    }
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

    fn try_dedup(&mut self, cursor: &mut Cursor) {
        // replacing a side-effecting instruction with a non-side-effecting
        // Insn::Get is currently wrong (would be quite complicated to handle)
        let orig_insn = cursor.prog.get(cursor.reg).unwrap();
        if !orig_insn.is_replaceable() {
            return;
        }

        match self.rev_lookup.get_mut(orig_insn) {
            Some(&mut prev_reg) if prev_reg != cursor.reg => {
                cursor.prog.set(cursor.reg, Insn::Get(prev_reg));
            }
            _ => {
                self.rev_lookup.insert(orig_insn.clone(), cursor.reg);
            }
        }
    }
}

fn find_mem_ref(prog: &ssa::Program) -> Option<Reg> {
    // TODO make this architecture agnostic
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
