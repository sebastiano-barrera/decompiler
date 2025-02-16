use crate::{
    mil::{self, ArithOp, Insn},
    ssa,
};

// TODO Fix the algorithm to work with different instruction output sizes.
// NOTE Right now folding is done across instructions of different sizes. It's a known limitation.
fn fold_constants(insn: mil::Insn, prog: &ssa::Program) -> Insn {
    use mil::{ArithOp, Insn, Reg};

    fn widen(prog: &ssa::Program, reg: Reg) -> Insn {
        match prog.get(reg).unwrap().insn.get() {
            Insn::Const1(k) => Insn::Const8(k as i64),
            Insn::Const2(k) => Insn::Const8(k as i64),
            Insn::Const4(k) => Insn::Const8(k as i64),
            Insn::Const8(k) => Insn::Const8(k as i64),

            Insn::Arith1(op, a, b) => Insn::Arith8(op, a, b),
            Insn::Arith2(op, a, b) => Insn::Arith8(op, a, b),
            Insn::Arith4(op, a, b) => Insn::Arith8(op, a, b),
            Insn::Arith8(op, a, b) => Insn::Arith8(op, a, b),

            Insn::ArithK1(op, r, k) => Insn::ArithK8(op, r, k as i64),
            Insn::ArithK2(op, r, k) => Insn::ArithK8(op, r, k as i64),
            Insn::ArithK4(op, r, k) => Insn::ArithK8(op, r, k as i64),
            Insn::ArithK8(op, r, k) => Insn::ArithK8(op, r, k),

            insn => insn,
        }
    }

    /// Evaluate expression (ak (op) bk)
    fn eval_const(op: ArithOp, ak: i64, bk: i64) -> i64 {
        match op {
            ArithOp::Add => ak + bk,
            ArithOp::Sub => ak - bk,
            ArithOp::Mul => ak * bk,
            ArithOp::Shl => ak << bk,
            ArithOp::BitXor => ak ^ bk,
            ArithOp::BitAnd => ak & bk,
            ArithOp::BitOr => ak | bk,
        }
    }

    /// Compute rk such that, for all x:
    ///   (x <op_in> ak) <op_out> bk <===> x <op_res> rk
    ///
    /// Not all operators are supported. For the unsupported ones, None is returned.
    fn assoc_const(op_in: ArithOp, op_out: ArithOp, ak: i64, bk: i64) -> Option<(ArithOp, i64)> {
        match (op_in, op_out) {
            (ArithOp::Add, ArithOp::Add) => Some((ArithOp::Add, (ak + bk))),
            (ArithOp::Sub, ArithOp::Sub) => Some((ArithOp::Sub, (ak + bk))),

            (ArithOp::Sub, ArithOp::Add) => Some((ArithOp::Sub, (ak + bk))),
            (ArithOp::Add, ArithOp::Sub) => Some((ArithOp::Sub, (ak - bk))),

            (ArithOp::Mul, ArithOp::Mul) => Some((ArithOp::Mul, (ak * bk))),
            (ArithOp::Shl, ArithOp::Shl) => Some((ArithOp::Shl, (ak + bk))),
            _ => None,
        }
    }

    match insn {
        Insn::Arith1(op, a, b)
        | Insn::Arith2(op, a, b)
        | Insn::Arith4(op, a, b)
        | Insn::Arith8(op, a, b) => {
            let wa = widen(prog, a);
            let wb = widen(prog, b);

            match (op, wa, wb) {
                (ArithOp::Add, Insn::Const8(0), _) => Insn::Get(b),
                (ArithOp::Add, _, Insn::Const8(0)) => Insn::Get(a),
                (ArithOp::Mul, Insn::Const8(1), _) => Insn::Get(b),
                (ArithOp::Mul, _, Insn::Const8(1)) => Insn::Get(a),

                (op, Insn::Const8(ak), Insn::Const8(bk)) => Insn::Const8(eval_const(op, ak, bk)),

                (op_out, Insn::ArithK8(op_in, r, k1), Insn::Const8(k2))
                | (op_out, Insn::Const8(k2), Insn::ArithK8(op_in, r, k1)) => {
                    if let Some((op, rk)) = assoc_const(op_out, op_in, k2, k1) {
                        Insn::ArithK8(op, r, rk)
                    } else {
                        insn
                    }
                }

                (op, Insn::Const8(ak), _) => Insn::ArithK8(op, b, ak as i64),
                (op, _, Insn::Const8(bk)) => Insn::ArithK8(op, a, bk as i64),

                _ => insn,
            }
        }

        Insn::ArithK1(op, a, bk)
        | Insn::ArithK2(op, a, bk)
        | Insn::ArithK4(op, a, bk)
        | Insn::ArithK8(op, a, bk) => match widen(prog, a) {
            Insn::Const8(ak) => Insn::Const8(eval_const(op, ak as i64, bk)),
            Insn::ArithK8(op_in, ar, ak) => {
                let op_out = op;
                if let Some((op, rk)) = assoc_const(op_in, op_out, ak as i64, bk) {
                    Insn::ArithK8(op, ar, rk)
                } else {
                    insn
                }
            }
            _ => insn,
        },

        _ => insn,
    }
}

fn fold_subregs(insn: mil::Insn, prog: &ssa::Program) -> Insn {
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

    let src_sz = prog.value_type(src).bytes_size().unwrap();
    assert!(end as usize <= src_sz);

    let src = prog.get(src).unwrap();
    match src.insn.get() {
        Insn::Part {
            src: up_src,
            offset: up_offset,
            size: up_size,
        } => {
            let up_end = up_offset + up_size;
            let up_src_sz = prog.value_type(up_src).bytes_size().unwrap();
            assert!(up_end as usize <= up_src_sz);

            Insn::Part {
                src: up_src,
                offset: offset + up_offset,
                size,
            }
        }

        Insn::Concat { lo, hi } => {
            let lo_sz = prog
                .value_type(lo)
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

fn fold_bitops(insn: mil::Insn) -> Insn {
    match insn {
        Insn::Arith1(ArithOp::BitXor, a, b) if a == b => Insn::Const1(0),
        Insn::Arith2(ArithOp::BitXor, a, b) if a == b => Insn::Const2(0),
        Insn::Arith4(ArithOp::BitXor, a, b) if a == b => Insn::Const4(0),
        Insn::Arith8(ArithOp::BitXor, a, b) if a == b => Insn::Const8(0),

        Insn::Arith1(ArithOp::BitAnd, a, b) if a == b => Insn::Get(a),
        Insn::Arith2(ArithOp::BitAnd, a, b) if a == b => Insn::Get(a),
        Insn::Arith4(ArithOp::BitAnd, a, b) if a == b => Insn::Get(a),
        Insn::Arith8(ArithOp::BitAnd, a, b) if a == b => Insn::Get(a),

        Insn::Arith1(ArithOp::BitOr, a, b) if a == b => Insn::Get(a),
        Insn::Arith2(ArithOp::BitOr, a, b) if a == b => Insn::Get(a),
        Insn::Arith4(ArithOp::BitOr, a, b) if a == b => Insn::Get(a),
        Insn::Arith8(ArithOp::BitOr, a, b) if a == b => Insn::Get(a),

        _ => insn,
    }
}

fn fold_get(mut insn: mil::Insn, prog: &ssa::Program) -> Insn {
    for input in insn.input_regs_iter_mut() {
        loop {
            let input_def = prog.get(*input).unwrap().insn.get();
            if let Insn::Get(arg) = input_def {
                *input = arg;
            } else {
                break;
            }
        }
    }

    insn
}

fn simplify_half_null_concat(insn: Insn, prog: &ssa::Program) -> Insn {
    if let Insn::Concat { lo, hi } = insn {
        let is_lo_null = matches!(prog.get(lo).unwrap().insn.get(), Insn::Part { size: 0, .. });
        let is_hi_null = matches!(prog.get(hi).unwrap().insn.get(), Insn::Part { size: 0, .. });

        match (is_lo_null, is_hi_null) {
            (true, true) => panic!("assertion error"),
            (false, true) => Insn::Get(lo),
            (true, false) => Insn::Get(hi),
            (false, false) => insn,
        }
    } else {
        insn
    }
}

/// Perform the standard chain of transformations that we intend to generally apply to programs
pub fn canonical(prog: &mut ssa::Program) {
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

    for bid in prog.cfg().block_ids_rpo() {
        for insn_cell in prog.block_normal_insns(bid).unwrap().insns.iter() {
            let insn = insn_cell.get();
            let insn = fold_get(insn, prog);
            let insn = fold_subregs(insn, prog);
            let insn = fold_bitops(insn);
            let insn = fold_constants(insn, prog);
            let insn = simplify_half_null_concat(insn, prog);
            insn_cell.set(insn);
        }
    }

    prog.assert_invariants();

    ssa::eliminate_dead_code(prog);
}

#[cfg(test)]
mod tests {
    use crate::{mil, ssa};
    use mil::{ArithOp, Insn, Reg};

    mod constant_folding {
        use crate::{mil, ssa, xform};
        use mil::{ArithOp, Insn, Reg};

        #[test]
        fn addk() {
            let prog = {
                let mut b = mil::ProgramBuilder::new();
                b.push(Reg(0), Insn::Ancestral(mil::ANC_STACK_BOTTOM));
                b.push(Reg(1), Insn::Const8(5));
                b.push(Reg(2), Insn::Const8(44));
                b.push(Reg(0), Insn::Arith8(ArithOp::Add, Reg(1), Reg(0)));
                b.push(Reg(3), Insn::Arith8(ArithOp::Add, Reg(0), Reg(1)));
                b.push(Reg(4), Insn::Arith8(ArithOp::Add, Reg(2), Reg(1)));
                b.push(Reg(3), Insn::Const8(0));
                b.push(Reg(4), Insn::Ancestral(mil::ANC_STACK_BOTTOM));
                b.push(Reg(3), Insn::Arith8(ArithOp::Add, Reg(3), Reg(4)));
                b.push(Reg(0), Insn::Ret(Reg(4)));
                b.build()
            };
            let mut prog = ssa::mil_to_ssa(ssa::ConversionParams::new(prog));
            xform::canonical(&mut prog);

            assert_eq!(prog.cfg().block_count(), 1);
            let insns = prog
                .block_normal_insns(prog.cfg().entry_block_id())
                .unwrap();
            assert_eq!(insns.insns.len(), 10);
            assert_eq!(insns.insns[3].get(), Insn::ArithK8(ArithOp::Add, Reg(0), 5));
            assert_eq!(
                insns.insns[4].get(),
                Insn::ArithK8(ArithOp::Add, Reg(0), 10)
            );
            assert_eq!(insns.insns[5].get(), Insn::Const8(49));
            assert_eq!(insns.insns[8].get(), Insn::Get(Reg(7)));
        }

        #[test]
        fn mulk() {
            let prog = {
                let mut b = mil::ProgramBuilder::new();
                b.push(Reg(0), Insn::Ancestral(mil::ANC_STACK_BOTTOM));
                b.push(Reg(1), Insn::Const8(5));
                b.push(Reg(2), Insn::Const8(44));
                b.push(Reg(0), Insn::Arith8(ArithOp::Mul, Reg(1), Reg(0)));
                b.push(Reg(3), Insn::Arith8(ArithOp::Mul, Reg(0), Reg(1)));
                b.push(Reg(4), Insn::Arith8(ArithOp::Mul, Reg(2), Reg(1)));
                b.push(Reg(3), Insn::Const8(1));
                b.push(Reg(4), Insn::Ancestral(mil::ANC_STACK_BOTTOM));
                b.push(Reg(4), Insn::Arith8(ArithOp::Mul, Reg(3), Reg(4)));
                b.push(Reg(0), Insn::Ret(Reg(4)));
                b.build()
            };
            let mut prog = ssa::mil_to_ssa(ssa::ConversionParams::new(prog));
            xform::canonical(&mut prog);

            let insns = prog
                .block_normal_insns(prog.cfg().entry_block_id())
                .unwrap();
            assert_eq!(insns.insns.len(), 10);
            assert_eq!(insns.insns[3].get(), Insn::ArithK8(ArithOp::Mul, Reg(0), 5));
            assert_eq!(
                insns.insns[4].get(),
                Insn::ArithK8(ArithOp::Mul, Reg(0), 25)
            );
            assert_eq!(insns.insns[5].get(), Insn::Const8(5 * 44));
            assert_eq!(insns.insns[8].get(), Insn::Get(Reg(7)));
        }
    }

    mod subreg_folding {
        use crate::{mil, ssa, xform};

        define_ancestral_name!(ANC_A, "A");
        define_ancestral_name!(ANC_B, "B");

        #[test]
        fn part_of_concat() {
            use mil::{Insn, Reg};

            #[derive(Clone, Copy)]
            struct VariantParams {
                anc_a_sz: u8,
                anc_b_sz: u8,
                offset: u8,
                size: u8,
            }
            fn gen_prog(vp: VariantParams) -> mil::Program {
                let mut b = mil::ProgramBuilder::new();
                b.set_ancestral_type(ANC_A, mil::RegType::Bytes(vp.anc_a_sz as usize));
                b.set_ancestral_type(ANC_B, mil::RegType::Bytes(vp.anc_b_sz as usize));
                b.push(Reg(0), Insn::Ancestral(ANC_A));
                b.push(Reg(1), Insn::Ancestral(ANC_B));
                b.push(
                    Reg(2),
                    Insn::Concat {
                        lo: Reg(0),
                        hi: Reg(1),
                    },
                );
                b.push(
                    Reg(3),
                    Insn::Part {
                        src: Reg(2),
                        offset: vp.offset,
                        size: vp.size,
                    },
                );
                b.build()
            }

            for anc_a_sz in 1..=7 {
                for anc_b_sz in 1..=(8 - anc_a_sz) {
                    let concat_sz = anc_a_sz + anc_b_sz;

                    // case: fall within lo
                    for offset in 0..=(anc_a_sz - 1) {
                        for size in 1..=(anc_a_sz - offset) {
                            let prog = gen_prog(VariantParams {
                                anc_a_sz,
                                anc_b_sz,
                                offset,
                                size,
                            });
                            let mut prog = ssa::mil_to_ssa(ssa::ConversionParams::new(prog));
                            xform::canonical(&mut prog);

                            assert_eq!(
                                prog.get(Reg(3)).unwrap().insn.get(),
                                Insn::Part {
                                    src: Reg(0),
                                    offset,
                                    size
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
                            let mut prog = ssa::mil_to_ssa(ssa::ConversionParams::new(prog));
                            xform::canonical(&mut prog);

                            assert_eq!(
                                prog.get(Reg(3)).unwrap().insn.get(),
                                Insn::Part {
                                    src: Reg(1),
                                    offset: offset - anc_a_sz,
                                    size,
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
                            let mut prog = ssa::mil_to_ssa(ssa::ConversionParams::new(prog));
                            let orig_insn = prog.get(Reg(3)).unwrap().insn.get();

                            xform::canonical(&mut prog);
                            assert_eq!(prog.get(Reg(3)).unwrap().insn.get(), orig_insn);
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
                src_sz: u8,
                offs0: u8,
                size0: u8,
                offs1: u8,
                size1: u8,
            }

            fn gen_prog(vp: VariantParams) -> mil::Program {
                let mut b = mil::ProgramBuilder::new();
                b.set_ancestral_type(ANC_A, mil::RegType::Bytes(vp.src_sz as usize));
                b.push(Reg(0), Insn::Ancestral(ANC_A));
                b.push(
                    Reg(1),
                    Insn::Part {
                        src: Reg(0),
                        offset: vp.offs0,
                        size: vp.size0,
                    },
                );
                b.push(
                    Reg(2),
                    Insn::Part {
                        src: Reg(1),
                        offset: vp.offs1,
                        size: vp.size1,
                    },
                );
                b.build()
            }

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
                                let mut prog = ssa::mil_to_ssa(ssa::ConversionParams::new(prog));
                                xform::canonical(&mut prog);

                                let exp_offset = offs0 + offs1;
                                let exp_size = size1;
                                assert_eq!(
                                    prog.get(Reg(2)).unwrap().insn.get(),
                                    Insn::Part {
                                        src: Reg(0),
                                        offset: exp_offset,
                                        size: exp_size,
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

        let prog = {
            let mut b = mil::ProgramBuilder::new();
            b.push(Reg(1), Insn::Const8(5));
            b.push(Reg(2), Insn::Const8(44));

            // removed by fold_bitops
            b.push(Reg(1), Insn::Arith8(ArithOp::BitAnd, Reg(1), Reg(1)));
            b.push(Reg(2), Insn::Arith8(ArithOp::BitAnd, Reg(2), Reg(2)));

            // removed by fold_constants IF the Insn::Get's added by fold_bitops
            // is dereferenced
            b.push(Reg(0), Insn::Arith8(ArithOp::Mul, Reg(1), Reg(2)));
            b.push(Reg(0), Insn::Ret(Reg(0)));
            b.build()
        };
        let mut prog = ssa::mil_to_ssa(ssa::ConversionParams::new(prog));
        super::canonical(&mut prog);

        let insns = prog
            .block_normal_insns(prog.cfg().entry_block_id())
            .unwrap();
        assert_eq!(insns.insns.len(), 6);
        assert_eq!(insns.insns[4].get(), Insn::Const8(5 * 44));
    }
}
