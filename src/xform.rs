use crate::{
    mil::{self, ArithOp, Insn},
    ssa,
};

// TODO Fix the algorithm to work with different instruction output sizes.
// NOTE Right now folding is done across instructions of different sizes. It's a known limitation.
pub fn fold_constants(prog: &mut ssa::Program) {
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

    for bid in prog.cfg().block_ids_rpo() {
        let insns = prog.block_normal_insns(bid).unwrap();
        for insn_cell in insns.insns.iter() {
            let repl_insn = match insn_cell.get() {
                Insn::Arith1(op, a, b)
                | Insn::Arith2(op, a, b)
                | Insn::Arith4(op, a, b)
                | Insn::Arith8(op, a, b) => {
                    let wa = widen(prog, a);
                    let wb = widen(prog, b);

                    match (op, wa, wb) {
                        (ArithOp::Add, Insn::Const8(0), _) => Insn::Get8(b),
                        (ArithOp::Add, _, Insn::Const8(0)) => Insn::Get8(a),
                        (ArithOp::Mul, Insn::Const8(1), _) => Insn::Get8(b),
                        (ArithOp::Mul, _, Insn::Const8(1)) => Insn::Get8(a),

                        (op, Insn::Const8(ak), Insn::Const8(bk)) => {
                            Insn::Const8(eval_const(op, ak, bk))
                        }

                        (op_out, Insn::ArithK8(op_in, r, k1), Insn::Const8(k2))
                        | (op_out, Insn::Const8(k2), Insn::ArithK8(op_in, r, k1)) => {
                            let Some((op, rk)) = assoc_const(op_out, op_in, k2, k1) else {
                                continue;
                            };
                            Insn::ArithK8(op, r, rk)
                        }

                        (op, Insn::Const8(ak), _) => Insn::ArithK8(op, b, ak as i64),
                        (op, _, Insn::Const8(bk)) => Insn::ArithK8(op, a, bk as i64),

                        _ => continue,
                    }
                }

                Insn::ArithK1(op, a, bk)
                | Insn::ArithK2(op, a, bk)
                | Insn::ArithK4(op, a, bk)
                | Insn::ArithK8(op, a, bk) => match widen(prog, a) {
                    Insn::Const8(ak) => Insn::Const8(eval_const(op, ak as i64, bk)),
                    Insn::ArithK8(op_in, ar, ak) => {
                        let op_out = op;
                        let Some((op, rk)) = assoc_const(op_in, op_out, ak as i64, bk) else {
                            continue;
                        };
                        Insn::ArithK8(op, ar, rk)
                    }
                    _ => continue,
                },

                _ => continue,
            };

            insn_cell.set(repl_insn);
        }
    }
}

pub fn fold_subregs(prog: &mut ssa::Program) {
    for bid in prog.cfg().block_ids_rpo() {
        for (_, insn_cell) in prog.block_normal_insns(bid).unwrap().iter() {
            let mut subreg_insn = insn_cell.get();
            let mut arg = match subreg_insn {
                Insn::V8WithL1(big, _) | Insn::V8WithL2(big, _) | Insn::V8WithL4(big, _) => big,
                _ => continue,
            };

            loop {
                let arg_def = prog.get(arg).unwrap().insn.get();

                (arg, subreg_insn) = match (subreg_insn, arg_def) {
                    (Insn::V8WithL1(_, small), Insn::V8WithL1(big, _)) => {
                        (big, Insn::V8WithL1(big, small))
                    }
                    (Insn::V8WithL2(_, small), Insn::V8WithL2(big, _)) => {
                        (big, Insn::V8WithL2(big, small))
                    }
                    (Insn::V8WithL4(_, small), Insn::V8WithL4(big, _)) => {
                        (big, Insn::V8WithL4(big, small))
                    }
                    _ => break,
                };

                insn_cell.set(subreg_insn);
            }
        }
    }

    for bid in prog.cfg().block_ids_rpo() {
        for (_, insn_cell) in prog.block_normal_insns(bid).unwrap().iter() {
            let subreg_insn = insn_cell.get();
            let mut arg = match subreg_insn {
                Insn::L1(x) | Insn::L2(x) | Insn::L4(x) => x,
                _ => continue,
            };

            loop {
                let arg_def = prog.get(arg).unwrap().insn.get();

                arg = match (subreg_insn, arg_def) {
                    (Insn::L1(_), Insn::V8WithL1(_, small)) => small,
                    (Insn::L2(_), Insn::V8WithL2(_, small)) => small,
                    (Insn::L4(_), Insn::V8WithL4(_, small)) => small,
                    _ => break,
                };
            }

            // actually, we should use the properly sized Get# insn
            insn_cell.set(Insn::Get8(arg));
        }
    }
}

pub fn fold_bitops(prog: &mut ssa::Program) {
    for bid in prog.cfg().block_ids_rpo() {
        for (_, insn_cell) in prog.block_normal_insns(bid).unwrap().iter() {
            let repl = match insn_cell.get() {
                Insn::Arith1(ArithOp::BitXor, a, b) if a == b => Insn::Const1(0),
                Insn::Arith2(ArithOp::BitXor, a, b) if a == b => Insn::Const2(0),
                Insn::Arith4(ArithOp::BitXor, a, b) if a == b => Insn::Const4(0),
                Insn::Arith8(ArithOp::BitXor, a, b) if a == b => Insn::Const8(0),

                Insn::Arith1(ArithOp::BitAnd, a, b) if a == b => Insn::Get8(a),
                Insn::Arith2(ArithOp::BitAnd, a, b) if a == b => Insn::Get8(a),
                Insn::Arith4(ArithOp::BitAnd, a, b) if a == b => Insn::Get8(a),
                Insn::Arith8(ArithOp::BitAnd, a, b) if a == b => Insn::Get8(a),

                Insn::Arith1(ArithOp::BitOr, a, b) if a == b => Insn::Get8(a),
                Insn::Arith2(ArithOp::BitOr, a, b) if a == b => Insn::Get8(a),
                Insn::Arith4(ArithOp::BitOr, a, b) if a == b => Insn::Get8(a),
                Insn::Arith8(ArithOp::BitOr, a, b) if a == b => Insn::Get8(a),

                _ => continue,
            };

            insn_cell.set(repl);
        }
    }
}

/// Remove `Get` instructions.
///
/// `Get` instructions are not really required in an SSA program. They generally
/// come out of several transform passes as a way to simplify the transform
/// algorithm themselves. They can then be safely removed by this pass.
///
/// (Actually, this pass removes the *dependency* on these insns; as a
/// result, they merely become dead and will be properly eliminated by
/// `ssa::eliminate_dead_code`)
pub fn fold_get(prog: &mut ssa::Program) {
    for bid in prog.cfg().block_ids_rpo() {
        for (_, insn_cell) in prog.block_normal_insns(bid).unwrap().iter() {
            // Get instructions are affected too!
            // this allows us to skip entire chains of multiple Get insns

            let mut insn = insn_cell.get();
            for input_reg in insn.input_regs_mut().into_iter().flatten() {
                let input_def = prog.get(*input_reg).unwrap().insn.get();
                if let Insn::Get8(x) = input_def {
                    *input_reg = x;
                }
            }

            insn_cell.set(insn);
        }
    }
}

/// Perform the standard chain of transformations that we intend to generally apply to programs
pub fn canonical(prog: &mut ssa::Program) {
    prog.assert_invariants();

    fold_subregs(prog);
    #[cfg(debug_assertions)]
    prog.assert_invariants();

    fold_get(prog);
    #[cfg(debug_assertions)]
    prog.assert_invariants();

    fold_bitops(prog);
    #[cfg(debug_assertions)]
    prog.assert_invariants();

    fold_get(prog);
    #[cfg(debug_assertions)]
    prog.assert_invariants();

    fold_constants(prog);
    #[cfg(debug_assertions)]
    prog.assert_invariants();

    ssa::eliminate_dead_code(prog);
    prog.assert_invariants();
}

#[cfg(test)]
mod tests {
    mod constant_folding {
        use crate::{mil, ssa, xform};

        #[test]
        fn addk() {
            use mil::{ArithOp, Insn, Reg};

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
            xform::fold_constants(&mut prog);

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
            assert_eq!(insns.insns[8].get(), Insn::Get8(Reg(7)));
        }

        #[test]
        fn mulk() {
            use mil::{ArithOp, Insn, Reg};

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
            xform::fold_constants(&mut prog);

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
            assert_eq!(insns.insns[8].get(), Insn::Get8(Reg(7)));
        }
    }

    mod subreg_folding {
        use crate::{mil, ssa, xform};

        #[test]
        fn simple_l4() {
            simple_tmpl(mil::Insn::V8WithL4, mil::Insn::L4);
        }

        #[test]
        fn simple_l2() {
            simple_tmpl(mil::Insn::V8WithL2, mil::Insn::L2);
        }

        #[test]
        fn simple_l1() {
            simple_tmpl(mil::Insn::V8WithL1, mil::Insn::L1);
        }

        fn simple_tmpl(
            replacer: fn(mil::Reg, mil::Reg) -> mil::Insn,
            selector: fn(mil::Reg) -> mil::Insn,
        ) {
            use mil::{Insn, Reg};

            let prog = {
                let mut b = mil::ProgramBuilder::new();
                b.push(Reg(0), Insn::Ancestral(mil::ANC_STACK_BOTTOM));
                b.push(Reg(1), Insn::Const8(123));
                b.push(Reg(0), replacer(Reg(0), Reg(1)));
                b.push(Reg(1), selector(Reg(0)));
                b.push(Reg(1), Insn::Ret(Reg(1)));
                b.build()
            };
            let mut prog = ssa::mil_to_ssa(ssa::ConversionParams::new(prog));
            xform::fold_subregs(&mut prog);
            eprintln!("{:?}", prog);

            assert_eq!(prog.get(Reg(3)).unwrap().insn.get(), Insn::Get8(Reg(1)));
        }
    }
}
