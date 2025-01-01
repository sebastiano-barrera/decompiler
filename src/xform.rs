use crate::{mil, ssa};

// TODO Fix the algorithm to work with different instruction output sizes.
// NOTE Right now folding is done across instructions of different sizes. It's a known limitation.
pub fn fold_constants(prog: &mut ssa::Program) {
    use mil::{ArithOp, Insn, Reg};

    fn widen(prog: &ssa::Program, mut reg: mil::Reg) -> Insn {
        loop {
            #[rustfmt::skip]
            match prog.get(reg).unwrap().insn.get() {
                Insn::Const1(k) => return Insn::Const8(k as u64),
                Insn::Const2(k) => return Insn::Const8(k as u64),
                Insn::Const4(k) => return Insn::Const8(k as u64),
                Insn::Const8(k) => return Insn::Const8(k as u64),

                Insn::Arith1(op, a, b) => return Insn::Arith8(op, a, b),
                Insn::Arith2(op, a, b) => return Insn::Arith8(op, a, b),
                Insn::Arith4(op, a, b) => return Insn::Arith8(op, a, b),
                Insn::Arith8(op, a, b) => return Insn::Arith8(op, a, b),

                Insn::ArithK1(op, r, k) => return Insn::ArithK8(op, r, k as i64),
                Insn::ArithK2(op, r, k) => return Insn::ArithK8(op, r, k as i64),
                Insn::ArithK4(op, r, k) => return Insn::ArithK8(op, r, k as i64),
                Insn::ArithK8(op, r, k) => return Insn::ArithK8(op, r, k),

                Insn::Get8(r) => {
                    reg = r;
                }
                insn => return insn,
            };
        }
    }

    let order = cfg::traverse_reverse_postorder(prog.cfg());

    for &bid in order.block_ids() {
        let insns = prog.block_normal_insns(bid).unwrap();
        for (ndx, insn_cell) in insns.insns.iter().enumerate() {
            let repl_insn = match insn_cell.get() {
                Insn::Arith1(op @ (ArithOp::Add | ArithOp::Mul), a, b)
                | Insn::Arith2(op @ (ArithOp::Add | ArithOp::Mul), a, b)
                | Insn::Arith4(op @ (ArithOp::Add | ArithOp::Mul), a, b)
                | Insn::Arith8(op @ (ArithOp::Add | ArithOp::Mul), a, b) => {
                    let wa = widen(prog, a);
                    let wb = widen(prog, b);

                    match (op, wa, wb) {
                        (ArithOp::Add, Insn::Const8(0), wb) => wb,
                        (ArithOp::Add, wa, Insn::Const8(0)) => wa,
                        (ArithOp::Mul, Insn::Const8(1), wb) => wb,
                        (ArithOp::Mul, wa, Insn::Const8(1)) => wa,

                        (ArithOp::Add, Insn::Const8(ak), Insn::Const8(bk)) => {
                            Insn::Const8((ak + bk) as u64)
                        }
                        (ArithOp::Mul, Insn::Const8(ak), Insn::Const8(bk)) => {
                            Insn::Const8((ak * bk) as u64)
                        }

                        (ArithOp::Add, Insn::Const8(ak), Insn::ArithK8(ArithOp::Add, r, bk)) => {
                            Insn::ArithK8(ArithOp::Add, r, ak as i64 + bk)
                        }
                        (ArithOp::Add, Insn::ArithK8(ArithOp::Add, r, ak), Insn::Const8(bk)) => {
                            Insn::ArithK8(ArithOp::Add, r, ak + bk as i64)
                        }
                        (ArithOp::Mul, Insn::Const8(ak), Insn::ArithK8(ArithOp::Mul, r, bk)) => {
                            Insn::ArithK8(ArithOp::Mul, r, ak as i64 * bk)
                        }
                        (ArithOp::Mul, Insn::ArithK8(ArithOp::Mul, r, ak), Insn::Const8(bk)) => {
                            Insn::ArithK8(ArithOp::Mul, r, ak * bk as i64)
                        }

                        (ArithOp::Add, Insn::Const8(ak), _) => {
                            Insn::ArithK8(ArithOp::Add, b, ak as i64)
                        }
                        (ArithOp::Add, _, Insn::Const8(bk)) => {
                            Insn::ArithK8(ArithOp::Add, a, bk as i64)
                        }
                        (ArithOp::Mul, Insn::Const8(ak), _) => {
                            Insn::ArithK8(ArithOp::Mul, b, ak as i64)
                        }
                        (ArithOp::Mul, _, Insn::Const8(bk)) => {
                            Insn::ArithK8(ArithOp::Mul, a, bk as i64)
                        }

                        _ => continue,
                    }
                }

                Insn::ArithK1(ArithOp::Add, a, bk)
                | Insn::ArithK2(ArithOp::Add, a, bk)
                | Insn::ArithK4(ArithOp::Add, a, bk)
                | Insn::ArithK8(ArithOp::Add, a, bk) => match widen(prog, a) {
                    Insn::Const8(ak) => Insn::Const8(ak + bk as u64),
                    Insn::ArithK8(ArithOp::Add, ar, ak) => Insn::ArithK8(ArithOp::Add, ar, ak + bk),
                    _ => continue,
                },

                Insn::ArithK1(ArithOp::Mul, a, bk)
                | Insn::ArithK2(ArithOp::Mul, a, bk)
                | Insn::ArithK4(ArithOp::Mul, a, bk)
                | Insn::ArithK8(ArithOp::Mul, a, bk) => match widen(prog, a) {
                    Insn::Const8(ak) => Insn::Const8(ak + bk as u64),
                    Insn::ArithK8(ArithOp::Mul, ar, ak) => Insn::ArithK8(ArithOp::Mul, ar, ak * bk),
                    _ => continue,
                },

                _ => continue,
            };

            eprintln!(
                "replacing [{}]: {:?} -> {:?}",
                ndx,
                insn_cell.get(),
                repl_insn
            );
            insn_cell.set(repl_insn);
        }
    }
}
