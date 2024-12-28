use crate::{cfg, mil, ssa};

// TODO Fix the algorithm to work with different instruction output sizes.
// NOTE Right now folding is done across instructions of different sizes. It's a known limitation.
pub fn fold_constants(prog: &mut ssa::Program) {
    use mil::{ArithOp, Insn, Reg};

    /// Associativity status of an instruction.
    ///
    /// It's a "projection" of an instruction: only represents just the bit of info
    /// that the algorithm cares about. Mostly for simplifying the algo and reducing branching.
    enum Assoc {
        Opaque,
        Const(i64),
        Add(Reg, i64),
        Mul(Reg, i64),
    }

    fn assoc_of(prog: &ssa::Program, mut reg: mil::Reg) -> Assoc {
        loop {
            match prog.get(reg).unwrap().insn.get() {
                Insn::Const1(k) => return Assoc::Const(k as i64),
                Insn::Const2(k) => return Assoc::Const(k as i64),
                Insn::Const4(k) => return Assoc::Const(k as i64),
                Insn::Const8(k) => return Assoc::Const(k as i64),
                Insn::ArithK1(ArithOp::Add, r, k)
                | Insn::ArithK2(ArithOp::Add, r, k)
                | Insn::ArithK4(ArithOp::Add, r, k)
                | Insn::ArithK8(ArithOp::Add, r, k) => return Assoc::Add(r, k),
                Insn::ArithK1(ArithOp::Mul, r, k)
                | Insn::ArithK2(ArithOp::Mul, r, k)
                | Insn::ArithK4(ArithOp::Mul, r, k)
                | Insn::ArithK8(ArithOp::Mul, r, k) => return Assoc::Mul(r, k),
                Insn::Get8(r) => {
                    reg = r;
                }
                _ => return Assoc::Opaque,
            };
        }
    }

    let order = cfg::traverse_reverse_postorder(prog.cfg());

    for &bid in order.block_ids() {
        let insns = prog.block_normal_insns(bid).unwrap();
        for insn_cell in insns.insns.iter() {
            let insn = insn_cell.get();

            let (op, a, b) = match insn {
                Insn::Arith1(op @ (ArithOp::Add | ArithOp::Mul), a, b)
                | Insn::Arith2(op @ (ArithOp::Add | ArithOp::Mul), a, b)
                | Insn::Arith4(op @ (ArithOp::Add | ArithOp::Mul), a, b)
                | Insn::Arith8(op @ (ArithOp::Add | ArithOp::Mul), a, b) => (op, a, b),
                _ => continue,
            };

            let aa = assoc_of(&*prog, a);
            let ba = assoc_of(&*prog, b);

            let repl_insn = match (op, aa, ba) {
                (ArithOp::Add, Assoc::Const(0), _) => Insn::Get8(b),
                (ArithOp::Add, _, Assoc::Const(0)) => Insn::Get8(a),
                (ArithOp::Mul, Assoc::Const(1), _) => Insn::Get8(b),
                (ArithOp::Mul, _, Assoc::Const(1)) => Insn::Get8(a),

                (ArithOp::Add, Assoc::Const(ak), Assoc::Const(bk)) => {
                    Insn::Const8((ak + bk) as u64)
                }
                (ArithOp::Mul, Assoc::Const(ak), Assoc::Const(bk)) => {
                    Insn::Const8((ak * bk) as u64)
                }

                (ArithOp::Add, Assoc::Const(ak), Assoc::Add(r, bk)) => {
                    Insn::ArithK8(ArithOp::Add, r, ak + bk)
                }
                (ArithOp::Add, Assoc::Add(r, ak), Assoc::Const(bk)) => {
                    Insn::ArithK8(ArithOp::Add, r, ak + bk)
                }
                (ArithOp::Add, Assoc::Const(ak), _) => Insn::ArithK8(ArithOp::Add, b, ak),
                (ArithOp::Add, _, Assoc::Const(bk)) => Insn::ArithK8(ArithOp::Add, a, bk),

                (ArithOp::Mul, Assoc::Const(ak), Assoc::Mul(r, bk)) => {
                    Insn::ArithK8(ArithOp::Mul, r, ak * bk)
                }
                (ArithOp::Mul, Assoc::Mul(r, ak), Assoc::Const(bk)) => {
                    Insn::ArithK8(ArithOp::Mul, r, ak * bk)
                }
                (ArithOp::Mul, Assoc::Const(ak), _) => Insn::ArithK8(ArithOp::Mul, b, ak),
                (ArithOp::Mul, _, Assoc::Const(bk)) => Insn::ArithK8(ArithOp::Mul, a, bk),

                (_, _, _) => continue,
            };

            insn_cell.set(repl_insn);
        }
    }
}
