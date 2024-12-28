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
                Insn::Get(r) => {
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
                (ArithOp::Add, Assoc::Const(0), _) => Insn::Get(b),
                (ArithOp::Add, _, Assoc::Const(0)) => Insn::Get(a),
                (ArithOp::Mul, Assoc::Const(1), _) => Insn::Get(b),
                (ArithOp::Mul, _, Assoc::Const(1)) => Insn::Get(a),

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

mod typing {
    use crate::ssa;

    pub fn edit_types<'a, F>(program: &'a mut ssa::Program, action: F) -> ssa::ty::CheckResult
    where
        F: FnOnce(TypeEditor),
    {
        action(TypeEditor(program));
        ssa::ty::check_types(program)
    }

    pub struct TypeEditor<'a>(&'a mut ssa::Program);

    impl<'a> TypeEditor<'a> {}

    #[cfg(test)]
    mod tests {
        use crate::ssa;

        define_ancestral_name!(ANC_ARG0, "arg0");
        define_ancestral_name!(ANC_ARG1, "arg1");
        define_ancestral_name!(ANC_CALLEE, "callee");

        fn basic_program() -> ssa::Program {
            let prog = {
                use crate::mil::{self, Insn, Reg};

                let mut b = mil::ProgramBuilder::new();

                // Main entry point with some arithmetic
                b.push(Reg(0), Insn::Const8(100));
                b.push(Reg(1), Insn::Const4(5));
                b.push(Reg(0), Insn::Arith8(mil::ArithOp::Mul, Reg(0), Reg(1)));

                // Conditional branch based on comparison
                b.push(Reg(0), Insn::Cmp(mil::CmpOp::LT, Reg(0), Reg(1)));
                b.push(
                    Reg(2),
                    Insn::JmpIf {
                        cond: Reg(0),
                        target: 10,
                    },
                );

                // True path: do some pointer arithmetic
                b.push(Reg(0), Insn::Ancestral(ANC_ARG0));
                b.push(Reg(1), Insn::Const2(8));
                b.push(Reg(0), Insn::ArithK8(mil::ArithOp::Add, Reg(0), 16));
                b.push(Reg(1), Insn::LoadMem8(Reg(0)));
                b.push(Reg(0), Insn::Ret(Reg(1)));

                // False path: call a function
                b.push(Reg(0), Insn::Ancestral(ANC_ARG1));
                b.push(Reg(4), Insn::Ancestral(ANC_CALLEE));
                b.push(Reg(2), Insn::Call(Reg(4)));
                b.push(Reg(3), Insn::CArg(Reg(0)));
                b.push(Reg(0), Insn::Ret(Reg(1)));

                b.build()
            };

            let prog = ssa::mil_to_ssa(ssa::ConversionParams::new(prog));
            eprintln!();
            eprintln!("SSA:\n{:?}", prog);

            prog
        }

        #[test]
        fn simple_error() {
            let mut program = basic_program();
            super::edit_types(&mut program, |_typing| {}).unwrap();
        }
    }
}
