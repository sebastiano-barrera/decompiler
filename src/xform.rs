use crate::{cfg, mil, ssa};

pub fn fold_constants(prog: &mut ssa::Program) {
    use mil::{Insn, Reg};

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
                Insn::AddK(r, k) => return Assoc::Add(r, k),
                Insn::MulK(r, k) => return Assoc::Mul(r, k),
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

            let (a, b) = match insn {
                Insn::Add(a, b) | Insn::Mul(a, b) => (a, b),
                _ => continue,
            };

            let aa = assoc_of(&*prog, a);
            let ba = assoc_of(&*prog, b);

            let repl_insn = match insn {
                Insn::Add(_, _) => match (aa, ba) {
                    (Assoc::Const(0), _) => Some(Insn::Get(b)),
                    (_, Assoc::Const(0)) => Some(Insn::Get(a)),
                    (Assoc::Const(ak), Assoc::Const(bk)) => Some(Insn::Const8((ak + bk) as u64)),
                    (Assoc::Const(ak), Assoc::Add(r, bk)) => Some(Insn::AddK(r, ak + bk)),
                    (Assoc::Add(r, ak), Assoc::Const(bk)) => Some(Insn::AddK(r, ak + bk)),
                    (Assoc::Const(ak), _) => Some(Insn::AddK(b, ak)),
                    (_, Assoc::Const(bk)) => Some(Insn::AddK(a, bk)),
                    (_, _) => None,
                },

                Insn::Mul(_, _) => match (aa, ba) {
                    (Assoc::Const(1), _) => Some(Insn::Get(b)),
                    (_, Assoc::Const(1)) => Some(Insn::Get(a)),
                    (Assoc::Const(ak), Assoc::Const(bk)) => Some(Insn::Const8((ak * bk) as u64)),
                    (Assoc::Const(ak), Assoc::Mul(r, bk)) => Some(Insn::MulK(r, ak * bk)),
                    (Assoc::Mul(r, ak), Assoc::Const(bk)) => Some(Insn::MulK(r, ak * bk)),
                    (Assoc::Const(ak), _) => Some(Insn::MulK(b, ak)),
                    (_, Assoc::Const(bk)) => Some(Insn::MulK(a, bk)),
                    (_, _) => None,
                },

                _ => None,
            };

            // reborrow here, so that the match above runs with prog borrowed immut.
            if let Some(repl_insn) = repl_insn {
                insn_cell.set(repl_insn);
            }
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

        fn basic_program() -> ssa::Program {
            let prog = {
                use crate::mil::{self, Insn, Reg};

                let mut b = mil::ProgramBuilder::new();

                // Main entry point with some arithmetic
                b.push(Reg(0), Insn::Const8(100));
                b.push(Reg(1), Insn::Const4(5));
                b.push(Reg(0), Insn::Mul(Reg(0), Reg(1)));

                // Conditional branch based on comparison
                b.push(Reg(0), Insn::LT(Reg(0), Reg(1)));
                b.push(
                    Reg(2),
                    Insn::JmpIf {
                        cond: Reg(0),
                        target: 10,
                    },
                );

                // True path: do some pointer arithmetic
                b.push(Reg(0), Insn::Ancestral(mil::Ancestral::Pre("arg0")));
                b.push(Reg(1), Insn::Const2(8));
                b.push(Reg(0), Insn::AddK(Reg(0), 16));
                b.push(Reg(1), Insn::LoadMem8(Reg(0)));
                b.push(Reg(0), Insn::Ret(Reg(1)));

                // False path: call a function
                b.push(Reg(0), Insn::Ancestral(mil::Ancestral::Pre("arg1")));
                b.push(Reg(4), Insn::Ancestral(mil::Ancestral::Pre("callee")));
                b.push(Reg(2), Insn::Call(Reg(4)));
                b.push(Reg(3), Insn::CArg(Reg(0)));
                b.push(Reg(0), Insn::Ret(Reg(1)));

                b.build()
            };

            let prog = ssa::mil_to_ssa(prog);
            eprintln!();
            eprintln!("SSA:\n{:?}", prog);

            prog
        }

        #[test]
        fn simple_error() {
            let mut program = basic_program();
            super::edit_types(&mut program, |typing| {});
        }
    }
}
