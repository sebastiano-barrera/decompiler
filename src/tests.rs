#![cfg(test)]

mod logical_vars {
    use crate::{ssa, x86_to_mil};

    use iced_x86::code_asm::CodeAssembler;
    use iced_x86::Instruction;

    #[test]
    fn simple() {
        let input = {
            use iced_x86::code_asm::{eax, ptr, rdi};
            let mut asm = CodeAssembler::new(64).unwrap();
            asm.lea(eax, ptr(rdi + rdi)).unwrap();
            asm.ret().unwrap();

            asm.take_instructions()
        };

        let output = test_with_code(&input).unwrap();
        insta::assert_snapshot!(output);
    }

    #[test]
    fn assign_in_composite() {
        let input = {
            use iced_x86::code_asm::{qword_ptr, rdi};
            let mut asm = CodeAssembler::new(64).unwrap();
            asm.mov(qword_ptr(rdi + 16), 0).unwrap();
            asm.mov(qword_ptr(rdi + 8), 0).unwrap();
            asm.mov(qword_ptr(rdi), 0).unwrap();
            asm.ret().unwrap();

            asm.take_instructions()
        };

        let output = test_with_code(&input).unwrap();
        insta::assert_snapshot!(output);
    }

    fn test_with_code(instrs: &[Instruction]) -> Result<String, std::fmt::Error> {
        use std::fmt::Write;

        let mut out = String::new();

        use iced_x86::{Formatter, IntelFormatter};

        let mut formatter = IntelFormatter::new();
        let mut instr_strbuf = String::new();
        for instr in instrs {
            write!(out, "{:16x}: ", instr.ip())?;
            instr_strbuf.clear();
            formatter.format(&instr, &mut instr_strbuf);
            writeln!(out, "{}", instr_strbuf)?;
        }

        let prog = x86_to_mil::translate(instrs.iter().copied()).unwrap();
        writeln!(out, "mil program = ")?;
        writeln!(out, "{:?}", prog)?;
        writeln!(out,)?;

        let mut prog = ssa::mil_to_ssa(prog);
        crate::xform::fold_constants(&mut prog);
        ssa::eliminate_dead_code(&mut prog);
        writeln!(out, "{:?}", prog)?;

        // TODO print AST as well?

        // let out = std::io::stdout().lock();
        // let mut pp = PrettyPrinter::start(IoAsFmt(out));

        // println!();
        // let ast = ast::ssa_to_ast(&prog);
        // ast.pretty_print(&mut pp).unwrap()

        Ok(out)
    }
}

mod constant_folding {
    use crate::{mil, ssa, xform};

    #[test]
    fn addk() {
        use mil::{Insn, Reg};

        let prog = {
            let mut b = mil::ProgramBuilder::new();
            b.push(Reg(0), Insn::Ancestral(mil::Ancestral::StackBot));
            b.push(Reg(1), Insn::Const8(5));
            b.push(Reg(2), Insn::Const8(44));
            b.push(Reg(0), Insn::Add(Reg(1), Reg(0)));
            b.push(Reg(3), Insn::Add(Reg(0), Reg(1)));
            b.push(Reg(4), Insn::Add(Reg(2), Reg(1)));
            b.push(Reg(3), Insn::Const8(0));
            b.push(Reg(4), Insn::Ancestral(mil::Ancestral::StackBot));
            b.push(Reg(4), Insn::Add(Reg(3), Reg(4)));
            b.push(Reg(0), Insn::Ret(Reg(4)));
            b.build()
        };
        let mut prog = ssa::mil_to_ssa(prog);
        xform::fold_constants(&mut prog);

        assert_eq!(prog.len(), 10);
        assert_eq!(prog.get(3).unwrap().insn, &Insn::AddK(Reg(0), 5));
        assert_eq!(prog.get(4).unwrap().insn, &Insn::AddK(Reg(0), 10));
        assert_eq!(prog.get(5).unwrap().insn, &Insn::Const8(49));
        assert_eq!(prog.get(8).unwrap().insn, &Insn::Get(Reg(7)));
    }

    #[test]
    fn mulk() {
        use mil::{Insn, Reg};

        let prog = {
            let mut b = mil::ProgramBuilder::new();
            b.push(Reg(0), Insn::Ancestral(mil::Ancestral::StackBot));
            b.push(Reg(1), Insn::Const8(5));
            b.push(Reg(2), Insn::Const8(44));
            b.push(Reg(0), Insn::Mul(Reg(1), Reg(0)));
            b.push(Reg(3), Insn::Mul(Reg(0), Reg(1)));
            b.push(Reg(4), Insn::Mul(Reg(2), Reg(1)));
            b.push(Reg(3), Insn::Const8(1));
            b.push(Reg(4), Insn::Ancestral(mil::Ancestral::StackBot));
            b.push(Reg(4), Insn::Mul(Reg(3), Reg(4)));
            b.push(Reg(0), Insn::Ret(Reg(4)));
            b.build()
        };
        let mut prog = ssa::mil_to_ssa(prog);
        xform::fold_constants(&mut prog);

        assert_eq!(prog.len(), 10);
        assert_eq!(prog.get(3).unwrap().insn, &Insn::MulK(Reg(0), 5));
        assert_eq!(prog.get(4).unwrap().insn, &Insn::MulK(Reg(0), 25));
        assert_eq!(prog.get(5).unwrap().insn, &Insn::Const8(5 * 44));
        assert_eq!(prog.get(8).unwrap().insn, &Insn::Get(Reg(7)));
    }
}
