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
        writeln!(out)?;
        writeln!(out, "mil program = ")?;
        writeln!(out, "{:?}", prog)?;

        let mut prog = ssa::mil_to_ssa(ssa::ConversionParams::new(prog));
        crate::xform::fold_constants(&mut prog);
        ssa::eliminate_dead_code(&mut prog);
        writeln!(out)?;
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
