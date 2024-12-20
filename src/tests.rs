#![cfg(test)]

use iced_x86::Instruction;

use crate::{ssa, x86_to_mil};

mod logical_vars {
    use super::test_with_code;
    use iced_x86::code_asm::CodeAssembler;

    #[test]
    #[ignore]
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
    #[ignore]
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
            b.push(Reg(0), Insn::Add(Reg(1), Reg(0)));
            b.push(Reg(2), Insn::Add(Reg(0), Reg(1)));
            b.push(Reg(0), Insn::Ret(Reg(0)));
            b.build()
        };
        let mut prog = ssa::mil_to_ssa(prog);

        eprintln!();
        eprintln!("PRE:");
        eprintln!("{:?}", prog);
        xform::fold_constants(&mut prog);

        eprintln!();
        eprintln!("POST:");
        eprintln!("{:?}", prog);

        assert_eq!(prog.len(), 1);
        assert_eq!(prog.get(0).unwrap().insn, &Insn::Const8(128));
    }
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

    let prog = ssa::mil_to_ssa(prog);
    writeln!(out, "{:?}", prog)?;

    // TODO print AST as well?

    // let out = std::io::stdout().lock();
    // let mut pp = PrettyPrinter::start(IoAsFmt(out));

    // println!();
    // let ast = ast::ssa_to_ast(&prog);
    // ast.pretty_print(&mut pp).unwrap()

    Ok(out)
}
