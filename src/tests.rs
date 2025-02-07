#![cfg(test)]

mod callconv_x86_64 {
    use crate::pp::PrettyPrinter;
    use crate::{ast, mil, ssa, ty, x86_to_mil};

    use iced_x86::code_asm::CodeAssembler;
    use iced_x86::Instruction;

    #[test]
    fn struct_2p() {
        let input = {
            use iced_x86::code_asm::{eax, ptr, rsi};
            let mut asm = CodeAssembler::new(64).unwrap();

            let mut lbl_end = asm.create_label();
            let mut lbl_loop = asm.create_label();

            asm.mov(eax, 1).unwrap();
            asm.test(rsi, rsi).unwrap();
            asm.je(lbl_end).unwrap();
            asm.set_label(&mut lbl_loop).unwrap();
            asm.inc(eax).unwrap();
            asm.mov(rsi, ptr(rsi + 8)).unwrap();
            asm.test(rsi, rsi).unwrap();
            asm.jne(lbl_loop).unwrap();
            asm.set_label(&mut lbl_end).unwrap();
            asm.ret().unwrap();

            asm.take_instructions()
        };

        let mut types = ty::TypeSet::new();
        let tyid_char = types.add(ty::Type {
            name: "char".to_owned().into(),
            ty: ty::Ty::Int(ty::Int {
                size: 1,
                signed: ty::Signedness::Signed,
            }),
        });
        let tyid_char_ptr = types.add(ty::Type {
            name: "char*".to_owned().into(),
            ty: ty::Ty::Ptr(tyid_char),
        });
        let tyid_name_item = {
            let tyid = types.add(ty::Type {
                name: "name_item".to_owned().into(),
                ty: ty::Ty::Void,
            });
            let self_ptr = types.add(ty::Type {
                name: "name_item*".to_owned().into(),
                ty: ty::Ty::Ptr(tyid),
            });

            types.set(
                tyid,
                ty::Type {
                    name: "name_item".to_owned().into(),
                    ty: ty::Ty::Struct(ty::Struct {
                        size: 16,
                        members: vec![
                            ty::StructMember {
                                offset: 0,
                                name: "name".to_owned().into(),
                                tyid: tyid_char_ptr,
                            },
                            ty::StructMember {
                                offset: 8,
                                name: "next".to_owned().into(),
                                tyid: self_ptr,
                            },
                        ],
                    }),
                },
            );

            tyid
        };

        print_asm(&input);
        let prog = {
            let mut b = x86_to_mil::Builder::new();
            b.use_types(&types);
            x86_to_mil::callconv::read_func_params(&mut b, &[tyid_name_item]).unwrap();
            b.translate(input.iter().copied())
        }
        .unwrap();
        let output = finish_prog(prog, &types).unwrap();
        insta::assert_snapshot!(output);
    }

    fn finish_prog(prog: mil::Program, types: &ty::TypeSet) -> std::io::Result<String> {
        use std::io::Write;

        let mut out = Vec::new();
        writeln!(out)?;
        writeln!(out, "mil program = ")?;
        writeln!(out, "{:?}", prog)?;

        let mut prog = ssa::mil_to_ssa(ssa::ConversionParams::new(prog));
        crate::xform::canonical(&mut prog);
        ssa::eliminate_dead_code(&mut prog);
        writeln!(out)?;
        writeln!(out, "{:?}", prog)?;

        let ast = ast::Ast::new(&prog);
        writeln!(out)?;
        let mut pp = PrettyPrinter::start(&mut out);
        ast.pretty_print(&mut pp).unwrap();

        Ok(String::from_utf8(out).unwrap())
    }

    fn print_asm(instrs: &[Instruction]) {
        use iced_x86::{Formatter, IntelFormatter};
        let mut formatter = IntelFormatter::new();
        let mut instr_strbuf = String::new();
        for instr in instrs {
            print!("{:16x}: ", instr.ip());
            instr_strbuf.clear();
            formatter.format(&instr, &mut instr_strbuf);
            println!("{}", instr_strbuf);
        }
    }
}
