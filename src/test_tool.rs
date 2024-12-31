use iced_x86::Formatter;
use thiserror::Error;

use crate::{ast, pp::PrettyPrinter, ssa, ty, x86_to_mil, xform};

#[derive(Debug, Error)]
pub enum Error {
    #[error("unsupported executable format: {0}")]
    UnsupportedExecFormat(&'static str),

    #[error("fmt: {0}")]
    Fmt(#[from] std::fmt::Error),

    #[error("symbol `{0}` is not a function")]
    NotAFunction(String),
}
pub type Result<T> = std::result::Result<T, Error>;

fn obj_format_name(object: &goblin::Object) -> &'static str {
    match object {
        goblin::Object::Elf(_) => "Elf",
        goblin::Object::PE(_) => "PE",
        goblin::Object::COFF(_) => "COFF",
        goblin::Object::Mach(_) => "Mach",
        goblin::Object::Archive(_) => "Archive",
        _ => "Unknown",
    }
}

pub fn run<W: std::fmt::Write>(raw_binary: &[u8], function_name: &str, out: &mut W) -> Result<()> {
    let object = goblin::Object::parse(&raw_binary).expect("elf parse error");
    let elf = match object {
        goblin::Object::Elf(elf) => elf,
        _ => return Err(Error::UnsupportedExecFormat(obj_format_name(&object))),
    };

    {
        let mut types = ty::TypeSet::new();
        let res = ty::dwarf::load_dwarf_types(&elf, &raw_binary, &mut types);
        writeln!(out, "dwarf types --[[")?;
        match res {
            Ok(report) => {
                let mut pp = PrettyPrinter::start(&mut *out);
                types.dump(&mut pp).unwrap();

                writeln!(out)?;
                writeln!(out, "{} non-fatal errors:", report.errors.len())?;
                for (ofs, err) in &report.errors {
                    writeln!(out, "offset 0x{:8x}: {}", ofs, err)?;
                }
            }
            Err(err) => writeln!(out, "fatal error: {}", err)?,
        }
        writeln!(out, "]]--")?;
    }

    let func_sym = elf
        .syms
        .iter()
        .find(|sym| elf.strtab.get_at(sym.st_name) == Some(function_name))
        .expect("symbol not found");

    if !func_sym.is_function() {
        return Err(Error::NotAFunction(function_name.to_owned()));
    }

    let func_addr = func_sym.st_value as usize;
    let func_size = func_sym.st_size as usize;
    let func_end = func_addr + func_size;

    let text_section = elf
        .section_headers
        .iter()
        .find(|sec| sec.is_executable() && elf.shdr_strtab.get_at(sec.sh_name) == Some(".text"))
        .expect("no .text section?!");

    let vm_range = text_section.vm_range();
    if vm_range.start > func_addr || vm_range.end < func_end {
        writeln!(
            out,
            "function memory range (0x{:x}-0x{:x}) out of .text section vm range (0x{:x}-0x{:x})",
            func_addr, func_end, vm_range.start, vm_range.end
        )?;
    }

    // function's offset into the file
    let func_section_ofs = func_addr - vm_range.start;
    let func_fofs = text_section.sh_offset as usize + func_section_ofs as usize;
    let func_text = &raw_binary[func_fofs..func_fofs + func_size];
    writeln!(
        out,
        "{} 0x{:x}+{} (file 0x{:x})",
        function_name, func_addr, func_size, func_fofs,
    )?;

    let decoder = iced_x86::Decoder::with_ip(
        64,
        func_text,
        func_addr.try_into().unwrap(),
        iced_x86::DecoderOptions::NONE,
    );
    let mut formatter = iced_x86::IntelFormatter::new();
    let mut instr_strbuf = String::new();
    for instr in decoder {
        print!("{:16x}: ", instr.ip());
        let ofs = instr.ip() as usize - func_addr;
        let len = instr.len();
        for i in 0..8 {
            if i < len {
                print!("{:02x} ", func_text[ofs + i]);
            } else {
                print!("   ");
            }
        }

        instr_strbuf.clear();
        formatter.format(&instr, &mut instr_strbuf);
        writeln!(out, "{}", instr_strbuf)?;
    }

    writeln!(out)?;
    let mut decoder = iced_x86::Decoder::with_ip(
        64,
        func_text,
        func_addr.try_into().unwrap(),
        iced_x86::DecoderOptions::NONE,
    );
    let prog = x86_to_mil::translate(decoder.iter()).unwrap();
    writeln!(out, "mil program = ")?;
    writeln!(out, "{:?}", prog)?;

    writeln!(out)?;
    let mut prog = ssa::mil_to_ssa(ssa::ConversionParams::new(prog));
    ssa::eliminate_dead_code(&mut prog);
    xform::fold_constants(&mut prog);
    writeln!(out, "{:?}", prog)?;

    writeln!(out)?;
    let mut pp = PrettyPrinter::start(&mut *out);
    let ast = ast::Builder::new(&prog).compile();
    ast.pretty_print(&mut pp).unwrap();

    Ok(())
}
