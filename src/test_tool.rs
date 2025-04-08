use std::collections::HashMap;

use iced_x86::Formatter;
use thiserror::Error;

use crate::{
    ast,
    pp::{self, PP},
    ssa, ty, x86_to_mil, xform,
};

#[derive(Debug, Error)]
pub enum Error {
    #[error("unsupported executable format: {0}")]
    ExecIo(#[from] crate::elf::Error),

    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),

    #[error("symbol `{0}` is not a function")]
    NotAFunction(String),

    #[error("while parsing DWARF type info: {0}")]
    DwarfTypeParserError(#[from] ty::dwarf::Error),
}
pub type Result<T> = std::result::Result<T, Error>;

pub fn run<W: PP + ?Sized>(raw_binary: &[u8], function_name: &str, out: &mut W) -> Result<()> {
    Tester::start(raw_binary)?.process_function(function_name, out)
}

pub struct Tester<'a> {
    raw_binary: &'a [u8],
    elf: goblin::elf::Elf<'a>,
    func_syms: HashMap<String, AddrRange>,
    types: ty::TypeSet,
}

#[derive(Clone, Copy)]
struct AddrRange {
    base: usize,
    size: usize,
}

impl<'a> Tester<'a> {
    pub fn start(raw_binary: &'a [u8]) -> Result<Self> {
        let elf = crate::elf::parse_elf(raw_binary)?;
        let func_syms = elf
            .syms
            .iter()
            .filter(|sym| sym.is_function())
            .filter_map(|sym| {
                let name = elf.strtab.get_at(sym.st_name)?;
                let base = sym.st_value as usize;
                let size = sym.st_size as usize;
                Some((name.to_owned(), AddrRange { base, size }))
            })
            .collect();

        let mut types = ty::TypeSet::new();
        let dwarf_report = ty::dwarf::load_dwarf_types(&elf, raw_binary, &mut types).unwrap();
        println!(
            "dwarf types parsed with {} errors",
            dwarf_report.errors.len()
        );
        for (ndx, (addr, err)) in dwarf_report.errors.into_iter().enumerate() {
            println!(" #{}: 0x{:08x}: {}", ndx, addr, err);
        }

        Ok(Tester {
            raw_binary,
            elf,
            func_syms,
            types,
        })
    }

    pub fn function_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.func_syms.keys().map(|s| s.as_str())
    }

    pub fn process_function<W: pp::PP + ?Sized>(
        &self,
        function_name: &str,
        out: &mut W,
    ) -> std::result::Result<(), Error> {
        let AddrRange {
            base: func_addr,
            size: func_size,
        } = self
            .func_syms
            .get(function_name)
            .copied()
            .ok_or(Error::NotAFunction(function_name.to_owned()))?;

        let func_end = func_addr + func_size;

        let text_section = self
            .elf
            .section_headers
            .iter()
            .find(|sec| {
                sec.is_executable() && self.elf.shdr_strtab.get_at(sec.sh_name) == Some(".text")
            })
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
        let func_text = &self.raw_binary[func_fofs..func_fofs + func_size];
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
            write!(out, "{:16x}: ", instr.ip())?;
            let ofs = instr.ip() as usize - func_addr;
            let len = instr.len();
            for i in 0..8 {
                if i < len {
                    write!(out, "{:02x} ", func_text[ofs + i])?;
                } else {
                    write!(out, "   ")?;
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
        let prog = {
            let insns = decoder.iter();
            let mut b = x86_to_mil::Builder::new();

            let func_tyid_opt = self.types.get_known_object(func_addr.try_into().unwrap());
            if let Some(func_tyid) = func_tyid_opt {
                let func_typ = self.types.get(func_tyid).unwrap();
                write!(out, "function type: ")?;
                self.types.dump_type(out, &func_typ).unwrap();
                writeln!(out)?;

                let func_ty = match &func_typ.ty {
                    ty::Ty::Subroutine(subr_ty) => subr_ty.clone(),
                    other => panic!(
                        "can't use type ID {:?} type is not a subroutine: {:?}",
                        func_tyid, other
                    ),
                };
                b.use_types(&self.types, func_ty).unwrap();
            } else {
                writeln!(out, "function type: 0x{func_addr:x}: no type info")?;
            }

            b.translate(insns).unwrap()
        };
        writeln!(out, "mil program = ")?;
        writeln!(out, "{:?}", prog)?;

        writeln!(out)?;
        writeln!(out, "ssa pre-xform:")?;
        let mut prog = ssa::mil_to_ssa(ssa::ConversionParams::new(prog));
        writeln!(out, "{:?}", prog)?;

        writeln!(out)?;
        writeln!(out, "cfg:")?;
        let cfg = prog.cfg();
        for bid in cfg.block_ids() {
            writeln!(out, "  {:?} -> {:?}", bid, cfg.block_cont(bid))?;
        }
        writeln!(out, "  domtree:")?;
        cfg.dom_tree().dump(out)?;

        writeln!(out)?;
        writeln!(out, "ssa post-xform:")?;
        xform::canonical(&mut prog);
        writeln!(out, "{:?}", prog)?;

        writeln!(out)?;
        let mut ast = ast::Ast::new(&prog);
        ast.pretty_print(out).unwrap();

        Ok(())
    }
}
