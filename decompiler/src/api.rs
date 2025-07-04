use std::{collections::HashMap, sync::Arc};

use iced_x86::Formatter;
use thiserror::Error;

use crate::{ast, mil, pp, ssa, ty, x86_to_mil, xform};

pub use crate::ast::{precedence, PrecedenceLevel};
pub use crate::cfg::{BlockCont, BlockID, BlockMap, Dest};
pub use crate::mil::{to_expanded, ExpandedInsn, ExpandedValue, Insn, Reg, RegType};
pub use crate::ssa::{count_readers, Program as SSAProgram, RegMap};

#[derive(Debug, Error)]
pub enum Error {
    #[error("unsupported executable format: {0}")]
    ExecIo(#[from] crate::elf::Error),

    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),

    #[error("symbol `{0}` is not a function")]
    NotAFunction(String),

    #[error("no .text section?!")]
    NoTextSection,

    #[error("can't use type ID {0:?} as subroutine type")]
    NotASubroutineType(ty::TypeID),

    #[error("while parsing DWARF type info: {0}")]
    DwarfTypeParserError(#[from] ty::dwarf::Error),

    #[error("while compiling to MIL: {0}")]
    FrontendError(String),

    #[error("while translating into SSA: {0}")]
    SSAError(String),

    #[error("while applying clarifying transformations: {0}")]
    XformError(String),
}
pub type Result<T> = std::result::Result<T, Error>;

/// Represents an executable analyzed by the decompiler.
///
/// An [Executable] borrows the executable's raw content in ELF format as a
/// bytes slice (`&'a [u8]`).
///
/// This is the main entry point for the decompiler's API.
///
/// The typical usage is:
///  - Create an object via [`Executable::parse`]
///  - Check the list of functions via [`Executable::function_names`]
///  - Decompile a given function via [`Executable::process_function`]
pub struct Executable<'a> {
    raw_binary: &'a [u8],
    elf: goblin::elf::Elf<'a>,
    func_syms: HashMap<String, AddrRange>,
    types: Arc<ty::TypeSet>,
}

#[derive(Clone, Copy)]
struct AddrRange {
    base: usize,
    size: usize,
}

impl<'a> Executable<'a> {
    pub fn parse(raw_binary: &'a [u8]) -> Result<Self> {
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

        Ok(Executable {
            raw_binary,
            elf,
            func_syms,
            types: Arc::new(types),
        })
    }

    pub fn function_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.func_syms.keys().map(|s| s.as_str())
    }

    pub fn has_function_named(&self, name: &str) -> bool {
        self.func_syms.contains_key(name)
    }

    // This function is legacy. Only used by the test tool.
    //
    // We should change `decompile_function` so that it is good enough for
    // test_tool, then migrate the latter.
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
        let func_fofs = text_section.sh_offset as usize + func_section_ofs;
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
            let b = x86_to_mil::Builder::new(Arc::clone(&self.types));

            let func_tyid_opt = self.types.get_known_object(func_addr.try_into().unwrap());
            let func_ty = if let Some(func_tyid) = func_tyid_opt {
                let func_typ = self.types.get_through_alias(func_tyid).unwrap();
                write!(out, "function type: ")?;
                self.types.dump_type(out, func_typ).unwrap();
                writeln!(out)?;

                match &func_typ.ty {
                    ty::Ty::Subroutine(subr_ty) => Some(subr_ty),
                    other => panic!(
                        "can't use type ID {:?} type is not a subroutine: {:?}",
                        func_tyid, other
                    ),
                }
            } else {
                writeln!(out, "function type: 0x{func_addr:x}: no type info")?;
                None
            };

            let (prog, warnings) = b.translate(insns, func_ty).unwrap();
            writeln!(out, "{:?}", warnings)?;
            writeln!(out)?;

            prog
        };
        writeln!(out, "mil program = ")?;
        writeln!(out, "{:?}", prog)?;

        writeln!(out)?;
        writeln!(out, "ssa pre-xform:")?;
        let mut prog = ssa::mil_to_ssa(ssa::ConversionParams::new(prog));
        ssa::eliminate_dead_code(&mut prog);
        writeln!(out, "{:?}", prog)?;

        writeln!(out)?;
        writeln!(out, "cfg:")?;
        let cfg = prog.cfg();
        writeln!(out, "  entry: {:?}", cfg.direct().entry_bid())?;
        for bid in cfg.block_ids() {
            let regs: Vec<_> = prog.block_regs(bid).collect();
            writeln!(out, "  {:?} -> {:?} {:?}", bid, cfg.block_cont(bid), regs)?;
        }
        write!(out, "  domtree:\n    ")?;
        out.open_box();
        cfg.dom_tree().dump(out)?;
        out.close_box();

        writeln!(out)?;
        writeln!(out, "ssa post-xform:")?;
        xform::canonical(&mut prog);
        writeln!(out, "{:?}", prog)?;

        writeln!(out)?;
        let mut ast = ast::Ast::new(&prog);
        ast.pretty_print(out).unwrap();

        Ok(())
    }

    pub fn decompile_function(&self, function_name: &str) -> Result<DecompiledFunction> {
        let mut df = self.find_function(function_name)?;

        let mut decoder = df.disassemble(self);
        let mil_res = std::panic::catch_unwind(move || {
            let b = x86_to_mil::Builder::new(Arc::clone(&self.types));

            let func_tyid_opt = self.types.get_known_object(df.vm_addr.try_into().unwrap());
            let func_ty = match func_tyid_opt {
                Some(func_tyid) => {
                    let func_typ = self.types.get_through_alias(func_tyid).unwrap();
                    match &func_typ.ty {
                        ty::Ty::Subroutine(subr_ty) => Some(subr_ty),
                        _ => {
                            return Err(Error::NotASubroutineType(func_tyid));
                        }
                    }
                }
                None => None,
            };

            b.translate(decoder.iter(), func_ty)
                .map_err(|anyhow_err| Error::FrontendError(anyhow_err.to_string()))
        });

        // fold a panic into a regular error
        let mil_res = match mil_res {
            Ok(Ok(res)) => Ok(res),
            Ok(Err(err)) => Err(err),
            Err(panic_err) => Err(Error::FrontendError(panic_message(panic_err))),
        };

        let (mil, warnings) = match mil_res {
            Ok(mw) => mw,
            Err(err) => {
                df.error = Some(err);
                return Ok(df);
            }
        };

        df.mil = Some(mil.clone());
        df.warnings.extend(
            warnings
                .into_vec()
                .into_iter()
                .map(|err| Error::FrontendError(err.to_string())),
        );

        let mut ssa =
            match std::panic::catch_unwind(|| ssa::mil_to_ssa(ssa::ConversionParams::new(mil))) {
                Ok(p) => p,
                Err(err) => {
                    df.error = Some(Error::SSAError(panic_message(err)));
                    return Ok(df);
                }
            };
        ssa::eliminate_dead_code(&mut ssa);
        df.ssa_pre_xform = Some(ssa.clone());

        let xform_res = std::panic::catch_unwind(move || {
            xform::canonical(&mut ssa);
            ssa
        });
        match xform_res {
            Err(err) => {
                df.warnings.push(Error::XformError(panic_message(err)));
                return Ok(df);
            }
            Ok(ssa) => {
                df.ssa = Some(ssa);
            }
        };

        Ok(df)
    }

    fn find_function(&self, function_name: &str) -> Result<DecompiledFunction> {
        let AddrRange {
            base: vm_addr,
            size: code_size,
        } = self
            .func_syms
            .get(function_name)
            .copied()
            .ok_or(Error::NotAFunction(function_name.to_owned()))?;
        let func_end = vm_addr + code_size;
        let text_section = self
            .elf
            .section_headers
            .iter()
            .find(|sec| {
                sec.is_executable() && self.elf.shdr_strtab.get_at(sec.sh_name) == Some(".text")
            })
            .ok_or(Error::NoTextSection)?;
        let vm_range = text_section.vm_range();
        if vm_range.start > vm_addr || vm_range.end < func_end {
            eprintln!(
                "function memory range (0x{:x}-0x{:x}) out of .text section vm range (0x{:x}-0x{:x})",
                vm_addr, func_end, vm_range.start, vm_range.end,
            );
        }
        let text_section_offset = vm_addr - vm_range.start;
        let file_offset = text_section.sh_offset as usize + text_section_offset;
        Ok(DecompiledFunction {
            function_name: function_name.to_string(),
            text_section_offset,
            file_offset,
            vm_addr,
            code_size,
            mil: None,
            ssa_pre_xform: None,
            ssa: None,
            error: None,
            warnings: Vec::new(),
        })
    }
}

fn panic_message(panic_err: Box<dyn std::any::Any + Send>) -> String {
    let msg = if let Some(err_str) = panic_err.downcast_ref::<String>() {
        err_str.clone()
    } else {
        "(no description)".to_string()
    };
    msg
}

pub struct DecompiledFunction {
    function_name: String,

    /// Offset of the machine code from the ELF's .text section
    text_section_offset: usize,
    /// Offset of the machine code from the start of the executable file.
    file_offset: usize,
    /// Address of the machine code at runtime, in the process' virtual memory.
    vm_addr: usize,

    /// Size of the original machine code, in bytes.
    code_size: usize,

    mil: Option<mil::Program>,
    ssa_pre_xform: Option<crate::ssa::Program>,
    ssa: Option<crate::ssa::Program>,

    error: Option<Error>,
    warnings: Vec<Error>,
}

impl DecompiledFunction {
    pub fn name(&self) -> &str {
        &self.function_name
    }

    pub fn error(&self) -> Option<&Error> {
        self.error.as_ref()
    }

    pub fn warnings(&self) -> &[Error] {
        self.warnings.as_slice()
    }

    pub fn machine_code<'e>(&self, exe: &'e Executable) -> &'e [u8] {
        &exe.raw_binary[self.file_offset..self.file_offset + self.code_size]
    }

    pub fn disassemble<'e>(&self, exe: &'e Executable) -> iced_x86::Decoder<'e> {
        iced_x86::Decoder::with_ip(
            64,
            self.machine_code(exe),
            self.vm_addr.try_into().unwrap(),
            iced_x86::DecoderOptions::NONE,
        )
    }

    pub fn mil(&self) -> Option<&mil::Program> {
        self.mil.as_ref()
    }

    pub fn ssa_pre_xform(&self) -> Option<&crate::ssa::Program> {
        self.ssa_pre_xform.as_ref()
    }

    pub fn ssa(&self) -> Option<&crate::ssa::Program> {
        self.ssa.as_ref()
    }
}
