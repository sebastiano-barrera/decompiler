use std::collections::HashMap;
use std::ffi::OsString;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use sha1::Digest;
use thiserror::Error;
use tracing::*;

use crate::{mil, ssa, ty, x86_to_mil, xform};

pub use crate::ast::{precedence, Ast, AstBuilder, PrecedenceLevel};
pub use crate::cfg::{BlockCont, BlockID, BlockMap, Dest, Graph};
pub use crate::mil::{
    to_expanded, AncestralName, ExpandedInsn, ExpandedValue, Insn, Reg, RegType, R,
};
pub use crate::ssa::{count_readers, Program as SSAProgram, RegMap};

pub mod proto;

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

    #[error("type database error: {0}")]
    TypeSetError(#[from] ty::Error),

    #[error("while compiling to MIL: {0}")]
    FrontendError(String),

    #[error("while translating into SSA: {0}")]
    SSAError(String),

    #[error("while applying clarifying transformations: {0}")]
    XformError(String),

    #[error("function memory range (0x{:x}-0x{:x}) out of .text section vm range (0x{:x}-0x{:x})", func_range.0, func_range.1, text_range.0, text_range.1)]
    InvalidRange {
        func_range: (usize, usize),
        text_range: (usize, usize),
    },

    #[error("runtime envionment error: {0}")]
    EnvironmentError(String),
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
    types: ty::TypeSet,
}

#[derive(Clone, Copy)]
struct AddrRange {
    base: usize,
    size: usize,
}

/// Indication of where in the executable a function's machine code is located.
struct FuncCoords {
    /// Offset of the machine code from the ELF's .text section
    #[allow(dead_code)]
    text_section_offset: usize,
    /// Offset of the machine code from the start of the executable file.
    file_offset: usize,
    /// Address of the machine code at runtime, in the process' virtual memory.
    vm_addr: usize,
    /// Size of the original machine code, in bytes.
    code_size: usize,
}

fn path_of_key(key: &[u8]) -> Result<PathBuf> {
    use std::fmt::Write;

    let mut dir = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|p| p.join(".cache"))
        })
        .ok_or(Error::EnvironmentError(
            "environment variables not set: XDG_CACHE_HOME, HOME".to_string(),
        ))?;

    dir = dir.join("decompiler");

    let mut file_name = OsString::with_capacity(3 * 8);
    for (ndx, b) in key.iter().enumerate() {
        let pfx = if ndx == 0 { "" } else { "-" };
        write!(file_name, "{}{:02x}", pfx, b).unwrap();
    }

    Ok(dir.join(&file_name))
}

fn open_typeset(dir_path: &Path) -> ty::Result<(ty::TypeSet, bool)> {
    let is_new = !std::fs::exists(&dir_path).unwrap();
    let types = ty::TypeSet::open(ty::Location::Dir(dir_path))?;
    Ok((types, is_new))
}

impl<'a> Executable<'a> {
    #[instrument(skip_all)]
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

        // the bet is that the binary's hash is unique enough for a key

        let cache_key: [u8; 20] = {
            let mut hasher = sha1::Sha1::new();
            hasher.update(raw_binary);
            hasher.finalize().into()
        };

        let (mut types, is_new) = path_of_key(&cache_key)
            .and_then(|p| open_typeset(&p).map_err(Error::TypeSetError))
            .unwrap_or_else(|err| {
                event!(
                    Level::WARN,
                    ?err,
                    "could not open type cache, using in-memory database"
                );
                let types =
                    ty::TypeSet::open(ty::Location::Memory).expect("creating TypeSet in memory");
                (types, true)
            });

        if is_new {
            event!(Level::INFO, "database is brand new, rescanning executable");
            let result = ty::dwarf::load_dwarf_types(&elf, raw_binary, &mut types);
            event!(Level::WARN, ?result, "failed to parse dwarf types");
        }

        {
            let rtx = types.read_tx()?;
            event!(
                Level::INFO,
                types_count = rtx.read().types_count()?,
                "types loaded"
            );
        }

        Ok(Executable {
            raw_binary,
            elf,
            func_syms,
            types,
        })
    }

    pub fn function_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.func_syms.keys().map(|s| s.as_str())
    }

    pub fn has_function_named(&self, name: &str) -> bool {
        self.func_syms.contains_key(name)
    }

    pub fn types(&self) -> &ty::TypeSet {
        &self.types
    }

    #[instrument(skip(self))]
    pub fn decompile_function(&self, function_name: &str) -> Result<DecompiledFunction> {
        let coords = self.find_function(function_name)?;

        let vm_addr = coords.vm_addr.try_into().unwrap();
        let mut decoder = iced_x86::Decoder::with_ip(
            64,
            self.machine_code(&coords),
            vm_addr,
            iced_x86::DecoderOptions::NONE,
        );

        let types_rtx = self.types.read_tx()?;
        let func_tyid_opt = types_rtx.read().get_known_object(vm_addr)?;
        let mil_res = x86_to_mil::Builder::new(&self.types)?
            .translate(decoder.iter(), func_tyid_opt)
            .map_err(|anyhow_err| Error::FrontendError(anyhow_err.to_string()));

        let mut df = DecompiledFunction {
            function_name: function_name.to_string(),
            coords,
            mil: None,
            ssa_pre_xform: None,
            ssa: None,
            ast: None,
            error: None,
            warnings: Vec::new(),
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

        let mut ssa = match std::panic::catch_unwind(|| ssa::Program::from_mil(mil)) {
            Ok(p) => p,
            Err(err) => {
                df.error = Some(Error::SSAError(panic_message(err)));
                return Ok(df);
            }
        };
        ssa::eliminate_dead_code(&mut ssa);
        df.ssa_pre_xform = Some(ssa.clone());

        xform::canonical(&mut ssa, &self.types);
        let ast = AstBuilder::new(&ssa).build();
        df.ast = Some(ast);
        df.ssa = Some(ssa);

        Ok(df)
    }

    fn find_function(&self, function_name: &str) -> Result<FuncCoords> {
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
            // error: we can't compute offset in text section in this case
            return Err(Error::InvalidRange {
                func_range: (vm_addr, func_end),
                text_range: (vm_range.start, vm_range.end),
            });
        }

        let text_section_offset = vm_addr - vm_range.start;

        let file_offset = text_section.sh_offset as usize + text_section_offset;
        Ok(FuncCoords {
            text_section_offset,
            file_offset,
            vm_addr,
            code_size,
        })
    }

    fn machine_code(&self, coords: &FuncCoords) -> &'a [u8] {
        let FuncCoords {
            file_offset,
            code_size,
            ..
        } = *coords;
        &self.raw_binary[file_offset..file_offset + code_size]
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

    coords: FuncCoords,
    mil: Option<mil::Program>,
    ssa_pre_xform: Option<crate::ssa::Program>,
    ssa: Option<crate::ssa::Program>,
    ast: Option<crate::ast::Ast>,

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

    pub fn disassemble<'e>(&self, exe: &'e Executable) -> iced_x86::Decoder<'e> {
        iced_x86::Decoder::with_ip(
            64,
            self.machine_code(exe),
            self.base_ip(),
            iced_x86::DecoderOptions::NONE,
        )
    }
    pub fn machine_code<'a>(&self, exe: &Executable<'a>) -> &'a [u8] {
        exe.machine_code(&self.coords)
    }
    pub fn base_ip(&self) -> u64 {
        self.coords.vm_addr.try_into().unwrap()
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

    pub fn ast(&self) -> Option<&crate::ast::Ast> {
        self.ast.as_ref()
    }
}
