use thiserror::Error;

/// Parse a raw binary into an ELF.
///
/// Returns an error ([Error::UnsupportedExecFormat]) if the binary is detected
/// to be of a different format.
///
/// **WARNING**: (Known issue.) The ELF is "simply" parsed without changing the
/// original payload. This implies that _relocations are not applied_, which in
/// turn causes many sections (including symbol names in DWARF debug info) to
/// be non-functional. (ref: `limitation--no-relocatable`)
pub fn parse_elf(raw_binary: &[u8]) -> Result<goblin::elf::Elf<'_>> {
    let object = goblin::Object::parse(&raw_binary).expect("elf parse error");
    let elf = match object {
        goblin::Object::Elf(elf) => elf,
        _ => return Err(Error::UnsupportedExecFormat(obj_format_name(&object))),
    };

    if elf.dynrelas.len() > 0 || elf.shdr_relocs.len() > 0 || elf.dynrels.len() > 0 {
        eprintln!("WARNING: The given executable contains relocations. These are not going to be applied, so some functionality might misbehave.");
    }
    Ok(elf)
}

pub type Result<T> = std::result::Result<T, Error>;
#[derive(Debug, Error)]
pub enum Error {
    #[error("unsupported executable format: {0}")]
    UnsupportedExecFormat(&'static str),
}

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
