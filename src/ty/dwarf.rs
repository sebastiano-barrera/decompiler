use std::{any::Any, cell::RefCell, collections::HashMap, rc::Rc};

use crate::ty;

use gimli::EndianSlice;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("unsupported executable format (only ELF is supported)")]
    UnsupportedExecFormat,

    #[error("no compilation unit in DWARF debug info")]
    NoCompileUnit,

    #[error("error while parsing DWARF: {0}")]
    ParserError(#[from] gimli::Error),

    #[error("unsupported DWARF entry tag: {0}")]
    UnsupportedDwarfTag(gimli::DwTag),

    #[error("required attribute is missing: {} ({})", .0.static_string().unwrap_or("?"), .0)]
    MissingRequiredAttr(gimli::DwAt),

    #[error("data type discarded due to exceeding maximum supported size")]
    TypeTooLarge,

    #[error("DIE at offset {0} has wrong type of value for attribute {1}")]
    InvalidValueType(usize, gimli::DwAt),

    #[error("unsupported DWARF feature: {} {} ({})", .0, .1, .2.unwrap_or("?"))]
    UnsupportedFeature(&'static str, gimli::DwAte, Option<&'static str>),
}

pub struct Report {
    errors: Vec<(usize, Error)>,
}

pub fn load_dwarf_types(contents: &[u8], types: &mut ty::TypeSet) -> Result<Report> {
    let parser = TypeParser::new(contents)?;
    parser.load_types(types)?;

    let errors = parser.errors.take();
    Ok(Report { errors })
}

type ESlice<'d> = EndianSlice<'d, gimli::RunTimeEndian>;
type GimliNode<'a, 'b, 'c, 'd> = gimli::EntriesTreeNode<'a, 'b, 'c, ESlice<'d>>;

struct TypeParser<'a> {
    dwarf: gimli::Dwarf<ESlice<'a>>,
    unit: gimli::Unit<ESlice<'a>, usize>,
    errors: RefCell<Vec<(usize, Error)>>,
    tyid_of_node: RefCell<HashMap<gimli::UnitOffset, ty::TypeID>>,
}

type DIE<'a, 'abbrev, 'unit> = gimli::DebuggingInformationEntry<'abbrev, 'unit, ESlice<'a>, usize>;

impl<'a> TypeParser<'a> {
    fn new(contents: &'a [u8]) -> Result<Self> {
        let object = goblin::Object::parse(&contents).expect("could not parse ELF");
        let elf = match object {
            goblin::Object::Elf(elf) => elf,
            _ => return Err(Error::UnsupportedExecFormat),
        };

        let endianity = if elf.little_endian {
            gimli::RunTimeEndian::Little
        } else {
            gimli::RunTimeEndian::Big
        };

        let dwarf = gimli::Dwarf::load(|section| -> std::result::Result<_, ()> {
            let bytes = elf
                .section_headers
                .iter()
                .find(|sec| elf.shdr_strtab.get_at(sec.sh_name) == Some(section.name()))
                .and_then(|sec_hdr| {
                    let file_range = sec_hdr.file_range()?;
                    Some(&contents[file_range])
                })
                .unwrap_or(&[]);
            Ok(EndianSlice::new(bytes, endianity))
        })
        .unwrap(); // never fails...

        // "unit" means "compilation unit" here
        let unit_hdr = {
            let mut units = dwarf.debug_info.units();
            let Some(unit_hdr) = units.next()? else {
                return Err(Error::NoCompileUnit);
            };
            if units.next()?.is_some() {
                eprintln!("warning: the given ELF contains debug info for multiple compilation units. only one will be parsed; the others will be ignored.");
            }
            unit_hdr
        };

        let unit = dwarf.unit(unit_hdr)?;
        Ok(TypeParser {
            dwarf,
            unit,
            errors: RefCell::new(Vec::new()),
            tyid_of_node: RefCell::new(HashMap::new()),
        })
    }

    fn load_types(&self, types: &'a mut ty::TypeSet) -> Result<()> {
        let mut entries = self.unit.entries_tree(None)?;
        let root = entries.root()?;

        assert_eq!(root.entry().tag(), gimli::DW_TAG_compile_unit);

        let mut children = root.children();
        while let Some(node) = children.next()? {
            let node_ofs = node.entry().offset().0;

            match self.try_parse_type(node, types) {
                Ok(_) => {}
                Err(Error::UnsupportedDwarfTag(_)) => {}
                Err(other_err) => {
                    let mut errors = self.errors.borrow_mut();
                    errors.push((node_ofs, other_err));
                }
            }
        }

        types.assert_invariants();
        Ok(())
    }

    fn try_parse_type(&self, node: GimliNode, types: &'a mut ty::TypeSet) -> Result<ty::TypeID> {
        // The parsing might fail, but a TypeID is always associated to the
        // GimliNode.
        //
        // In case of failure, the error is captured in self.errors and type is
        // left as a `ty::Unknown`.
        //
        // This function is memoized via `self.tyid_of_node`.
        let tyid = types.alloc_type_id();
        let key = node.entry().offset();
        {
            let mut tyid_of_node = self.tyid_of_node.borrow_mut();
            if let Some(&tyid) = tyid_of_node.get(&key) {
                return Ok(tyid);
            }
            tyid_of_node.insert(key, tyid);
        }

        let entry = node.entry();

        let res = match entry.tag() {
            // tag types I'm going to support, least to most common:
            // - [ ] DW_TAG_volatile_type
            // - [ ] DW_TAG_restrict_type
            // - [ ] DW_TAG_atomic_type
            // - [ ] DW_TAG_union_type
            // - [ ] DW_TAG_enumeration_type
            // - [ ] DW_TAG_const_type
            // - [ ] DW_TAG_base_type
            // - [ ] DW_TAG_array_type
            // - [ ] DW_TAG_subrange_type
            // - [x] DW_TAG_subroutine_type
            // - [x] DW_TAG_structure_type
            // - [x] DW_TAG_pointer_type
            gimli::constants::DW_TAG_structure_type => self.parse_struct_type(node, types),
            gimli::constants::DW_TAG_pointer_type => self.parse_pointer_type(node, types),
            // subprograms (functions, in C) are considered as subroutine types
            // with a only single instance existing in the program.
            gimli::constants::DW_TAG_subprogram | gimli::constants::DW_TAG_subroutine_type => {
                self.parse_subroutine_type(node, types)
            }
            gimli::constants::DW_TAG_base_type => self.parse_base_type(node, types),

            other => Err(Error::UnsupportedDwarfTag(other)),
        };

        match res {
            Ok(typ) => {
                types.assign(tyid, typ);
                Ok(tyid)
            }
            Err(err) => {
                let name = Rc::new(format!("unknown{}", key.0));
                let ty = ty::Ty::Unknown(ty::Unknown { size: 0 });
                let typ = ty::Type { name, ty };
                types.assign(tyid, typ);
                Err(err)
            }
        }
    }

    fn take_error<T>(&self, offset: usize, res: Result<T>) -> Option<T> {
        match res {
            Ok(res) => Some(res),
            Err(err) => {
                let mut errors = self.errors.borrow_mut();
                errors.push((offset, err));
                None
            }
        }
    }

    fn parse_subroutine_type(&self, node: GimliNode, types: &mut ty::TypeSet) -> Result<ty::Type> {
        let entry = node.entry();
        assert!(
            entry.tag() == gimli::constants::DW_TAG_subroutine_type
                || entry.tag() == gimli::constants::DW_TAG_subprogram
        );

        let name = self
            .get_name(entry)?
            .map(|s| s.to_owned())
            .unwrap_or_else(String::new);
        let return_tyid = match self.try_parse_type_of(entry, types) {
            Err(Error::MissingRequiredAttr(gimli::DW_AT_type)) => Ok(ty::TYID_VOID),
            res => res,
        }?;

        let mut params = Vec::new();
        let mut children = node.children();
        while let Some(child_node) = children.next()? {
            let child_entry = child_node.entry();
            let name = self.get_name(child_entry)?.map(|s| Rc::new(s.to_owned()));
            let tyid = self.try_parse_type_of(&child_entry, types)?;
            params.push(ty::SubroutineParam { name, tyid });
        }

        Ok(ty::Type {
            name: Rc::new(name),
            ty: ty::Ty::Subroutine(ty::Subroutine {
                return_tyid,
                params,
            }),
        })
    }

    fn parse_pointer_type(&self, node: GimliNode, types: &mut ty::TypeSet) -> Result<ty::Type> {
        let entry = node.entry();
        assert_eq!(entry.tag(), gimli::DW_TAG_pointer_type);

        let pointee_tyid = self.try_parse_type_of(entry, types)?;
        Ok(ty::Type {
            // TODO make name optional
            name: Rc::new(String::new()),
            ty: ty::Ty::Ptr(pointee_tyid),
        })
    }

    fn parse_struct_type(&self, node: GimliNode, types: &mut ty::TypeSet) -> Result<ty::Type> {
        let entry = node.entry();
        assert_eq!(entry.tag(), gimli::constants::DW_TAG_structure_type);

        // TODO properly make this name optional (don't fallback to an empty string)
        let name = self
            .get_name(entry)?
            .map(|s| s.to_owned())
            .unwrap_or_else(String::new);
        let size = get_required_attr(entry, gimli::DW_AT_byte_size)?
            .value()
            .udata_value()
            .ok_or(Error::InvalidValueType(
                entry.offset().0,
                gimli::DW_AT_byte_size,
            ))?
            .try_into()
            .map_err(|_| Error::TypeTooLarge)?;
        let mut members = Vec::new();
        let mut children = node.children();
        while let Some(child_node) = children.next()? {
            let member = child_node.entry();
            match member.tag() {
                gimli::DW_TAG_member => {
                    let res = self.parse_struct_member(member, types);
                    if let Some(memb) = self.take_error(member.offset().0, res) {
                        members.push(memb);
                    }
                }
                _ => continue, // considered unsupported
            }
        }

        Ok(ty::Type {
            name: Rc::new(name),
            ty: ty::Ty::Struct(ty::Struct { members, size }),
        })
    }

    fn parse_struct_member(
        &self,
        member: &DIE<'_, '_, '_>,
        types: &mut ty::TypeSet,
    ) -> Result<ty::StructMember> {
        assert_eq!(member.tag(), gimli::DW_TAG_member);

        let offset = get_required_attr(member, gimli::DW_AT_data_member_location)?
            .udata_value()
            .ok_or(Error::InvalidValueType(
                member.offset().0,
                gimli::DW_AT_data_member_location,
            ))?
            .try_into()
            .map_err(|_| Error::TypeTooLarge)?;

        let name = Rc::new(
            self.get_name(member)?
                .map(|s| s.to_owned())
                .unwrap_or_else(String::new),
        );

        let tyid = self.try_parse_type_of(member, types)?;
        Ok(ty::StructMember { offset, name, tyid })
    }

    fn parse_base_type(&self, node: GimliNode, types: &mut ty::TypeSet) -> Result<ty::Type> {
        let entry = node.entry();
        assert_eq!(entry.tag(), gimli::constants::DW_TAG_base_type);

        let attr = get_required_attr(entry, gimli::DW_AT_encoding)?;
        let enc = match attr.value() {
            gimli::AttributeValue::Encoding(enc) => enc,
            _ => return Err(Error::InvalidValueType(entry.offset().0, attr.name())),
        };

        let size = get_required_attr(entry, gimli::DW_AT_byte_size)?
            .udata_value()
            .ok_or(Error::InvalidValueType(
                entry.offset().0,
                gimli::DW_AT_byte_size,
            ))?
            .try_into()
            .map_err(|_| Error::TypeTooLarge)?;

        let ty = {
            // DwAte is not an enum ... why?!
            use gimli::constants::*;

            if enc == DW_ATE_boolean {
                ty::Ty::Bool(ty::Bool { size })
            } else if enc == DW_ATE_float {
                ty::Ty::Float(ty::Float { size })
            } else if enc == DW_ATE_signed || enc == DW_ATE_signed_char {
                ty::Ty::Int(ty::Int {
                    size,
                    signed: ty::Signedness::Signed,
                })
            } else if enc == DW_ATE_unsigned || enc == DW_ATE_unsigned_char {
                ty::Ty::Int(ty::Int {
                    size,
                    signed: ty::Signedness::Unsigned,
                })
            } else {
                return Err(Error::UnsupportedFeature(
                    "base type encoding",
                    enc,
                    enc.static_string(),
                ));
            }
        };

        let name = self.get_name(entry)?.unwrap_or("").to_owned();

        Ok(ty::Type {
            name: Rc::new(name),
            ty,
        })
    }

    fn try_parse_type_of(
        &self,
        entry_with_type: &DIE<'_, '_, '_>,
        types: &mut ty::TypeSet,
    ) -> Result<ty::TypeID> {
        let type_unit_offset = match get_required_attr(entry_with_type, gimli::DW_AT_type)?.value()
        {
            gimli::AttributeValue::UnitRef(ofs) => ofs,
            _ => {
                return Err(Error::InvalidValueType(
                    entry_with_type.offset().0,
                    gimli::DW_AT_type,
                ))
            }
        };
        let mut type_tree = self.unit.entries_tree(Some(type_unit_offset))?;
        let tyid = self.try_parse_type(type_tree.root()?, types)?;
        Ok(tyid)
    }

    fn get_name(&self, entry: &DIE<'a, '_, '_>) -> Result<Option<&'a str>> {
        let Some(attr) = entry.attr(gimli::DW_AT_name)? else {
            return Ok(None);
        };
        let name = self
            .dwarf
            .attr_string(&self.unit, attr.value())
            .map_err(|_| Error::InvalidValueType(entry.offset().0, gimli::DW_AT_name))?
            .slice();
        // TODO lift the utf8 encoding restriction
        Ok(Some(std::str::from_utf8(name).unwrap()))
    }
}

fn get_required_attr<'a>(
    die: &DIE<'a, '_, '_>,
    attr: gimli::DwAt,
) -> Result<gimli::Attribute<ESlice<'a>>> {
    die.attr(attr)?.ok_or(Error::MissingRequiredAttr(attr))
}

#[cfg(test)]
mod tests {
    use insta::assert_snapshot;

    use super::super::tests::DATA_DIR;
    use super::*;

    #[test]
    fn struct_type() {
        let contents = DATA_DIR
            .get_file("test_composite_type.so")
            .unwrap()
            .contents();

        let mut types = ty::TypeSet::new();
        let report = load_dwarf_types(contents, &mut types).unwrap();

        let mut buf = String::new();
        let mut pp = crate::pp::PrettyPrinter::start(&mut buf);
        types.dump(&mut pp).unwrap();

        use std::fmt::Write;
        writeln!(buf, "{} non-fatal errors:", report.errors.len()).unwrap();
        for (ofs, err) in &report.errors {
            writeln!(buf, "offset 0x{:8x}: {}", ofs, err).unwrap();
        }

        assert_snapshot!(buf);
    }
}
