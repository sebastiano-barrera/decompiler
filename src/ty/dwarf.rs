use std::{cell::RefCell, collections::HashMap, sync::Arc};

use crate::ty;

use gimli::EndianSlice;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
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

    #[error("unsupported DWARF feature: {0}")]
    UnsupportedFeature(String),
}

pub struct Report {
    pub errors: Vec<(usize, Error)>,
}

pub fn load_dwarf_types(
    elf: &goblin::elf::Elf,
    raw_elf: &[u8],
    types: &mut ty::TypeSet,
) -> Result<Report> {
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
                Some(&raw_elf[file_range])
            })
            .unwrap_or(&[]);
        Ok(EndianSlice::new(bytes, endianity))
    })
    .unwrap(); // never fails...

    // "unit" means "compilation unit" here
    // we only parse .debug_info, not .debug_types
    // an effort is only made to support DWARF5, not DWARF4
    let mut errors = Vec::new();
    let mut units = dwarf.debug_info.units();

    let mut is_empty = true;
    while let Some(unit_hdr) = units.next()? {
        is_empty = false;

        let unit = dwarf.unit(unit_hdr)?;
        let parser = TypeParser::new(&dwarf, unit);
        parser.load_types(types)?;

        errors.extend(parser.errors.take());
    }

    if is_empty {
        return Err(Error::NoCompileUnit);
    }

    Ok(Report { errors })
}

type ESlice<'d> = EndianSlice<'d, gimli::RunTimeEndian>;
type GimliNode<'a, 'b, 'c, 'd> = gimli::EntriesTreeNode<'a, 'b, 'c, ESlice<'d>>;

struct TypeParser<'a> {
    dwarf: &'a gimli::Dwarf<ESlice<'a>>,
    unit: gimli::Unit<ESlice<'a>, usize>,
    errors: RefCell<Vec<(usize, Error)>>,
    tyid_of_node: RefCell<HashMap<gimli::DebugInfoOffset, ty::TypeID>>,
    default_type: ty::Type,
    void_type: ty::Type,
}

type DIE<'a, 'abbrev, 'unit> = gimli::DebuggingInformationEntry<'abbrev, 'unit, ESlice<'a>, usize>;

impl<'a> TypeParser<'a> {
    fn new(dwarf: &'a gimli::Dwarf<ESlice<'a>>, unit: gimli::Unit<ESlice<'a>, usize>) -> Self {
        TypeParser {
            dwarf,
            unit,
            errors: RefCell::new(Vec::new()),
            tyid_of_node: RefCell::new(HashMap::new()),
            default_type: ty::Type {
                name: Arc::new("unprocessed".to_owned()),
                ty: ty::Ty::Unknown(ty::Unknown { size: 0 }),
            },
            void_type: ty::Type {
                name: Arc::new("void".to_owned()),
                ty: ty::Ty::Void,
            },
        }
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
        let key = node
            .entry()
            .offset()
            .to_debug_info_offset(&self.unit.header)
            .expect("unit must be in .debug_info section");
        let tyid = {
            let mut tyid_of_node = self.tyid_of_node.borrow_mut();
            if let Some(&tyid) = tyid_of_node.get(&key) {
                return Ok(tyid);
            }

            let tyid = types.add(self.default_type.clone());
            tyid_of_node.insert(key, tyid);
            tyid
        };

        let entry = node.entry();
        let offset = entry.offset().0;
        let addr_av = entry.attr_value(gimli::constants::DW_AT_low_pc)?;

        let res = match entry.tag() {
            // tag types I'm going to support, least to most common:
            // - [ ] DW_TAG_volatile_type
            // - [ ] DW_TAG_restrict_type
            // - [ ] DW_TAG_atomic_type
            // - [ ] DW_TAG_union_type
            // - [ ] DW_TAG_enumeration_type
            // - [ ] DW_TAG_const_type
            // - [ ] DW_TAG_array_type
            // - [ ] DW_TAG_subrange_type
            // - [x] DW_TAG_subroutine_type
            // - [x] DW_TAG_structure_type
            // - [x] DW_TAG_pointer_type
            // - [x] DW_TAG_base_type
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

        // otherwise, leave tyid assigned to a clone of default_type
        let typ = res?;

        if !matches!(&typ.ty, ty::Ty::Unknown(_)) {
            if let Some(addr_attrvalue) = addr_av {
                let addr = self.dwarf.attr_address(&self.unit, addr_attrvalue)?.ok_or(
                    Error::InvalidValueType(offset, gimli::constants::DW_AT_low_pc),
                )?;
                types.set_known_object(addr, tyid);
            }
        }

        types.set(tyid, typ);
        Ok(tyid)
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
            Err(Error::MissingRequiredAttr(gimli::DW_AT_type)) => {
                Ok(types.add(self.void_type.clone()))
            }
            res => res,
        }?;

        let mut param_names = Vec::new();
        let mut param_tyids = Vec::new();
        let mut children = node.children();
        while let Some(child_node) = children.next()? {
            let child_entry = child_node.entry();
            match child_entry.tag() {
                gimli::DW_TAG_formal_parameter => {
                    let name = self.get_name(child_entry)?.map(|s| Arc::new(s.to_owned()));
                    let tyid = self.try_parse_type_of(&child_entry, types)?;
                    param_names.push(name);
                    param_tyids.push(tyid);
                }
                // not supported yet => ignored
                _ => {}
            }
        }

        assert_eq!(param_names.len(), param_tyids.len());
        Ok(ty::Type {
            name: Arc::new(name),
            ty: ty::Ty::Subroutine(ty::Subroutine {
                return_tyid,
                param_names,
                param_tyids,
            }),
        })
    }

    fn parse_pointer_type(&self, node: GimliNode, types: &mut ty::TypeSet) -> Result<ty::Type> {
        let entry = node.entry();
        assert_eq!(entry.tag(), gimli::DW_TAG_pointer_type);

        let pointee_tyid = self.try_parse_type_of(entry, types)?;
        Ok(ty::Type {
            // TODO make name optional
            name: Arc::new(String::new()),
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
            name: Arc::new(name),
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

        let name = Arc::new(
            self.get_name(member)?
                .map(|s| s.to_owned())
                .unwrap_or_else(String::new),
        );

        let tyid = self.try_parse_type_of(member, types)?;
        Ok(ty::StructMember { offset, name, tyid })
    }

    fn parse_base_type(&self, node: GimliNode, _types: &mut ty::TypeSet) -> Result<ty::Type> {
        let entry = node.entry();
        assert_eq!(entry.tag(), gimli::constants::DW_TAG_base_type);

        if entry.attr(gimli::DW_AT_bit_size)?.is_some()
            || entry.attr(gimli::DW_AT_data_bit_offset)?.is_some()
        {
            return Err(Error::UnsupportedFeature(
                "bit field (base type with bit_size and/or data_bit_offset)".to_owned(),
            ));
        }

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
                let msg = format!(
                    "base type encoding {} ({})",
                    enc.0,
                    enc.static_string().unwrap_or("?")
                );
                return Err(Error::UnsupportedFeature(msg));
            }
        };

        let name = self.get_name(entry)?.unwrap_or("").to_owned();

        Ok(ty::Type {
            name: Arc::new(name),
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

    fn load_test_elf(rel_path: &str) -> (goblin::elf::Elf, &[u8]) {
        let raw = DATA_DIR.get_file(rel_path).unwrap().contents();
        let object = goblin::Object::parse(&raw).expect("could not parse ELF");
        let elf = match object {
            goblin::Object::Elf(elf) => elf,
            _ => panic!("unsupported exec format: {:?}", object),
        };
        (elf, raw)
    }

    #[test]
    fn struct_type() {
        let (elf, raw) = load_test_elf("test_composite_type.so");
        let mut types = ty::TypeSet::new();
        let report = load_dwarf_types(&elf, raw, &mut types).unwrap();

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
