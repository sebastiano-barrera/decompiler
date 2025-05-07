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
    Parser(#[from] gimli::Error),

    #[error("unsupported DWARF entry tag: {0}")]
    UnsupportedDwarfTag(gimli::DwTag),

    #[error("required attribute is missing: {} ({})", .0.static_string().unwrap_or("?"), .0)]
    MissingRequiredAttr(gimli::DwAt),

    #[error("data type discarded due to exceeding maximum supported size")]
    TypeTooLarge,

    #[error("DIE at {0:?} has wrong type of value for attribute {1}")]
    InvalidValueType(gimli::DebugInfoOffset<usize>, gimli::DwAt),

    #[error("unsupported DWARF feature: {0}")]
    UnsupportedFeature(String),

    #[error("no type ID for DIE at {0:?}")]
    BrokenLink(gimli::DebugInfoOffset<usize>),
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

    let default_type = ty::Type {
        name: Arc::new("$unprocessed$".to_string()),
        ty: ty::Ty::Unknown(ty::Unknown { size: 0 }),
    };
    let tyid_of_node = allocate_tyids(&dwarf, types, &default_type)?;

    let mut units = dwarf.debug_info.units();

    let mut is_empty = true;
    while let Some(unit_hdr) = units.next()? {
        is_empty = false;

        let unit = dwarf.unit(unit_hdr)?;
        let parser = TypeParser {
            dwarf: &dwarf,
            errors: RefCell::new(Vec::new()),
            tyid_of_node: &tyid_of_node,
            unit: &unit,
        };
        parser.load_types(types)?;

        {
            let mut entries = unit.entries();
            while let Some((_, die)) = entries.next_dfs()? {
                if die.tag() == gimli::constants::DW_TAG_call_site {
                    parser.parse_call_site(die, types)?;
                }
            }
        }

        errors.extend(parser.errors.take());
    }

    if is_empty {
        return Err(Error::NoCompileUnit);
    }

    types.assert_invariants();
    Ok(Report { errors })
}

type ESlice<'d> = EndianSlice<'d, gimli::RunTimeEndian>;
type GimliNode<'a, 'b, 'c, 'd> = gimli::EntriesTreeNode<'a, 'b, 'c, ESlice<'d>>;

struct TypeParser<'a> {
    dwarf: &'a gimli::Dwarf<ESlice<'a>>,
    errors: RefCell<Vec<(usize, Error)>>,
    tyid_of_node: &'a HashMap<DebugInfoOffset, ty::TypeID>,
    unit: &'a gimli::Unit<ESlice<'a>>,
}

#[allow(clippy::upper_case_acronyms)]
type DIE<'a, 'abbrev, 'unit> = gimli::DebuggingInformationEntry<'abbrev, 'unit, ESlice<'a>, usize>;

impl<'a> TypeParser<'a> {
    fn load_types(&self, types: &mut ty::TypeSet) -> Result<()> {
        assert!(!self.tyid_of_node.is_empty());

        let mut entries = self.unit.entries_tree(None)?;
        let root = entries.root()?;

        assert_eq!(root.entry().tag(), gimli::DW_TAG_compile_unit);

        let mut children = root.children();
        while let Some(node) = children.next()? {
            let node_ofs = node.entry().offset().0;

            match self.try_parse_type(node, types, self.unit) {
                Ok(_) => {}
                Err(Error::UnsupportedDwarfTag(_)) => {}
                Err(other_err) => {
                    let mut errors = self.errors.borrow_mut();
                    errors.push((node_ofs, other_err));
                }
            }
        }

        Ok(())
    }

    fn try_parse_type(
        &self,
        node: GimliNode,
        types: &mut ty::TypeSet,
        unit: &gimli::Unit<ESlice>,
    ) -> Result<ty::TypeID> {
        let entry = node.entry();
        let addr_av = entry.attr_value(gimli::constants::DW_AT_low_pc)?;

        let diofs = entry
            .offset()
            .to_debug_info_offset(&unit.header)
            .expect(ERRMSG_DEBUG_INFO)
            .into();
        let tyid = self.get_tyid(diofs)?;

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
            // - [x] DW_TAG_typedef
            gimli::constants::DW_TAG_structure_type => self.parse_struct_type(node),
            gimli::constants::DW_TAG_pointer_type => self.parse_pointer_type(node, types),
            // subprograms (functions, in C) are considered as subroutine types
            // with a only single instance existing in the program.
            gimli::constants::DW_TAG_subprogram | gimli::constants::DW_TAG_subroutine_type => {
                // Share the same TypeID for all concrete out-of-line abstract instances of a subprogram
                if let (gimli::constants::DW_TAG_subprogram, Some(attr_value)) = (
                    entry.tag(),
                    entry.attr_value(gimli::constants::DW_AT_abstract_origin)?,
                ) {
                    let tyid = self.resolve_reference(attr_value, diofs)?;
                    Ok(ty::Type {
                        name: Arc::new(String::new()),
                        ty: ty::Ty::Alias(tyid),
                    })
                } else {
                    self.parse_subroutine_type(node, types)
                }
            }
            gimli::constants::DW_TAG_base_type => self.parse_base_type(node, types),
            gimli::constants::DW_TAG_typedef => self.parse_alias(node),

            other => Err(Error::UnsupportedDwarfTag(other)),
        };

        // otherwise, leave tyid assigned to a clone of default_type
        let typ = res?;

        if !matches!(&typ.ty, ty::Ty::Unknown(_)) {
            if let Some(addr_attrvalue) = addr_av {
                let addr = self.dwarf.attr_address(unit, addr_attrvalue)?.ok_or(
                    Error::InvalidValueType(diofs.0, gimli::constants::DW_AT_low_pc),
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
        let return_tyid = match self.resolve_type_of(entry) {
            Err(Error::MissingRequiredAttr(gimli::DW_AT_type)) => Ok(types.tyid_void()),
            res => res,
        }?;

        let mut param_names = Vec::new();
        let mut param_tyids = Vec::new();
        let mut children = node.children();
        while let Some(child_node) = children.next()? {
            let child_entry = child_node.entry();
            if child_entry.tag() == gimli::DW_TAG_formal_parameter {
                let name = self.get_name(child_entry)?.map(|s| Arc::new(s.to_owned()));
                let tyid = self.resolve_type_of(child_entry)?;
                param_names.push(name);
                param_tyids.push(tyid);
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

        let res = self.resolve_type_of(entry);

        // special case: C's `void*` is represented as a DW_TAG_pointer_type without DW_AT_type
        let pointee_tyid = match res {
            Err(Error::MissingRequiredAttr(gimli::DW_AT_type)) => types.tyid_void(),
            _ => res?,
        };

        Ok(ty::Type {
            // TODO make name optional
            name: Arc::new(String::new()),
            ty: ty::Ty::Ptr(pointee_tyid),
        })
    }

    fn parse_struct_type(&self, node: GimliNode) -> Result<ty::Type> {
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
                self.entry_diofs(entry).0,
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
                    let res = self.parse_struct_member(member);
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

    fn parse_struct_member(&self, member: &DIE<'_, '_, '_>) -> Result<ty::StructMember> {
        assert_eq!(member.tag(), gimli::DW_TAG_member);

        let offset = get_required_attr(member, gimli::DW_AT_data_member_location)?
            .udata_value()
            .ok_or(Error::InvalidValueType(
                self.entry_diofs(member).0,
                gimli::DW_AT_data_member_location,
            ))?
            .try_into()
            .map_err(|_| Error::TypeTooLarge)?;

        let name = Arc::new(
            self.get_name(member)?
                .map(|s| s.to_owned())
                .unwrap_or_else(String::new),
        );

        let tyid = self.resolve_type_of(member)?;
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
            _ => {
                return Err(Error::InvalidValueType(
                    self.entry_diofs(entry).0,
                    attr.name(),
                ))
            }
        };

        let size = get_required_attr(entry, gimli::DW_AT_byte_size)?
            .udata_value()
            .ok_or(Error::InvalidValueType(
                self.entry_diofs(entry).0,
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

    fn parse_alias(&self, node: GimliNode) -> Result<ty::Type> {
        let entry = node.entry();
        let ref_tyid = self.resolve_type_of(entry)?;
        let name = self.get_name(entry)?;
        Ok(ty::Type {
            name: Arc::new(name.unwrap_or("").to_owned()),
            ty: ty::Ty::Alias(ref_tyid),
        })
    }

    fn parse_call_site(&self, entry: &DIE, types: &mut ty::TypeSet) -> Result<()> {
        let diofs = self.entry_diofs(entry);

        assert_eq!(entry.tag(), gimli::constants::DW_TAG_call_site);

        let Some(attr_value) = entry.attr_value(gimli::constants::DW_AT_call_origin)? else {
            return Ok(());
        };
        let Some(return_pc) = entry.attr_value(gimli::constants::DW_AT_call_return_pc)? else {
            return Ok(());
        };

        let gimli::AttributeValue::Addr(return_pc) = return_pc else {
            return Err(Error::InvalidValueType(
                diofs.0,
                gimli::constants::DW_AT_call_origin,
            ));
        };

        let tyid = self.resolve_reference(attr_value, diofs)?;

        types.add_call_site_by_return_pc(return_pc, tyid);
        Ok(())
    }

    fn resolve_type_of(&self, entry_with_type: &DIE<'_, '_, '_>) -> Result<ty::TypeID> {
        let attr_value = get_required_attr(entry_with_type, gimli::DW_AT_type)?.value();
        self.resolve_reference(
            attr_value,
            uofs_to_diofs(entry_with_type.offset(), self.unit),
        )
    }

    fn resolve_reference(
        &self,
        attr_value: gimli::AttributeValue<ESlice>,
        diofs: DebugInfoOffset,
    ) -> Result<ty::TypeID> {
        let type_unit_offset =
            attr_value_as_diofs(attr_value, self.unit).ok_or(Error::InvalidValueType(
                diofs.0,
                // TODO this is wrong, different call sites may be passed
                // attributes with different tags
                gimli::constants::DW_AT_abstract_origin,
            ))?;
        self.get_tyid(type_unit_offset)
    }

    fn get_tyid(&self, type_unit_offset: DebugInfoOffset) -> Result<ty::TypeID> {
        self.tyid_of_node
            .get(&type_unit_offset)
            .ok_or(Error::BrokenLink(type_unit_offset.0))
            .copied()
    }

    // just a utility to convert a DIE's offset in either UnitOffset or DebugInfoOffset form into DebugInfoOffset
    fn entry_diofs(
        &self,
        entry: &gimli::DebuggingInformationEntry<ESlice, usize>,
    ) -> DebugInfoOffset {
        uofs_to_diofs(entry.offset(), self.unit)
    }

    fn get_name<'abbrev>(&self, entry: &DIE<'abbrev, '_, '_>) -> Result<Option<&'abbrev str>>
    where
        'a: 'abbrev,
    {
        let Some(attr) = entry.attr(gimli::DW_AT_name)? else {
            return Ok(None);
        };
        let name = self
            .dwarf
            .attr_string(self.unit, attr.value())
            .map_err(|_| Error::InvalidValueType(self.entry_diofs(entry).0, gimli::DW_AT_name))?
            .slice();
        // TODO lift the utf8 encoding restriction
        Ok(Some(std::str::from_utf8(name).unwrap()))
    }
}

fn allocate_tyids(
    dwarf: &gimli::Dwarf<ESlice>,
    types: &mut ty::TypeSet,
    default_type: &ty::Type,
) -> Result<HashMap<DebugInfoOffset, ty::TypeID>> {
    let mut map = HashMap::new();

    let mut units = dwarf.debug_info.units();
    while let Some(unit_hdr) = units.next()? {
        let abbreviations = dwarf.abbreviations(&unit_hdr)?;

        let mut entries = unit_hdr.entries(&abbreviations);
        while let Some((_, entry)) = entries.next_dfs()? {
            // TODO filter: only supported DW_TAG_*
            let key = entry
                .offset()
                .to_debug_info_offset(&unit_hdr)
                .expect(ERRMSG_DEBUG_INFO)
                .into();
            let tyid = types.add(default_type.clone());
            map.insert(key, tyid);
        }
    }

    Ok(map)
}

fn attr_value_as_diofs(
    attr_value: gimli::AttributeValue<ESlice, usize>,
    unit: &gimli::Unit<ESlice, usize>,
) -> Option<DebugInfoOffset> {
    match attr_value {
        // (There must be a built-in way to fold these two cases into an offset)
        gimli::AttributeValue::DebugInfoRef(ofs) => Some(ofs.into()),
        gimli::AttributeValue::UnitRef(unit_ref) => Some(uofs_to_diofs(unit_ref, unit)),
        _ => None,
    }
}

fn uofs_to_diofs(unit_ref: gimli::UnitOffset, unit: &gimli::Unit<ESlice>) -> DebugInfoOffset {
    unit_ref
        .to_debug_info_offset(&unit.header)
        .expect(ERRMSG_DEBUG_INFO)
        .into()
}

const ERRMSG_DEBUG_INFO: &str = "entry must be in .debug_info section (DWARF 4 is unsupported)";

fn get_required_attr<'a>(
    die: &DIE<'a, '_, '_>,
    attr: gimli::DwAt,
) -> Result<gimli::Attribute<ESlice<'a>>> {
    die.attr(attr)?.ok_or(Error::MissingRequiredAttr(attr))
}

// HACK same as gimli::DebugInfoOffset, but can be used as key in HashMaps
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
struct DebugInfoOffset(gimli::DebugInfoOffset<usize>);

impl From<gimli::DebugInfoOffset<usize>> for DebugInfoOffset {
    fn from(value: gimli::DebugInfoOffset<usize>) -> Self {
        Self(value)
    }
}

impl std::ops::Deref for DebugInfoOffset {
    type Target = gimli::DebugInfoOffset<usize>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
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

        let mut buf = Vec::new();
        let mut pp = crate::pp::PrettyPrinter::start(&mut buf);
        types.dump(&mut pp).unwrap();

        use std::io::Write;
        writeln!(buf, "{} non-fatal errors:", report.errors.len()).unwrap();
        for (ofs, err) in &report.errors {
            writeln!(buf, "offset 0x{:8x}: {}", ofs, err).unwrap();
        }

        let buf = String::from_utf8(buf).unwrap();
        assert_snapshot!(buf);
    }
}
