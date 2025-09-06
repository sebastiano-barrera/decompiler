use std::{cell::RefCell, sync::Arc};

use crate::ty;

use gimli::EndianSlice;
use thiserror::Error;
use tracing::{event, instrument, span, Level};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error, PartialEq, Eq)]
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

    let mut units = dwarf.debug_info.units();

    let mut is_empty = true;
    while let Some(unit_hdr) = units.next()? {
        is_empty = false;

        let unit = dwarf.unit(unit_hdr)?;
        let parser = TypeParser {
            dwarf: &dwarf,
            errors: RefCell::new(Vec::new()),
            unit: &unit,
            types,
        };
        let unit_errors = parser.load_types()?;

        errors.extend(unit_errors);
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
    errors: RefCell<Errors>,
    unit: &'a gimli::Unit<ESlice<'a>>,
    types: &'a mut ty::TypeSet,
}
type Errors = Vec<(usize, Error)>;

#[allow(clippy::upper_case_acronyms)]
type DIE<'a, 'abbrev, 'unit> = gimli::DebuggingInformationEntry<'abbrev, 'unit, ESlice<'a>, usize>;

impl<'a> TypeParser<'a> {
    fn load_types(mut self) -> Result<Errors> {
        let mut entries = self.unit.entries_tree(None)?;
        let root = entries.root()?;

        assert_eq!(root.entry().tag(), gimli::DW_TAG_compile_unit);

        let mut children = root.children();
        while let Some(node) = children.next()? {
            let node_ofs = node.entry().offset().0;

            let span = span!(Level::DEBUG, "dwarf-node", node_ofs);
            let _enter = span.enter();

            match self.try_parse_type(node, self.unit) {
                Ok(_) => {}
                Err(Error::UnsupportedDwarfTag(tag)) => {
                    let tag = tag.static_string().unwrap_or("?");
                    event!(Level::ERROR, tag, "UnsupportedDwarfTag");
                }
                Err(err) => {
                    event!(Level::ERROR, ?err);
                    let mut errors = self.errors.borrow_mut();
                    errors.push((node_ofs, err));
                }
            }
        }

        let mut entries = self.unit.entries();
        while let Some((_, die)) = entries.next_dfs()? {
            if die.tag() == gimli::constants::DW_TAG_call_site {
                self.parse_call_site(die)?;
            }
        }

        let mut errors = Vec::new();
        let mut self_errors = self.errors.borrow_mut();
        std::mem::swap(&mut errors, &mut *self_errors);
        Ok(errors)
    }

    #[instrument(skip_all)]
    fn try_parse_type(
        &mut self,
        node: GimliNode,
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
        event!(Level::TRACE, ?tyid);

        let res = match entry.tag() {
            // tag types I'm going to support, least to most common:
            // - [ ] DW_TAG_volatile_type
            // - [x] DW_TAG_restrict_type
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
            gimli::constants::DW_TAG_structure_type => self
                .parse_struct_type(node)
                .map(|(name, ty)| (Some(name), ty)),
            gimli::constants::DW_TAG_pointer_type => {
                self.parse_pointer_type(node).map(|ty| (None, ty))
            }
            // subprograms (functions, in C) are considered as subroutine types
            // with a only single instance existing in the program.
            gimli::constants::DW_TAG_subprogram | gimli::constants::DW_TAG_subroutine_type => {
                // Share the same TypeID for all concrete out-of-line abstract instances of a subprogram
                let attr_value_opt = entry.attr_value(gimli::constants::DW_AT_abstract_origin)?;

                if let (gimli::constants::DW_TAG_subprogram, Some(attr_value)) =
                    (entry.tag(), attr_value_opt)
                {
                    let tyid = self.resolve_reference(attr_value, diofs)?;
                    let name = self.get_name(entry)?.map(|s| s.to_owned());
                    Ok((name, ty::Ty::Alias(tyid)))
                } else {
                    self.parse_subroutine_type(node)
                        .map(|(name, ty)| (Some(name), ty))
                }
            }
            gimli::constants::DW_TAG_base_type => self.parse_base_type(node),
            gimli::constants::DW_TAG_typedef => self.parse_alias(node),

            // we currently don't care about the `restrict` attribute at all
            gimli::constants::DW_TAG_restrict_type => self.parse_alias(node),

            other => Err(Error::UnsupportedDwarfTag(other)),
        };

        // otherwise, leave tyid assigned to a clone of default_type
        match res {
            Err(err) => {
                // still ensure that TypeID is associated to a type
                event!(Level::TRACE, ?err, "error parsing type, assigning Unknown");
                self.types
                    .set(tyid, ty::Ty::Unknown(ty::Unknown { size: None }))
                    .unwrap();

                Err(err)
            }
            Ok((name, ty)) => {
                if !matches!(ty, ty::Ty::Unknown(_)) {
                    if let Some(addr_attrvalue) = addr_av {
                        let addr = self.dwarf.attr_address(unit, addr_attrvalue)?.ok_or(
                            Error::InvalidValueType(diofs.0, gimli::constants::DW_AT_low_pc),
                        )?;
                        self.types.set_known_object(addr, tyid);
                    }
                }

                // unwrap(): it's a programming error if set() fails here because that
                // would mean that tyid is non-regular (read-only)
                self.types.set(tyid, ty).unwrap();
                if let Some(name) = name {
                    self.types.set_name(tyid, name);
                }

                Ok(tyid)
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

    fn parse_subroutine_type(&mut self, node: GimliNode) -> Result<(String, ty::Ty)> {
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
            Err(Error::MissingRequiredAttr(gimli::DW_AT_type)) => Ok(self.types.tyid_shared_void()),
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
        let ty = ty::Ty::Subroutine(ty::Subroutine {
            return_tyid,
            param_names,
            param_tyids,
        });
        Ok((name, ty))
    }

    fn parse_pointer_type(&mut self, node: GimliNode) -> Result<ty::Ty> {
        let entry = node.entry();
        assert_eq!(entry.tag(), gimli::DW_TAG_pointer_type);

        let res = self.resolve_type_of(entry);

        // special case: C's `void*` is represented as a DW_TAG_pointer_type without DW_AT_type
        let pointee_tyid = match res {
            Err(Error::MissingRequiredAttr(gimli::DW_AT_type)) => self.types.tyid_shared_void(),
            _ => res?,
        };

        Ok(ty::Ty::Ptr(pointee_tyid))
    }

    fn parse_struct_type(&mut self, node: GimliNode) -> Result<(String, ty::Ty)> {
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

        Ok((name, ty::Ty::Struct(ty::Struct { members, size })))
    }

    fn parse_struct_member(&mut self, member: &DIE<'_, '_, '_>) -> Result<ty::StructMember> {
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

    fn parse_base_type(&mut self, node: GimliNode) -> Result<(Option<String>, ty::Ty)> {
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

        let name = self.get_name(entry)?.map(|s| s.to_owned());
        Ok((name, ty))
    }

    fn parse_alias(&mut self, node: GimliNode) -> Result<(Option<String>, ty::Ty)> {
        let entry = node.entry();
        let ref_tyid = self.resolve_type_of(entry)?;
        let name = self.get_name(entry)?.map(|s| s.to_owned());
        Ok((name, ty::Ty::Alias(ref_tyid)))
    }

    fn parse_call_site(&mut self, entry: &DIE) -> Result<()> {
        let diofs = self.entry_diofs(entry);

        assert_eq!(entry.tag(), gimli::constants::DW_TAG_call_site);

        let Some(call_origin) = entry.attr_value(gimli::constants::DW_AT_call_origin)? else {
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

        let tyid = self.resolve_reference(call_origin, diofs)?;

        self.types.add_call_site_by_return_pc(return_pc, tyid);
        Ok(())
    }

    fn resolve_type_of(&mut self, entry_with_type: &DIE<'_, '_, '_>) -> Result<ty::TypeID> {
        let attr_value = get_required_attr(entry_with_type, gimli::DW_AT_type)?.value();
        self.resolve_reference(
            attr_value,
            uofs_to_diofs(entry_with_type.offset(), self.unit),
        )
    }

    fn resolve_reference(
        &mut self,
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

    /// Get the TypeID corresponding to the given DIE (the one at the given
    /// offset) in the current compilation unit.
    ///
    /// If no such association exists yet, the DIE is associated to a new distinct unsized ty::Ty::Unknown.
    ///
    /// In all cases, the TypeID is returned.
    // TODO Remove the Result<_> from the return type
    fn get_tyid(&mut self, type_unit_offset: DebugInfoOffset) -> Result<ty::TypeID> {
        let rtyid = type_unit_offset.0 .0;
        let tyid = ty::TypeID::Regular(rtyid);
        self.types
            .get_or_create(tyid, || ty::Ty::Unknown(ty::Unknown { size: None }))
            .unwrap();
        Ok(tyid)
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
    use super::super::tests::DATA_DIR;
    use super::*;

    fn load_test_elf(rel_path: &str) -> (goblin::elf::Elf<'static>, &'static [u8]) {
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
        assert_eq!(&report.errors, &[]);
        insta::assert_debug_snapshot!(types);
    }
}
