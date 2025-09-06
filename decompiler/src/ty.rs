use std::{collections::HashMap, ops::Range, sync::Arc};

use smallvec::SmallVec;
use thiserror::Error;
use tracing::{event, Level};

pub mod dwarf;

use crate::pp::{self, PP};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum TypeID {
    Regular(RegularTypeID),
    Void,
    UnknownUnsized,
    Unknown { size: u32 },
}

pub type RegularTypeID = usize;

pub type Result<T> = std::result::Result<T, Error>;
#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    #[error("change forbidden; the referred type is read-only")]
    ReadOnly,
}

/// A set of types.
///
/// Each type contained in the set can be referred to via a TypeID.
///
/// Each type in the set may refer to other types defined in the same set using
/// the corresponding TypeID.
///
/// This is the only public API in `ty`. Data about types can be retrieved and
/// manipulated via this API.
pub struct TypeSet {
    types: HashMap<TypeID, Ty>,
    name_of_tyid: HashMap<TypeID, String>,
    known_objects: HashMap<Addr, TypeID>,
    call_sites: CallSites,
}

pub type Addr = u64;

impl TypeSet {
    pub fn new() -> Self {
        let mut types = HashMap::new();
        types.insert(TypeID::Void, Ty::Void);
        types.insert(TypeID::UnknownUnsized, Ty::Unknown(Unknown { size: None }));

        let mut name_of_tyid = HashMap::new();
        name_of_tyid.insert(TypeID::Void, "void".to_owned());
        name_of_tyid.insert(TypeID::UnknownUnsized, "unknown".to_owned());

        TypeSet {
            types,
            name_of_tyid,
            known_objects: HashMap::new(),
            call_sites: CallSites::new(),
        }
    }

    pub fn tyid_shared_void(&self) -> TypeID {
        TypeID::Void
    }
    pub fn tyid_shared_unknown_unsized(&self) -> TypeID {
        TypeID::UnknownUnsized
    }
    pub fn tyid_shared_unknown_of_size(&mut self, size: u32) -> TypeID {
        let tyid = TypeID::Unknown { size };
        if !self.types.contains_key(&tyid) {
            self.types
                .insert(tyid, Ty::Unknown(Unknown { size: Some(size) }));
            self.set_name(tyid, format!("unknown{}", size));
        }
        tyid
    }

    /// Get the name of a type, if it has one.
    ///
    /// Invalid TypeIDs result in None.
    pub fn name(&self, tyid: TypeID) -> Option<&str> {
        self.name_of_tyid.get(&tyid).map(|s| s.as_str())
    }
    /// Set the name of a type.
    pub fn set_name(&mut self, tyid: TypeID, name: String) {
        self.name_of_tyid.insert(tyid, name);
    }
    pub fn unset_name(&mut self, tyid: TypeID) {
        self.name_of_tyid.remove(&tyid);
    }

    /// Add a type to the database using the given TypeID, or change the type it refers to.
    ///
    /// Any name assigned to the type is preserved.
    ///
    /// Non-regular TypeIDs are read-only, so they cannot be redefined (this
    /// function will return an Error::ReadOnly).
    pub fn set(&mut self, tyid: TypeID, ty: Ty) -> Result<()> {
        match tyid {
            TypeID::Regular(_) => {
                self.types.insert(tyid, ty);
                Ok(())
            }
            _ => Err(Error::ReadOnly),
        }
        // the name remains (in self.name_of_tyid).
        // it can be removed separately if needed.
    }

    pub fn get(&self, tyid: TypeID) -> Option<&Ty> {
        self.types.get(&tyid)
    }

    pub fn get_through_alias(&self, mut tyid: TypeID) -> Option<&Ty> {
        loop {
            let ty = self.get(tyid)?;
            if let Ty::Alias(target_tyid) = ty {
                tyid = *target_tyid;
            } else {
                return Some(ty);
            }
        }
    }

    pub fn get_or_create(&mut self, tyid: TypeID, create_fn: impl FnOnce() -> Ty) -> Result<&Ty> {
        if !self.types.contains_key(&tyid) {
            let ty = create_fn();
            self.set(tyid, ty)?;
        }

        let ty = self.get(tyid).unwrap();
        Ok(ty)
    }

    pub fn bytes_size(&self, tyid: TypeID) -> Option<u32> {
        let ty = self.get(tyid)?;
        match ty {
            Ty::Int(int_ty) => Some(int_ty.size as u32),
            Ty::Bool(Bool { size }) => Some(*size as u32),
            Ty::Float(Float { size }) => Some(*size as u32),
            Ty::Enum(enum_ty) => Some(enum_ty.base_type.size as u32),
            Ty::Struct(struct_ty) => Some(struct_ty.size),
            Ty::Unknown(unk_ty) => unk_ty.size,
            // TODO architecture dependent!
            Ty::Ptr(_) => Some(8),
            // TODO does this even make sense?
            Ty::Subroutine(_) => Some(8),
            Ty::Void => Some(0),
            Ty::Alias(ref_tyid) => self.bytes_size(*ref_tyid),
            Ty::Array(Array {
                element_tyid,
                index_subrange,
            }) => {
                let element_size = self.bytes_size(*element_tyid)?;
                let element_count: u32 = index_subrange.count()?.try_into().ok()?;
                Some(element_size * element_count)
            }
        }
    }

    #[allow(dead_code)]
    pub fn select(
        &self,
        tyid: TypeID,
        byte_range: Range<u32>,
    ) -> std::result::Result<Selection, SelectError> {
        self.select_from(tyid, byte_range, SmallVec::new())
    }

    #[allow(dead_code)]
    fn select_from(
        &self,
        tyid: TypeID,
        byte_range: Range<u32>,
        mut path: SelectionPath,
    ) -> std::result::Result<Selection, SelectError> {
        assert!(!byte_range.is_empty());
        let ty = &self.get(tyid).unwrap();

        let size = self.bytes_size(tyid).unwrap_or(0);

        if byte_range.end > size {
            return Err(SelectError::InvalidRange);
        }

        if byte_range == (0..size) {
            return Ok(Selection { tyid, path });
        }

        match ty {
            Ty::Void
            | Ty::Ptr(_)
            | Ty::Int(_)
            | Ty::Enum(_)
            | Ty::Unknown(_)
            | Ty::Subroutine(_)
            | Ty::Float(_)
            | Ty::Bool(_) => Err(SelectError::InvalidRange),
            Ty::Alias(ref_tyid) => self.select(*ref_tyid, byte_range),
            Ty::Struct(struct_ty) => {
                let member = struct_ty.members.iter().find(|m| {
                    // TODO avoid the hashmap lookup?
                    let Some(memb_size) = self.bytes_size(m.tyid) else {
                        // struct member has unknown size; just ignore it in the search
                        return false;
                    };
                    (m.offset..m.offset + memb_size) == byte_range
                });

                if let Some(member) = member {
                    path.push(SelectStep::Member(Arc::clone(&member.name)));
                    return self.select_from(member.tyid, byte_range, path);
                } else {
                    return Err(SelectError::RangeCrossesBoundaries);
                }
            }
            Ty::Array(array_ty) => {
                let element_size = self
                    .bytes_size(array_ty.element_tyid)
                    .ok_or(SelectError::InvalidType)?;

                if byte_range.start % element_size == 0
                    && byte_range.end - byte_range.start == element_size
                {
                    let ndx = byte_range.start / element_size;
                    path.push(SelectStep::Index(ndx));
                    return self.select_from(array_ty.element_tyid, byte_range, path);
                } else {
                    return Err(SelectError::RangeCrossesBoundaries);
                }
            }
        }
    }

    pub fn set_known_object(&mut self, addr: Addr, tyid: TypeID) {
        event!(Level::TRACE, addr, ?tyid, "discovered call");
        self.known_objects.insert(addr, tyid);
    }

    pub fn get_known_object(&self, addr: Addr) -> Option<TypeID> {
        self.known_objects.get(&addr).copied()
    }

    pub fn alignment(&self, tyid: TypeID) -> Option<u8> {
        let ty = self.get(tyid)?;
        match ty {
            Ty::Int(int_ty) => Some(int_ty.alignment()),
            Ty::Enum(enum_ty) => Some(enum_ty.base_type.alignment()),
            Ty::Ptr(_) => Some(8),
            Ty::Float(float_ty) => Some(float_ty.alignment()),
            Ty::Alias(ref_tyid) => self.alignment(*ref_tyid),

            Ty::Void | Ty::Bool(_) | Ty::Subroutine(_) | Ty::Unknown(_) => None,

            Ty::Struct(struct_ty) => {
                // TODO any further check necessary?
                let mut align = 1;
                for memb in &struct_ty.members {
                    let memb_align = self.alignment(memb.tyid)?;
                    align = align.max(memb_align);
                }
                Some(align)
            }
            Ty::Array(array_ty) => {
                // TODO implement the full version of the rule for x86_64:
                // > An array uses the same alignment as its elements, except
                // > that a local or global array variable of length at least
                // > 16 bytes or a C99 variable-length array variable always has
                // > alignment of at least 16 bytes.6
                self.alignment(array_ty.element_tyid)
            }
        }
    }

    #[allow(dead_code)]
    pub fn dump<W: PP + ?Sized>(&self, out: &mut W) -> std::io::Result<()> {
        writeln!(out, "TypeSet ({} types) = {{", self.types.len())?;

        // ensure that the iteration always happens in the same order
        let tyids = {
            let mut keys: Vec<_> = self.types.keys().collect();
            keys.sort();
            keys
        };

        for &tyid in tyids {
            let ty = self.get(tyid).unwrap();
            write!(out, "  <{:?}> = ", tyid)?;
            out.open_box();
            self.dump_type(out, tyid, ty)?;
            out.close_box();
            writeln!(out)?;
        }
        writeln!(out, "}}")
    }

    pub fn dump_type_ref<W: PP + ?Sized>(&self, out: &mut W, tyid: TypeID) -> std::io::Result<()> {
        let typ = self.get(tyid).unwrap();
        match self.name(tyid) {
            Some(name) => write!(out, "{} <{:?}>", name, tyid),
            None => self.dump_type(out, tyid, typ),
        }
    }

    pub fn dump_type<W: PP + ?Sized>(
        &self,
        out: &mut W,
        tyid: TypeID,
        ty: &Ty,
    ) -> std::io::Result<()> {
        if let Some(name) = self.name(tyid) {
            write!(out, "\"{}\" ", name)?;
        }
        self.dump_ty(out, ty)
    }

    pub fn dump_ty<W: PP + ?Sized>(&self, out: &mut W, ty: &Ty) -> std::io::Result<()> {
        match ty {
            Ty::Int(Int { size, signed }) => {
                let prefix = match signed {
                    Signedness::Signed => "i",
                    Signedness::Unsigned => "u",
                };
                write!(out, "{}{}", prefix, size * 8)?;
            }
            Ty::Bool(Bool { size }) => write!(out, "bool{}", *size * 8)?,
            Ty::Float(Float { size }) => write!(out, "float{}", *size * 8)?,
            Ty::Enum(_) => write!(out, "enum")?,
            Ty::Alias(ref_tyid) => self.dump_type_ref(out, *ref_tyid)?,
            Ty::Struct(struct_ty) => {
                write!(out, "struct {{\n    ")?;
                out.open_box();
                for (ndx, memb) in struct_ty.members.iter().enumerate() {
                    if ndx > 0 {
                        writeln!(out)?;
                    }
                    write!(out, "@{:3} {} ", memb.offset, memb.name)?;
                    self.dump_type_ref(out, memb.tyid)?;
                }
                out.close_box();
                write!(out, "\n}}")?;
            }
            Ty::Array(array_ty) => {
                write!(out, "[")?;
                let Subrange { lo, hi } = array_ty.index_subrange;
                match (lo, hi) {
                    (0, None) => {}
                    (0, Some(hi)) => {
                        write!(out, "{hi}")?;
                    }
                    (lo, None) => {
                        write!(out, "{lo}..")?;
                    }
                    (lo, Some(hi)) => {
                        write!(out, "{lo}..{hi}")?;
                    }
                }
                write!(out, "]")?;
                self.dump_type_ref(out, array_ty.element_tyid)?;
            }
            Ty::Ptr(type_id) => {
                write!(out, "*")?;
                self.dump_type_ref(out, *type_id)?;
            }
            Ty::Subroutine(subr_ty) => {
                write!(out, "func (")?;
                out.open_box();

                for (ndx, (name, tyid)) in subr_ty
                    .param_names
                    .iter()
                    .zip(subr_ty.param_tyids.iter())
                    .enumerate()
                {
                    if ndx > 0 {
                        writeln!(out, ",")?;
                    }
                    let name = name.as_ref().map(|s| s.as_str()).unwrap_or("<unnamed>");
                    write!(out, "{} ", name)?;
                    self.dump_type_ref(out, *tyid)?;
                }

                out.close_box();
                write!(out, ") ")?;
                self.dump_type_ref(out, subr_ty.return_tyid)?;
            }
            Ty::Unknown(_) => write!(out, "?")?,
            Ty::Void => write!(out, "void")?,
        }

        Ok(())
    }

    pub(crate) fn call_site_by_return_pc(&self, return_pc: u64) -> Option<TypeID> {
        self.call_sites
            .get_by_return_pc(return_pc)
            .map(|call_site| call_site.tyid)
    }
    pub(crate) fn add_call_site_by_return_pc(&mut self, return_pc: u64, tyid: TypeID) {
        self.call_sites
            .add_by_return_pc(return_pc, CallSite { tyid })
    }

    pub fn resolve_call(&self, key: CallSiteKey) -> Option<TypeID> {
        let CallSiteKey { return_pc, target } = key;

        let tyid = self
            .call_site_by_return_pc(return_pc)
            .or_else(|| self.get_known_object(target));

        let typ = tyid.map(|tyid| self.get_through_alias(tyid).unwrap());
        event!(Level::TRACE, ?tyid, ?typ, "call resolution");

        tyid
    }
}

impl std::fmt::Debug for TypeSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut pp = crate::pp::PrettyPrinter::start(pp::FmtAsIoUTF8(f));
        self.dump(&mut pp).unwrap();
        Ok(())
    }
}

pub struct CallSiteKey {
    pub return_pc: u64,
    pub target: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ty {
    Int(Int),
    Bool(Bool),
    #[allow(dead_code)]
    Enum(Enum),
    Struct(Struct),
    Array(Array),
    Ptr(TypeID),
    Float(Float),
    Subroutine(Subroutine),
    Unknown(Unknown),
    Void,
    Alias(TypeID),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Int {
    pub size: u8,
    pub signed: Signedness,
}
impl Int {
    fn alignment(&self) -> u8 {
        self.size
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Signedness {
    Signed,
    Unsigned,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bool {
    size: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Float {
    pub size: u8,
}
impl Float {
    fn alignment(&self) -> u8 {
        self.size
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Enum {
    pub variants: Vec<EnumVariant>,
    pub base_type: Int,
}
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumVariant {
    pub value: i64,
    pub name: Arc<String>,
}

/// Representation fo a structure or union type.
///
/// Unions may be represented by adding overlapping members. All members'
/// occupied byte range must fit into the structure's size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Struct {
    pub members: Vec<StructMember>,
    pub size: u32,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructMember {
    pub offset: u32,
    pub name: Arc<String>,
    pub tyid: TypeID,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Array {
    pub element_tyid: TypeID,
    pub index_subrange: Subrange,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Subrange {
    pub lo: i64,
    pub hi: Option<i64>,
}
impl Subrange {
    fn count(&self) -> Option<i64> {
        let hi = self.hi?;
        Some(hi - self.lo + 1)
    }
}

/// Placeholder for types that are not fully understood by the system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unknown {
    /// Size in bytes.  `None`, for types of unknown size.
    pub size: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Subroutine {
    pub return_tyid: TypeID,
    pub param_names: Vec<Option<Arc<String>>>,
    pub param_tyids: Vec<TypeID>,
}

//
// Selection
//

#[derive(Debug, PartialEq, Eq)]
pub struct Selection {
    pub tyid: TypeID,
    pub path: SelectionPath,
}
pub type SelectionPath = SmallVec<[SelectStep; 4]>;
#[allow(dead_code)]
#[derive(Debug, PartialEq, Eq)]
pub enum SelectStep {
    Index(u32),
    Member(Arc<String>),
}
impl SelectStep {
    #[allow(dead_code)]
    fn as_str(&self) -> Option<&str> {
        match self {
            SelectStep::Index(_) => None,
            SelectStep::Member(name) => Some(name.as_str()),
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SelectError {
    #[error("the given range crosses the boundaries of any of the type's members")]
    RangeCrossesBoundaries,

    #[error("the given range is invalid for the type")]
    InvalidRange,

    #[error("the type to select into is internally invalid")]
    InvalidType,
}

#[allow(dead_code)]
fn dump_types<W: pp::PP>(
    out: &mut W,
    report: &dwarf::Report,
    types: &TypeSet,
) -> std::io::Result<()> {
    writeln!(out, "dwarf types --[[")?;
    types.dump(out).unwrap();

    writeln!(out)?;
    writeln!(out, "{} non-fatal errors:", report.errors.len())?;
    for (ofs, err) in &report.errors {
        writeln!(out, "offset 0x{:8x}: {}", ofs, err)?;
    }
    writeln!(out, "]]--")?;
    Ok(())
}

pub struct CallSites {
    by_return_pc: HashMap<u64, CallSite>,
}

pub struct CallSite {
    pub tyid: TypeID,
}

impl CallSites {
    pub fn new() -> Self {
        CallSites {
            by_return_pc: HashMap::new(),
        }
    }

    pub fn add_by_return_pc(&mut self, return_pc: u64, callsite: CallSite) {
        self.by_return_pc.insert(return_pc, callsite);
    }

    pub fn get_by_return_pc(&self, return_pc: u64) -> Option<&CallSite> {
        self.by_return_pc.get(&return_pc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use include_dir::{include_dir, Dir};

    pub(crate) static DATA_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/test-data/ty/");

    #[test]
    fn shared_void() {
        let mut types = TypeSet::new();

        let tyid_void_1 = types.tyid_shared_void();
        let tyid_void_2 = types.tyid_shared_void();
        assert_eq!(tyid_void_1, tyid_void_2);
        assert_eq!(types.get(tyid_void_1), Some(&Ty::Void));

        types.set_name(tyid_void_1, "whatever".to_string());
        assert_eq!(types.name(tyid_void_1), Some("whatever"));
    }

    #[test]
    fn shared_unknown_unsized() {
        let mut types = TypeSet::new();

        let tyid_unk_1 = types.tyid_shared_unknown_unsized();
        let tyid_unk_2 = types.tyid_shared_unknown_unsized();
        assert_eq!(tyid_unk_1, tyid_unk_2);
        assert_eq!(
            types.get(tyid_unk_1),
            Some(&Ty::Unknown(Unknown { size: None }))
        );

        types.set_name(tyid_unk_1, "whatever".to_string());
        assert_eq!(types.name(tyid_unk_1), Some("whatever"));
    }

    #[test]
    fn shared_unknown_sized() {
        const SIZES: [u32; 6] = [0, 44, 120423, 21, 2, 4];

        let mut types = TypeSet::new();

        let tyids = SIZES.map(|size| types.tyid_shared_unknown_of_size(size));
        let tyids_check = SIZES.map(|size| types.tyid_shared_unknown_of_size(size));
        assert_eq!(tyids_check, tyids);

        for (size, tyid) in SIZES.iter().zip(tyids) {
            let ty = types.get(tyid).unwrap();
            assert_eq!(ty, &Ty::Unknown(Unknown { size: Some(*size) }));

            types.set_name(tyid, "whatever".to_string());
            assert_eq!(types.name(tyid), Some("whatever"));
        }
    }
    #[test]
    fn simple_struct() {
        let mut types = TypeSet::new();

        let mut next_ty_id: RegularTypeID = 0;
        let mut gen_tyid = || {
            let id = next_ty_id;
            next_ty_id += 1;
            TypeID::Regular(id)
        };

        let tyid_i32 = gen_tyid();
        let ty_i32 = Ty::Int(Int {
            size: 4,
            signed: Signedness::Signed,
        });
        types.set(tyid_i32, ty_i32).unwrap();
        types.set_name(tyid_i32, "i32".to_owned());

        let tyid_i8 = gen_tyid();
        let ty_i8 = Ty::Int(Int {
            size: 1,
            signed: Signedness::Signed,
        });
        types.set(tyid_i8, ty_i8).unwrap();
        types.set_name(tyid_i8, "i8".to_owned());

        let tyid_point = gen_tyid();
        let ty_point = Ty::Struct(Struct {
            size: 24,
            members: vec![
                StructMember {
                    offset: 0,
                    name: Arc::new("x".to_owned()),
                    tyid: tyid_i32,
                },
                // a bit of padding
                StructMember {
                    offset: 8,
                    name: Arc::new("y".to_owned()),
                    tyid: tyid_i32,
                },
                StructMember {
                    offset: 12,
                    name: Arc::new("cost".to_owned()),
                    tyid: tyid_i8,
                },
            ],
        });
        types.set(tyid_point, ty_point).unwrap();
        types.set_name(tyid_point, "point".to_owned());

        {
            let s = types.select(tyid_point, 0..4).unwrap();
            assert_eq!(s.tyid, tyid_i32);
            assert_eq!(s.path.len(), 1);
            assert_eq!(s.path[0].as_str(), Some("x"));
        }

        for byte_range in [(0..5), (1..5), (3..7), (8..16), (12..14)] {
            assert_eq!(
                types.select(tyid_point, byte_range),
                Err(SelectError::RangeCrossesBoundaries)
            );
        }

        {
            let s = types.select(tyid_point, 12..13).unwrap();
            assert_eq!(s.tyid, tyid_i8);
            assert_eq!(s.path.len(), 1);
            assert_eq!(s.path[0].as_str(), Some("cost"));
        }

        for byte_range in [(23..25), (24..25)] {
            assert_eq!(
                types.select(tyid_point, byte_range),
                Err(SelectError::InvalidRange)
            );
        }
    }
}
