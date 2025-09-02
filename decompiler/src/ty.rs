use std::{
    collections::{HashMap, HashSet},
    ops::Range,
    sync::Arc,
};

use slotmap::SlotMap;
use smallvec::{smallvec, SmallVec};
use thiserror::Error;
use tracing::{event, Level};

pub mod dwarf;

use crate::pp::{self, PP};
// important: TypeID is an *opaque* ID used by `ssa` to refer to complex data
// types represented and manipulated in this module, so we MUST use the same
// type here.
pub use crate::ssa::TypeID;

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
    types: SlotMap<TypeID, Ty>,
    name_of_tyid: slotmap::SecondaryMap<TypeID, String>,
    known_objects: HashMap<Addr, TypeID>,
    call_sites: CallSites,

    tyid_void: TypeID,
    tyid_unk_unsized: TypeID,
    tyid_unk_of_size: HashMap<u32, TypeID>,
    read_only: HashSet<TypeID>,
}

pub type Addr = u64;

impl TypeSet {
    pub fn new() -> Self {
        let mut types = SlotMap::with_key();
        let tyid_void = types.insert(Ty::Void);
        let tyid_unk_unsized = types.insert(Ty::Unknown(Unknown { size: None }));

        let mut read_only = HashSet::new();
        read_only.insert(tyid_void);
        read_only.insert(tyid_unk_unsized);

        let mut types = TypeSet {
            types,
            name_of_tyid: slotmap::SecondaryMap::new(),
            known_objects: HashMap::new(),
            call_sites: CallSites::new(),
            tyid_void,
            tyid_unk_unsized,
            tyid_unk_of_size: HashMap::new(),
            read_only,
        };

        types.set_name(tyid_void, "void".to_owned());
        types
    }

    pub fn tyid_shared_void(&self) -> TypeID {
        self.tyid_void
    }
    pub fn tyid_shared_unknown_unsized(&self) -> TypeID {
        self.tyid_unk_unsized
    }
    pub fn tyid_shared_unknown_of_size(&mut self, size: u32) -> TypeID {
        if let Some(tyid) = self.tyid_unk_of_size.get(&size) {
            debug_assert!(self.read_only.contains(&tyid));
            return *tyid;
        }

        let tyid = self.types.insert(Ty::Unknown(Unknown { size: Some(size) }));
        self.tyid_unk_of_size.insert(size, tyid);
        self.read_only.insert(tyid);
        tyid
    }

    /// Get the name of a type, if it has one.
    ///
    /// Invalid TypeIDs result in None.
    pub fn name(&self, tyid: TypeID) -> Option<&str> {
        self.name_of_tyid.get(tyid).map(|s| s.as_str())
    }

    /// Set the name of a type.
    pub fn set_name(&mut self, tyid: TypeID, name: String) -> Result<()> {
        if self.read_only.contains(&tyid) {
            return Err(Error::ReadOnly);
        }

        self.name_of_tyid.insert(tyid, name);
        Ok(())
    }

    /// Add a type to the set. Returns a TypeID corresponding to it.
    ///
    /// The return TypeID can "change meaning", i.e. refer to a different type,
    /// if `set` is called on it.
    pub fn add(&mut self, ty: Ty) -> TypeID {
        self.types.insert(ty)
    }

    /// Change the type referred to by a TypeID.
    ///
    /// Panics if the TypeID is invalid. Make sure to only pass TypeIDs that
    /// have been returned by this same TypeSet's `add` method in the past.
    pub fn set(&mut self, tyid: TypeID, ty: Ty) {
        *self.types.get_mut(tyid).unwrap() = ty;
    }

    pub fn get(&self, tyid: TypeID) -> Option<&Ty> {
        self.types.get(tyid)
    }

    pub fn get_through_alias(&self, mut tyid: TypeID) -> Option<&Ty> {
        loop {
            let ty = self.types.get(tyid)?;
            if let Ty::Alias(target_tyid) = ty {
                tyid = *target_tyid;
            } else {
                return Some(ty);
            }
        }
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
        }
    }

    pub fn assert_invariants(&self) {
        for (_, ty) in self.types.iter() {
            match &ty {
                Ty::Void
                | Ty::Subroutine(_)
                | Ty::Int(_)
                | Ty::Enum(_)
                | Ty::Ptr(_)
                | Ty::Unknown(_)
                | Ty::Bool(_)
                | Ty::Float(_)
                | Ty::Alias(_) => {}
                Ty::Struct(struct_ty) => {
                    assert!(struct_ty.members.iter().all(|m| {
                        let size = self.bytes_size(m.tyid).unwrap();
                        m.offset + size <= struct_ty.size
                    }));

                    // the code below makes sense if I never want to allow
                    // overlapping struct members. But it could be useful! this
                    // program is not beholden to the rules of programming languages

                    // let count = struct_ty.members.len();
                    // for i in 0..count {
                    //     for j in 0..count {
                    //         if i >= j {
                    //             continue;
                    //         }

                    //         let mi = &struct_ty.members[i];
                    //         let mi_end = mi.offset + self.get(mi.tyid).unwrap().ty.bytes_size();
                    //         let mj = &struct_ty.members[j];
                    //         let mj_end = mj.offset + self.get(mj.tyid).unwrap().ty.bytes_size();

                    //         if mj_end > mi.offset && mi_end > mj.offset {
                    //             panic!("struct '{}': members `{}` ({}:{}) and `{}` ({}:{}) overlap!");
                    //         }
                    //     }
                    // }
                }
            }
        }
    }

    #[allow(dead_code)]
    pub fn select(
        &self,
        tyid: TypeID,
        byte_range: Range<i64>,
    ) -> std::result::Result<Selection, SelectError> {
        assert!(!byte_range.is_empty());
        let ty = &self.get(tyid).unwrap();

        let size = self.bytes_size(tyid).unwrap_or(0) as i64;

        if byte_range.end > size {
            return Err(SelectError::InvalidRange);
        }

        if byte_range == (0..size) {
            return Ok(Selection {
                tyid,
                path: SmallVec::new(),
            });
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
                    let memb_size = self.bytes_size(m.tyid).unwrap() as i64;
                    let ofs = m.offset as i64;
                    (ofs..ofs + memb_size) == byte_range
                });

                if let Some(member) = member {
                    Ok(Selection {
                        tyid: member.tyid,
                        path: smallvec![SelectStep::Member(Arc::clone(&member.name))],
                    })
                } else {
                    Err(SelectError::RangeCrossesBoundaries)
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

        for tyid in tyids {
            let ty = self.types.get(tyid).unwrap();
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
    pub path: SmallVec<[SelectStep; 4]>,
}
#[allow(dead_code)]
#[derive(Debug, PartialEq, Eq)]
pub enum SelectStep {
    Index(i64),
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

        let res = types.set_name(tyid_void_1, "whatever".to_string());
        assert_eq!(res, Err(Error::ReadOnly));
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

        let res = types.set_name(tyid_unk_1, "whatever".to_string());
        assert_eq!(res, Err(Error::ReadOnly));
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

            let res = types.set_name(tyid, "whatever".to_string());
            assert_eq!(res, Err(Error::ReadOnly));
        }
    }

    #[test]
    fn simple_struct() {
        let mut types = TypeSet::new();

        let tyid_i32 = types.add(Ty::Int(Int {
            size: 4,
            signed: Signedness::Signed,
        }));
        types.set_name(tyid_i32, "i32".to_owned());

        let tyid_i8 = types.add(Ty::Int(Int {
            size: 1,
            signed: Signedness::Signed,
        }));
        types.set_name(tyid_i8, "i8".to_owned());

        let ty_point = types.add(Ty::Struct(Struct {
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
        }));
        types.set_name(ty_point, "point".to_owned());

        {
            let s = types.select(ty_point, 0..4).unwrap();
            assert_eq!(s.tyid, tyid_i32);
            assert_eq!(s.path.len(), 1);
            assert_eq!(s.path[0].as_str(), Some("x"));
        }

        for byte_range in [(0..5), (1..5), (3..7), (8..16), (12..14)] {
            assert_eq!(
                types.select(ty_point, byte_range),
                Err(SelectError::RangeCrossesBoundaries)
            );
        }

        {
            let s = types.select(ty_point, 12..13).unwrap();
            assert_eq!(s.tyid, tyid_i8);
            assert_eq!(s.path.len(), 1);
            assert_eq!(s.path[0].as_str(), Some("cost"));
        }

        for byte_range in [(23..25), (24..25)] {
            assert_eq!(
                types.select(ty_point, byte_range),
                Err(SelectError::InvalidRange)
            );
        }
    }

    #[test]
    #[should_panic]
    fn struct_size_out_of_bounds() {
        let mut types = TypeSet::new();

        let tyid_i32 = types.add(Ty::Int(Int {
            size: 4,
            signed: Signedness::Signed,
        }));
        types.set_name(tyid_i32, "i32".to_owned());

        let ty_point = types.add(Ty::Struct(Struct {
            size: 4,
            members: vec![StructMember {
                offset: 4,
                name: Arc::new("x".to_owned()),
                tyid: tyid_i32,
            }],
        }));
        types.set_name(ty_point, "point".to_owned());

        types.assert_invariants();
    }
}
