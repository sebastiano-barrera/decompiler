use std::{borrow::Cow, path::Path, sync::Arc};

use redb::{ReadableTable, ReadableTableMetadata};
use smallvec::SmallVec;
use thiserror::Error;
use tracing::{event, Level};

pub mod dwarf;

use crate::{
    mil,
    pp::{self, PP},
};

#[derive(
    Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct TypeID(pub u64);

impl std::fmt::Debug for TypeID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self == &TypeSet::TYID_SHARED_VOID {
            write!(f, "TID:void")
        } else {
            write!(f, "TID:{}", self.0)
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    #[error("change forbidden; the referred type is read-only")]
    ReadOnly,

    #[error("file does not exist")]
    FileNotFound,

    #[error("internal database error: {0}")]
    DatabaseError(String),
}

impl From<redb::DatabaseError> for Error {
    fn from(err: redb::DatabaseError) -> Self {
        match err {
            // this specific error situation is useful to single out
            redb::DatabaseError::Storage(redb::StorageError::Io(io_err))
                if io_err.kind() == std::io::ErrorKind::NotFound =>
            {
                Error::FileNotFound
            }
            _ => Error::DatabaseError(err.to_string()),
        }
    }
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
    db: redb::Database,
}

//
// redb tables
//
// values are JSON-serialized
//
const TABLE_TYPES: redb::TableDefinition<TypeID, Ty> = redb::TableDefinition::new("types");
const TABLE_NAME_OF_TYPE: redb::TableDefinition<TypeID, String> =
    redb::TableDefinition::new("name_of_type");
const TABLE_TYPE_OF_GLOBAL: redb::TableDefinition<Addr, TypeID> =
    redb::TableDefinition::new("type_of_global");
const TABLE_TYPE_OF_CALL: redb::TableDefinition<u64, TypeID> =
    redb::TableDefinition::new("type_of_call");

impl redb::Value for TypeID {
    type SelfType<'a> = Self;
    type AsBytes<'a> = <u64 as redb::Value>::AsBytes<'a>;

    fn fixed_width() -> Option<usize> {
        <u64 as redb::Value>::fixed_width()
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a,
    {
        let numeric = <u64 as redb::Value>::from_bytes(data);
        TypeID(numeric)
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a>
    where
        Self: 'b,
    {
        <u64 as redb::Value>::as_bytes(&value.0)
    }

    fn type_name() -> redb::TypeName {
        redb::TypeName::new("TypeID")
    }
}

impl redb::Key for TypeID {
    fn compare(data1: &[u8], data2: &[u8]) -> std::cmp::Ordering {
        <u64 as redb::Key>::compare(data1, data2)
    }
}

impl redb::Value for Ty {
    type SelfType<'a> = Self;
    type AsBytes<'a> = Vec<u8>;

    fn fixed_width() -> Option<usize> {
        None
    }

    fn type_name() -> redb::TypeName {
        redb::TypeName::new("Ty")
    }

    fn from_bytes<'a>(bytes: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a,
    {
        rmp_serde::from_slice(bytes).unwrap()
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a>
    where
        Self: 'b,
    {
        rmp_serde::to_vec(value).unwrap()
    }
}

pub type Addr = u64;

pub enum Location<'a> {
    File(&'a Path),
    Memory,
}

fn create_tables(db: &redb::Database) -> redb::Result<(), redb::Error> {
    let tx = db.begin_write()?;
    tx.open_table(TABLE_TYPES)?;
    tx.open_table(TABLE_NAME_OF_TYPE)?;
    tx.open_table(TABLE_TYPE_OF_GLOBAL)?;
    tx.open_table(TABLE_TYPE_OF_CALL)?;
    tx.commit()?;
    Ok(())
}

impl TypeSet {
    const TYID_SHARED_VOID: TypeID = TypeID(u64::MAX);

    pub fn new() -> Self {
        Self::open(Location::Memory).unwrap()
    }

    pub fn open(location: Location) -> Result<Self> {
        let db_res = match location {
            Location::File(path) => redb::Database::create(path),
            Location::Memory => redb::Database::builder()
                .create_with_backend(redb::backends::InMemoryBackend::new()),
        };
        let db = db_res?;

        create_tables(&db).map_err(|e| Error::DatabaseError(e.to_string()))?;

        let mut type_set = TypeSet { db };
        type_set.set(Self::TYID_SHARED_VOID, Ty::Void);
        type_set.set_name(Self::TYID_SHARED_VOID, "void".to_string());
        Ok(type_set)
    }

    pub fn tyid_shared_void(&self) -> TypeID {
        Self::TYID_SHARED_VOID
    }

    // TODO Handle database errors, expose them as Result<_>

    /// Get the name of a type, if it has one.
    ///
    /// Invalid TypeIDs result in None.
    pub fn name(&self, tyid: TypeID) -> Option<String> {
        let tx = self.db.begin_read().expect("begin_read");
        let table = tx.open_table(TABLE_NAME_OF_TYPE).expect("open_table");
        let value = table.get(&tyid).expect("get");
        value.map(|v| v.value().to_string())
    }
    /// Set the name of a type.
    ///
    /// Has no effect on non-stored types (see [TypeID]).
    pub fn set_name(&mut self, tyid: TypeID, name: String) {
        let tx = self.db.begin_write().expect("begin_write");
        {
            let mut table = tx.open_table(TABLE_NAME_OF_TYPE).expect("open_table");
            table
                .insert(&tyid, &name)
                .expect("insert into name_of_type");
        }
        tx.commit().expect("commit");
    }
    /// Removes a custom name on a type, as previously set by [`Self:set_name`]
    ///
    /// Has no effect on non-stored types (see [TypeID]).
    pub fn unset_name(&mut self, tyid: TypeID) {
        let tx = self.db.begin_write().expect("begin_write");
        {
            let mut table = tx.open_table(TABLE_NAME_OF_TYPE).expect("open_table");
            table.remove(&tyid).expect("remove");
        }
        tx.commit().expect("commit");
    }

    pub fn types_count(&self) -> usize {
        let tx = self.db.begin_read().expect("begin_read");
        let table = tx.open_table(TABLE_TYPES).expect("open_table");
        table.len().expect("table len") as usize
    }

    /// Add a type to the database using the given TypeID, or change the type it refers to.
    ///
    /// Any name assigned to the type is preserved.
    ///
    /// Non-regular TypeIDs are read-only, so they cannot be redefined (this
    /// function will return an Error::ReadOnly).
    pub fn set(&mut self, tyid: TypeID, ty: Ty) {
        if tyid == Self::TYID_SHARED_VOID {
            // read-only type
            return;
        }

        let tx = self.db.begin_write().expect("begin_write");
        {
            let mut table = tx.open_table(TABLE_TYPES).expect("open_table");
            table.insert(&tyid, &ty).expect("insert into types");
        }
        tx.commit().expect("commit");
    }

    pub fn get(&self, tyid: TypeID) -> Option<Cow<'_, Ty>> {
        let tx = self.db.begin_read().expect("begin_read");
        let table = tx.open_table(TABLE_TYPES).expect("open_table");
        let value = table.get(&tyid).expect("get");
        value.map(|v| Cow::Owned(v.value().clone()))
    }

    pub fn get_through_alias(&self, mut tyid: TypeID) -> Option<Cow<'_, Ty>> {
        loop {
            let ty = self.get(tyid)?;
            if let Ty::Alias(target_tyid) = &*ty {
                tyid = *target_tyid;
            } else {
                return Some(ty);
            }
        }
    }

    pub fn get_or_create(&mut self, tyid: TypeID, create_fn: impl FnOnce() -> Ty) -> Cow<'_, Ty> {
        if self.get(tyid).is_none() {
            let ty = create_fn();
            self.set(tyid, ty);
        }

        self.get(tyid).unwrap()
    }

    pub fn bytes_size(&self, tyid: TypeID) -> Option<usize> {
        let ty = self.get(tyid)?;
        match ty.as_ref() {
            Ty::Int(int_ty) => Some(int_ty.size as usize),
            Ty::Bool(Bool { size }) => Some(*size as usize),
            Ty::Float(Float { size }) => Some(*size as usize),
            Ty::Enum(enum_ty) => Some(enum_ty.base_type.size as usize),
            Ty::Struct(struct_ty) => Some(struct_ty.size),
            // TODO architecture dependent!
            Ty::Ptr(_) => Some(8),
            // TODO does this even make sense?
            Ty::Subroutine(_) => Some(8),
            Ty::Void => Some(0),
            Ty::Flag => None,
            Ty::Unknown => None,
            Ty::Alias(ref_tyid) => self.bytes_size(*ref_tyid),
            Ty::Array(Array {
                element_tyid,
                index_subrange,
            }) => {
                let element_size = self.bytes_size(*element_tyid)?;
                let element_count: usize = index_subrange.count()?.try_into().ok()?;
                Some(element_size * element_count)
            }
        }
    }

    pub fn select(
        &self,
        tyid: TypeID,
        byte_range: ByteRange,
    ) -> std::result::Result<Selection, SelectError> {
        self.select_from(tyid, byte_range, SmallVec::new())
    }

    fn select_from(
        &self,
        tyid: TypeID,
        byte_range: ByteRange,
        mut path: SelectionPath,
    ) -> std::result::Result<Selection, SelectError> {
        if byte_range.is_empty() {
            return Err(SelectError::InvalidRange);
        }

        let ty = self.get(tyid).ok_or(SelectError::InvalidType)?;
        let size = self.bytes_size(tyid).unwrap_or(0) as usize;

        if byte_range.end > size {
            return Err(SelectError::InvalidRange);
        }

        if byte_range == ByteRange::new(0, size) {
            return Ok(Selection { tyid, path });
        }

        match ty.as_ref() {
            Ty::Void
            | Ty::Flag
            | Ty::Ptr(_)
            | Ty::Int(_)
            | Ty::Enum(_)
            | Ty::Subroutine(_)
            | Ty::Float(_)
            | Ty::Unknown
            | Ty::Bool(_) => Err(SelectError::InvalidRange),
            Ty::Alias(ref_tyid) => self.select(*ref_tyid, byte_range),
            Ty::Struct(struct_ty) => {
                let member = struct_ty.members.iter().find(|m| {
                    let Some(memb_size) = self.bytes_size(m.tyid) else {
                        // struct member has unknown size; just ignore it in the search
                        return false;
                    };

                    let offset = m.offset as usize;
                    let memb_size = memb_size as usize;
                    ByteRange::new(offset, offset + memb_size).contains(&byte_range)
                });

                if let Some(member) = member {
                    let member_size = self.bytes_size(member.tyid).unwrap().try_into().unwrap();
                    path.push(SelectStep::Member {
                        name: Arc::clone(&member.name),
                        size: member_size,
                    });
                    let memb_range = byte_range.shift_left(member.offset);
                    return self.select_from(member.tyid, memb_range, path);
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
                    let index = byte_range.start / element_size;
                    let element_range = byte_range.shift_left(element_size * index);
                    path.push(SelectStep::Index {
                        index,
                        element_size,
                    });
                    return self.select_from(array_ty.element_tyid, element_range, path);
                } else {
                    return Err(SelectError::RangeCrossesBoundaries);
                }
            }
        }
    }

    pub fn set_known_object(&mut self, addr: Addr, tyid: TypeID) {
        event!(Level::TRACE, addr, ?tyid, "discovered call");
        let tx = self.db.begin_write().expect("begin_write");
        {
            let mut table = tx.open_table(TABLE_TYPE_OF_GLOBAL).expect("open_table");
            table
                .insert(&addr, &tyid)
                .expect("insert into type_of_global");
        }
        tx.commit().expect("commit");
    }

    pub fn get_known_object(&self, addr: Addr) -> Option<TypeID> {
        let tx = self.db.begin_read().expect("begin_read");
        let table = tx.open_table(TABLE_TYPE_OF_GLOBAL).expect("open_table");
        let value = table.get(&addr).expect("get");
        value.map(|v| v.value())
    }

    pub fn alignment(&self, tyid: TypeID) -> Option<u8> {
        let ty = self.get(tyid)?;
        match ty.as_ref() {
            Ty::Int(int_ty) => Some(int_ty.alignment()),
            Ty::Enum(enum_ty) => Some(enum_ty.base_type.alignment()),
            Ty::Ptr(_) => Some(8),
            Ty::Float(float_ty) => Some(float_ty.alignment()),
            Ty::Alias(ref_tyid) => self.alignment(*ref_tyid),

            Ty::Unknown | Ty::Void | Ty::Bool(_) | Ty::Subroutine(_) | Ty::Flag => None,

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
        let tx = self.db.begin_read().expect("begin_read");
        let table = tx.open_table(TABLE_TYPES).expect("open_table");

        let count = table.len().expect("table len");
        writeln!(out, "TypeSet ({} types) = {{", count)?;

        // ensure that the iteration always happens in the same order
        let tyids = {
            let mut keys: Vec<_> = table
                .iter()
                .expect("table iter")
                .map(|item| item.unwrap().0.value())
                .collect();
            keys.sort();
            keys
        };

        for tyid in tyids {
            let ty = self.get(tyid).unwrap();
            write!(out, "  <{:?}> = ", tyid)?;
            out.open_box();
            self.dump_type(out, tyid, &*ty)?;
            out.close_box();
            writeln!(out)?;
        }
        writeln!(out, "}}")
    }

    pub fn dump_type_ref<W: PP + ?Sized>(&self, out: &mut W, tyid: TypeID) -> std::io::Result<()> {
        let typ = self.get(tyid).unwrap();
        match self.name(tyid) {
            Some(name) => write!(out, "{} <{:?}>", name, tyid),
            None => self.dump_type(out, tyid, &*typ),
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
            Ty::Flag => {
                write!(out, "flag")?;
            }
            Ty::Unknown => {
                write!(out, "<?>")?;
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
            Ty::Void => write!(out, "void")?,
        }

        Ok(())
    }

    pub(crate) fn call_site_by_return_pc(&self, return_pc: u64) -> Option<TypeID> {
        let tx = self.db.begin_read().expect("begin_read");
        let table = tx.open_table(TABLE_TYPE_OF_CALL).expect("open_table");
        let value = table.get(&return_pc).expect("get");
        value.map(|v| v.value())
    }
    pub(crate) fn add_call_site_by_return_pc(&mut self, return_pc: u64, tyid: TypeID) {
        let tx = self.db.begin_write().expect("begin_write");
        {
            let mut table = tx.open_table(TABLE_TYPE_OF_CALL).expect("open_table");
            table
                .insert(&return_pc, &tyid)
                .expect("insert into type_of_call");
        }
        tx.commit().expect("commit");
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

/// A range of bytes.
///
/// Similar to `std::ops::Range<usize>`, but:
/// - this is not an Iterator
/// - there is no custom syntax to construct an instance
/// - once created, the instance is not mutable (via the public API)
/// - there are useful methods, e.g. to shift the interval left
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ByteRange {
    start: usize,
    end: usize,
}
impl ByteRange {
    pub fn new(start: usize, end: usize) -> Self {
        assert!(end >= start);
        ByteRange { start, end }
    }

    pub fn lo(&self) -> usize {
        self.start
    }
    pub fn hi(&self) -> usize {
        self.end
    }
    pub fn len(&self) -> usize {
        self.end - self.start
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn shift(&self, offset: isize) -> ByteRange {
        ByteRange {
            start: (self.start as isize + offset) as usize,
            end: (self.end as isize + offset) as usize,
        }
    }
    pub fn shift_left(&self, offset: usize) -> ByteRange {
        let offset: isize = offset.try_into().unwrap();
        self.shift(-offset)
    }

    fn contains(&self, other: &ByteRange) -> bool {
        self.start <= other.start && self.end >= other.end
    }
}

pub struct CallSiteKey {
    pub return_pc: u64,
    pub target: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Ty {
    Flag,
    Int(Int),
    Bool(Bool),
    #[allow(dead_code)]
    Enum(Enum),
    Struct(Struct),
    Array(Array),
    Ptr(TypeID),
    Float(Float),
    Subroutine(Subroutine),
    Unknown,
    Void,
    Alias(TypeID),
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Int {
    pub size: u8,
    pub signed: Signedness,
}
impl Int {
    fn alignment(&self) -> u8 {
        self.size
    }
}
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Signedness {
    Signed,
    Unsigned,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Bool {
    size: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Float {
    pub size: u8,
}
impl Float {
    fn alignment(&self) -> u8 {
        self.size
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Enum {
    pub variants: Vec<EnumVariant>,
    pub base_type: Int,
}
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EnumVariant {
    pub value: i64,
    pub name: Arc<String>,
}

/// Representation fo a structure or union type.
///
/// Unions may be represented by adding overlapping members. All members'
/// occupied byte range must fit into the structure's size.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Struct {
    pub members: Vec<StructMember>,
    pub size: usize,
}
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StructMember {
    pub offset: usize,
    pub name: Arc<String>,
    pub tyid: TypeID,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Array {
    pub element_tyid: TypeID,
    pub index_subrange: Subrange,
}
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
    Index { index: usize, element_size: usize },
    Member { name: Arc<String>, size: usize },
    RawBytes { byte_range: ByteRange },
}

impl SelectStep {
    pub fn to_insn(&self, src: mil::Reg) -> mil::Insn {
        match self {
            SelectStep::Index {
                index,
                element_size,
            } => mil::Insn::ArrayGetElement {
                array: src,
                index: (*index).try_into().unwrap(),
                size: (*element_size).try_into().unwrap(),
            },
            SelectStep::Member { name, size } => {
                mil::Insn::StructGetMember {
                    struct_value: src,
                    // TODO figure out memory managment for this
                    name: name.to_string().leak(),
                    size: (*size).try_into().unwrap(),
                }
            }
            SelectStep::RawBytes { byte_range } => mil::Insn::Part {
                src,
                offset: byte_range.lo().try_into().unwrap(),
                size: byte_range.len().try_into().unwrap(),
            },
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
        assert_eq!(types.get(tyid_void_1).as_deref(), Some(&Ty::Void));

        // the name is not editable
        types.set_name(tyid_void_1, "whatever".to_string());
        let name_check = types.name(tyid_void_1);
        assert_eq!(name_check.as_ref().map(|s| s.as_str()), Some("whatever"));
    }

    #[test]
    fn simple_struct() {
        let mut types = TypeSet::new();

        let mut next_ty_id = 0;
        let mut gen_tyid = || {
            let id = next_ty_id;
            next_ty_id += 1;
            TypeID(id)
        };

        let tyid_i32 = gen_tyid();
        let ty_i32 = Ty::Int(Int {
            size: 4,
            signed: Signedness::Signed,
        });
        types.set(tyid_i32, ty_i32);
        types.set_name(tyid_i32, "i32".to_owned());

        let tyid_i8 = gen_tyid();
        let ty_i8 = Ty::Int(Int {
            size: 1,
            signed: Signedness::Signed,
        });
        types.set(tyid_i8, ty_i8);
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
        types.set(tyid_point, ty_point);
        types.set_name(tyid_point, "point".to_owned());

        {
            let s = types.select(tyid_point, ByteRange::new(0, 4)).unwrap();
            assert_eq!(s.tyid, tyid_i32);
            assert_eq!(s.path.len(), 1);
            let SelectStep::Member {
                name: member_name,
                size: _,
            } = &s.path[0]
            else {
                panic!()
            };
            assert_eq!(member_name.as_str(), "x");
        }

        for byte_range in [
            (ByteRange::new(0, 5)),
            (ByteRange::new(1, 5)),
            (ByteRange::new(3, 7)),
            (ByteRange::new(8, 16)),
            (ByteRange::new(12, 14)),
        ] {
            assert_eq!(
                types.select(tyid_point, byte_range),
                Err(SelectError::RangeCrossesBoundaries)
            );
        }

        {
            let s = types.select(tyid_point, ByteRange::new(12, 13)).unwrap();
            assert_eq!(s.tyid, tyid_i8);
            assert_eq!(s.path.len(), 1);
            let SelectStep::Member {
                name: member_name,
                size: _,
            } = &s.path[0]
            else {
                panic!()
            };
            assert_eq!(member_name.as_str(), "cost");
        }

        for byte_range in [(ByteRange::new(23, 25)), (ByteRange::new(24, 25))] {
            assert_eq!(
                types.select(tyid_point, byte_range),
                Err(SelectError::InvalidRange)
            );
        }
    }
}
