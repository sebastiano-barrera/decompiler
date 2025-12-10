use std::{borrow::Cow, ops::Deref, path::Path, sync::Arc};

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

    #[error("I/O error: {0}")]
    Io(String),

    #[error("type selection error: {0}")]
    SelectError(SelectError),
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Error::Io(value.to_string())
    }
}
impl From<heed::Error> for Error {
    fn from(err: heed::Error) -> Self {
        match err {
            // this specific error situation is useful to single out
            heed::Error::Io(io_err) if io_err.kind() == std::io::ErrorKind::NotFound => {
                Error::FileNotFound
            }
            _ => Error::DatabaseError(err.to_string()),
        }
    }
}
impl From<SelectError> for Error {
    fn from(err: SelectError) -> Self {
        Error::SelectError(err)
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
    db_types: heed::Database<TypeID, heed_types::SerdeRmp<Ty>>,
    db_name_of_type: heed::Database<TypeID, heed_types::Str>,
    db_type_of_global: heed::Database<U64LE, TypeID>,
    db_type_of_call: heed::Database<U64LE, TypeID>,

    // only used to remove the tempfile (for "memory" databases) when TypeSet is
    // dropped
    _tempdir: Option<tempfile::TempDir>,
    env: heed::Env<heed::WithoutTls>,
}

type U64LE = heed_types::U64<byteorder::LittleEndian>;
pub type Addr = u64;

impl<'a> heed::BytesEncode<'a> for TypeID {
    type EItem = Self;

    fn bytes_encode(item: &Self::EItem) -> std::result::Result<Cow<'a, [u8]>, heed::BoxedError> {
        let bytes = item.0.to_le_bytes();
        Ok(Cow::Owned(bytes.to_vec()))
    }
}

impl<'a> heed::BytesDecode<'a> for TypeID {
    type DItem = Self;

    fn bytes_decode(slice: &'a [u8]) -> std::result::Result<Self::DItem, heed::BoxedError> {
        if slice.len() != 8 {
            let msg = format!("invalid TypeID: wrong length {} != 8", slice.len());
            return Err(Box::new(Error::DatabaseError(msg)));
        }
        let mut array = [0u8; 8];
        array.copy_from_slice(slice);
        Ok(TypeID(u64::from_le_bytes(array)))
    }
}

pub enum Location<'a> {
    Dir(&'a Path),
    Memory,
}

impl TypeSet {
    const TYID_SHARED_VOID: TypeID = TypeID(u64::MAX);

    pub fn new() -> Self {
        Self::open(Location::Memory).unwrap()
    }

    pub fn open(location: Location) -> Result<Self> {
        let (tempdir, path) = match location {
            Location::Dir(path) => (None, path.to_owned()),
            Location::Memory => {
                let tempdir = tempfile::TempDir::new().expect("creating temporary file");
                let path = tempdir.path().to_owned();
                (Some(tempdir), path)
            }
        };

        // path is a directory (that's how lmdb works)
        std::fs::create_dir_all(&path)?;

        let env = unsafe {
            heed::EnvOpenOptions::new()
                .read_txn_without_tls()
                .max_dbs(16)
                .map_size(16 * 1024 * 1024)
                .flags(heed::EnvFlags::WRITE_MAP)
                .open(path)
        }?;

        let mut wtx = env.write_txn()?;
        let db_types = env.create_database(&mut wtx, Some("types"))?;
        let db_name_of_type = env.create_database(&mut wtx, Some("name_of_type"))?;
        let db_type_of_global = env.create_database(&mut wtx, Some("type_of_global"))?;
        let db_type_of_call = env.create_database(&mut wtx, Some("type_of_call"))?;

        db_types.put(&mut wtx, &Self::TYID_SHARED_VOID, &Ty::Void)?;
        db_name_of_type.put(&mut wtx, &Self::TYID_SHARED_VOID, "void")?;
        wtx.commit()?;
        event!(Level::DEBUG, "lmdb databases created, tx committed");

        Ok(TypeSet {
            env,
            db_types,
            db_name_of_type,
            db_type_of_global,
            db_type_of_call,
            _tempdir: tempdir,
        })
    }

    pub fn tyid_shared_void(&self) -> TypeID {
        Self::TYID_SHARED_VOID
    }

    pub fn read_tx<'a>(&'a self) -> Result<ReadTx<'a>> {
        let tx = self.env.read_txn()?;
        Ok(ReadTx { ts: self, tx })
    }

    pub fn write_tx<'a>(&'a mut self) -> Result<WriteTx<'a>> {
        let wtx = self.env.write_txn()?;
        Ok(WriteTx { ts: self, tx: wtx })
    }
}

impl std::fmt::Debug for TypeSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut pp = crate::pp::PrettyPrinter::start(pp::FmtAsIoUTF8(f));
        let rtx = self
            .read_tx()
            .expect("creating read transaction for printing");
        rtx.read().dump(&mut pp).unwrap();
        Ok(())
    }
}

pub struct ReadTx<'a> {
    ts: &'a TypeSet,
    tx: heed::RoTxn<'a, heed::WithoutTls>,
}
impl<'a> ReadTx<'a> {
    #[inline(always)]
    pub fn read<'s>(&'s self) -> ReadTxRef<'s> {
        ReadTxRef {
            ts: self.ts,
            tx: &self.tx,
        }
    }
}

#[derive(Clone, Copy)]
pub struct ReadTxRef<'a> {
    ts: &'a TypeSet,
    tx: &'a heed::RoTxn<'a, heed::WithoutTls>,
}

impl<'a> ReadTxRef<'a> {
    pub fn tyid_shared_void(&self) -> TypeID {
        self.ts.tyid_shared_void()
    }

    pub fn types_count(&self) -> Result<u64> {
        self.ts.db_types.len(self.tx).map_err(Into::into)
    }

    /// Get the name of a type, if it has one.
    ///
    /// Invalid TypeIDs result in None.
    pub fn name(&self, tyid: TypeID) -> Result<Option<String>> {
        let name_opt = self.ts.db_name_of_type.get(self.tx, &tyid)?;
        Ok(name_opt.map(ToOwned::to_owned))
    }

    pub fn get(&self, tyid: TypeID) -> Result<Option<Cow<'a, Ty>>> {
        self.ts
            .db_types
            .get(self.tx, &tyid)
            .map(|opt| opt.map(Cow::Owned))
            .map_err(Into::into)
    }

    pub fn get_through_alias(&self, mut tyid: TypeID) -> Result<Option<Cow<'a, Ty>>> {
        loop {
            let ty_opt = self.get(tyid)?;
            let Some(ty) = ty_opt else {
                return Ok(None);
            };
            if let Ty::Alias(target_tyid) = &*ty {
                tyid = *target_tyid;
            } else {
                return Ok(Some(ty));
            }
        }
    }

    pub fn bytes_size(&self, tyid: TypeID) -> Result<Option<usize>> {
        let ty_opt = self.get(tyid)?;
        let Some(ty) = ty_opt else {
            return Ok(None);
        };
        match ty.as_ref() {
            Ty::Int(int_ty) => Ok(Some(int_ty.size as usize)),
            Ty::Bool(Bool { size }) => Ok(Some(*size as usize)),
            Ty::Float(Float { size }) => Ok(Some(*size as usize)),
            Ty::Enum(enum_ty) => Ok(Some(enum_ty.base_type.size as usize)),
            Ty::Struct(struct_ty) => Ok(Some(struct_ty.size)),
            // TODO architecture dependent!
            Ty::Ptr(_) => Ok(Some(8)),
            // TODO does this even make sense?
            Ty::Subroutine(_) => Ok(Some(8)),
            Ty::Void => Ok(Some(0)),
            Ty::Flag => Ok(None),
            Ty::Unknown => Ok(None),
            Ty::Alias(ref_tyid) => self.bytes_size(*ref_tyid),
            Ty::Array(Array {
                element_tyid,
                index_subrange,
            }) => {
                let element_size = self.bytes_size(*element_tyid)?;
                let element_size = element_size.ok_or(Error::DatabaseError(
                    "Array element type has unknown size".to_string(),
                ))?;
                let element_count: usize = index_subrange
                    .count()
                    .ok_or(Error::DatabaseError(
                        "Array subrange count is unknown".to_string(),
                    ))?
                    .try_into()
                    .ok()
                    .ok_or(Error::DatabaseError(
                        "Array element count conversion failed".to_string(),
                    ))?;
                Ok(Some(element_size * element_count))
            }
        }
    }

    pub fn select(&self, tyid: TypeID, byte_range: ByteRange) -> Result<Selection> {
        self.select_from(tyid, byte_range, SmallVec::new())
    }

    fn select_from(
        &self,

        tyid: TypeID,
        byte_range: ByteRange,
        mut path: SelectionPath,
    ) -> Result<Selection> {
        if byte_range.is_empty() {
            return Err(SelectError::InvalidRange.into());
        }

        let ty = self.get(tyid)?.ok_or(SelectError::InvalidType)?;
        let size = self
            .bytes_size(tyid)
            .map_err(|_| SelectError::InvalidType)?
            .unwrap_or(0);

        if byte_range.end > size {
            return Err(SelectError::InvalidRange.into());
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
            | Ty::Bool(_) => Err(SelectError::InvalidRange.into()),
            Ty::Alias(ref_tyid) => self.select(*ref_tyid, byte_range).map_err(Into::into),
            Ty::Struct(struct_ty) => {
                let member = struct_ty.members.iter().find(|m| {
                    let Ok(Some(memb_size)) = self.bytes_size(m.tyid) else {
                        // struct member has unknown size; just ignore it in the search
                        return false;
                    };

                    let offset = m.offset;
                    let memb_size = memb_size;
                    ByteRange::new(offset, offset + memb_size).contains(&byte_range)
                });

                if let Some(member) = member {
                    let member_size = self
                        .bytes_size(member.tyid)
                        .map_err(|_| SelectError::InvalidType)?
                        .unwrap()
                        .try_into()
                        .unwrap();
                    path.push(SelectStep::Member {
                        name: Arc::clone(&member.name),
                        size: member_size,
                    });
                    let memb_range = byte_range.shift_left(member.offset);
                    return self.select_from(member.tyid, memb_range, path);
                } else {
                    return Err(SelectError::RangeCrossesBoundaries.into());
                }
            }
            Ty::Array(array_ty) => {
                let element_size = self
                    .bytes_size(array_ty.element_tyid)?
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
                    return Err(SelectError::RangeCrossesBoundaries.into());
                }
            }
        }
    }

    pub fn get_known_object(&self, addr: Addr) -> Result<Option<TypeID>> {
        self.ts
            .db_type_of_global
            .get(self.tx, &addr)
            .map_err(Into::into)
    }

    pub fn alignment(&self, tyid: TypeID) -> Result<Option<u8>> {
        let ty_opt = self.get(tyid)?;
        let Some(ty) = ty_opt else {
            return Ok(None);
        };
        match ty.as_ref() {
            Ty::Int(int_ty) => Ok(Some(int_ty.alignment())),
            Ty::Enum(enum_ty) => Ok(Some(enum_ty.base_type.alignment())),
            Ty::Ptr(_) => Ok(Some(8)),
            Ty::Float(float_ty) => Ok(Some(float_ty.alignment())),
            Ty::Alias(ref_tyid) => self.alignment(*ref_tyid),

            Ty::Unknown | Ty::Void | Ty::Bool(_) | Ty::Subroutine(_) | Ty::Flag => Ok(None),

            Ty::Struct(struct_ty) => {
                // TODO any further check necessary?
                let mut align = 1;
                for memb in &struct_ty.members {
                    let memb_align_opt = self.alignment(memb.tyid)?;
                    let Some(memb_align) = memb_align_opt else {
                        return Ok(None); // If any member has unknown alignment, the struct's alignment is unknown
                    };
                    align = align.max(memb_align);
                }
                Ok(Some(align))
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
        let count = self
            .ts
            .db_types
            .len(self.tx)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        writeln!(out, "TypeSet ({} types) = {{", count)?;

        // ensure that the iteration always happens in the same order
        let tyids = {
            let mut keys: Vec<_> = self
                .ts
                .db_types
                .iter(self.tx)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
                .flatten()
                .map(|(tyid, _)| tyid)
                .collect();
            keys.sort();
            keys
        };

        for tyid in tyids {
            let ty_opt = self
                .get(tyid)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            let ty = ty_opt.ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("TypeID {:?} not found during dump", tyid),
                )
            })?;
            write!(out, "  <{:?}> = ", tyid)?;
            out.open_box();
            self.dump_type(out, tyid, &*ty)?;
            out.close_box();
            writeln!(out)?;
        }
        writeln!(out, "}}")
    }

    pub fn dump_type_ref<W: PP + ?Sized>(&self, out: &mut W, tyid: TypeID) -> std::io::Result<()> {
        let typ_opt = self
            .get(tyid)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        let typ = typ_opt.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("TypeID {:?} not found during dump_type_ref", tyid),
            )
        })?;

        let name_res = self
            .name(tyid)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        match name_res {
            Some(name) => write!(out, "{}", name),
            None => self.dump_type(out, tyid, &*typ),
        }
    }

    pub fn dump_type<W: PP + ?Sized>(
        &self,
        out: &mut W,
        tyid: TypeID,
        ty: &Ty,
    ) -> std::io::Result<()> {
        let name_res = self
            .name(tyid)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        if let Some(name) = name_res {
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

    pub(crate) fn call_site_by_return_pc(&self, return_pc: u64) -> Result<Option<TypeID>> {
        self.ts
            .db_type_of_call
            .get(self.tx, &return_pc)
            .map_err(Into::into)
    }

    pub fn resolve_call(&self, key: CallSiteKey) -> Result<Option<TypeID>> {
        let CallSiteKey { return_pc, target } = key;

        let tyid_from_call_site = self.call_site_by_return_pc(return_pc)?;
        let tyid_from_global = self.get_known_object(target)?;

        let tyid = tyid_from_call_site.or(tyid_from_global);

        let typ = if let Some(id) = tyid {
            self.get_through_alias(id)?
        } else {
            None
        };
        event!(Level::TRACE, ?tyid, ?typ, "call resolution");

        Ok(tyid)
    }
}

pub struct WriteTx<'a> {
    ts: &'a TypeSet,
    tx: heed::RwTxn<'a>,
}
impl<'a> WriteTx<'a> {
    #[inline(always)]
    pub fn read<'s>(&'s self) -> ReadTxRef<'s> {
        ReadTxRef {
            ts: self.ts,
            // an impl Deref converts &RwTxn -> &RoTxn
            tx: self.tx.deref(),
        }
    }

    #[inline(always)]
    pub fn write<'s>(&'s mut self) -> WriteTxRef<'s, 'a> {
        WriteTxRef {
            ts: self.ts,
            tx: &mut self.tx,
        }
    }

    pub fn commit(self) -> Result<()> {
        self.tx.commit().map_err(Into::into)
    }
}

pub struct WriteTxRef<'a, 'p> {
    ts: &'p TypeSet,
    tx: &'a mut heed::RwTxn<'p>,
}

impl WriteTxRef<'_, '_> {
    #[inline(always)]
    pub fn read<'s>(&'s self) -> ReadTxRef<'s> {
        ReadTxRef {
            ts: self.ts,
            // an impl Deref converts &RwTxn -> &RoTxn
            tx: self.tx.deref(),
        }
    }

    /// Set the name of a type.
    ///
    /// Has no effect on non-stored types (see [TypeID]).
    pub fn set_name(&mut self, tyid: TypeID, name: String) -> Result<()> {
        self.ts
            .db_name_of_type
            .put(self.tx, &tyid, &name)
            .map_err(Into::into)
    }
    /// Add a type to the database using the given TypeID, or change the type it refers to.
    ///
    /// Any name assigned to the type is preserved.
    ///
    /// Non-regular TypeIDs are read-only, so they cannot be redefined (this
    /// function will return an Error::ReadOnly).
    pub fn set(&mut self, tyid: TypeID, ty: Ty) -> Result<()> {
        if tyid == TypeSet::TYID_SHARED_VOID {
            // read-only type
            return Err(Error::ReadOnly);
        }

        self.ts
            .db_types
            .put(self.tx, &tyid, &ty)
            .map_err(Into::into)
    }

    pub fn get_or_create<'a>(
        &'a mut self,
        tyid: TypeID,
        create_fn: impl FnOnce() -> Ty,
    ) -> Result<Cow<'a, Ty>> {
        if self.ts.db_types.get(self.tx, &tyid)?.is_none() {
            let ty = create_fn();
            self.set(tyid, ty)?;
        }

        Ok(Cow::Owned(self.ts.db_types.get(self.tx, &tyid)?.unwrap()))
    }

    pub fn set_known_object(&mut self, addr: Addr, tyid: TypeID) -> Result<()> {
        event!(Level::TRACE, addr, ?tyid, "discovered call");
        self.ts
            .db_type_of_global
            .put(self.tx, &addr, &tyid)
            .map_err(Into::into)
    }

    pub(crate) fn add_call_site_by_return_pc(
        &mut self,
        return_pc: u64,
        tyid: TypeID,
    ) -> Result<()> {
        self.ts
            .db_type_of_call
            .put(self.tx, &return_pc, &tyid)
            .map_err(Into::into)
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
    let rtx = types
        .read_tx()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    rtx.read().dump(out)?;

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

        let rtx = types.read_tx().unwrap();
        assert_eq!(
            rtx.read().get(tyid_void_1).unwrap().as_deref(),
            Some(&Ty::Void)
        );
        drop(rtx); // Drop read transaction before getting a write transaction

        // the name is not editable (this comment might be misleading, as set_name on void is not explicitly prevented)
        {
            let mut wtx = types.write_tx().unwrap();
            wtx.write()
                .set_name(tyid_void_1, "whatever".to_string())
                .unwrap();
            wtx.commit().unwrap(); // Commit the name change
        }

        let rtx = types.read_tx().unwrap();
        let name_check = rtx.read().name(tyid_void_1).unwrap();
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
        let mut wtx = types.write_tx().unwrap();
        wtx.write().set(tyid_i32, ty_i32).unwrap();
        wtx.write().set_name(tyid_i32, "i32".to_owned()).unwrap();

        let tyid_i8 = gen_tyid();
        let ty_i8 = Ty::Int(Int {
            size: 1,
            signed: Signedness::Signed,
        });
        wtx.write().set(tyid_i8, ty_i8).unwrap();
        wtx.write().set_name(tyid_i8, "i8".to_owned()).unwrap();

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
        wtx.write().set(tyid_point, ty_point).unwrap();
        wtx.write()
            .set_name(tyid_point, "point".to_owned())
            .unwrap();
        wtx.commit().unwrap(); // Commit all type creations

        let rtx = types.read_tx().unwrap();

        {
            let s = rtx.read().select(tyid_point, ByteRange::new(0, 4)).unwrap();
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
                rtx.read().select(tyid_point, byte_range).unwrap_err(),
                Error::SelectError(SelectError::RangeCrossesBoundaries)
            );
        }

        {
            let s = rtx
                .read()
                .select(tyid_point, ByteRange::new(12, 13))
                .unwrap();
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
                rtx.read().select(tyid_point, byte_range).unwrap_err(),
                Error::SelectError(SelectError::InvalidRange)
            );
        }
    }

    #[test]
    fn write_then_read_in_tx() {
        // just checking: reads "see" the effect of writes that happened in the
        // same write transaction
        let mut types = TypeSet::new();
        let mut wtx = types.write_tx().unwrap();
        let tyid = TypeID(123);
        wtx.write()
            .set(
                tyid,
                Ty::Int(Int {
                    size: 4,
                    signed: Signedness::Unsigned,
                }),
            )
            .unwrap();
        wtx.write().set_name(tyid, "my_u32".to_owned()).unwrap();

        let check = wtx.read().name(tyid).unwrap().unwrap();
        assert_eq!(check, "my_u32");
    }
}
