use std::{collections::HashMap, ops::Range, rc::Rc};

use smallvec::{smallvec, SmallVec};
use thiserror::Error;

// important: TypeID is an *opaque* ID used by `ssa` to refer to complex data
// types represented and manipulated in this module, so we MUST use the same
// type here.
pub use crate::ssa::TypeID;

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
    types: HashMap<TypeID, Type>,
    next_tyid: TypeID,
}

pub const TYID_I8: TypeID = TypeID(0);
pub const TYID_I32: TypeID = TypeID(1);
const TYID_FIRST_FREE: TypeID = TypeID(2);

impl TypeSet {
    pub fn new() -> Self {
        let mut types = HashMap::new();

        types.insert(
            TYID_I8,
            Type {
                name: Rc::new("i8".to_owned()),
                ty: Ty::Int(Int {
                    size: 1,
                    signed: Signedness::Signed,
                }),
            },
        );
        types.insert(
            TYID_I32,
            Type {
                name: Rc::new("i32".to_owned()),
                ty: Ty::Int(Int {
                    size: 4,
                    signed: Signedness::Signed,
                }),
            },
        );

        assert!(!types.contains_key(&TYID_FIRST_FREE));
        TypeSet {
            types,
            next_tyid: TYID_FIRST_FREE,
        }
    }

    pub fn add(&mut self, type_: Type) -> TypeID {
        self.assert_invariants(&type_);

        let tyid = self.next_tyid;
        self.next_tyid.0 += 1;
        self.types.insert(tyid, type_);
        tyid
    }

    pub fn get(&self, tyid: TypeID) -> Option<&Type> {
        self.types.get(&tyid)
    }

    pub fn bytes_size(&self, tyid: TypeID) -> Option<u32> {
        self.get(tyid).map(|t| t.ty.bytes_size())
    }

    pub fn assert_invariants(&self, type_: &Type) {
        match &type_.ty {
            Ty::Int(_) => {}
            Ty::Enum(_) => {}
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

    pub fn select(&self, tyid: TypeID, byte_range: Range<u32>) -> Result<Selection, SelectError> {
        let ty = &self.get(tyid).unwrap().ty;

        if byte_range.end > ty.bytes_size() {
            return Err(SelectError::InvalidRange);
        }

        match ty {
            Ty::Int(_) | Ty::Enum(_) => {
                if byte_range == (0..ty.bytes_size()) {
                    Ok(Selection {
                        tyid,
                        path: SmallVec::new(),
                    })
                } else {
                    Err(SelectError::InvalidRange)
                }
            }
            Ty::Struct(struct_ty) => {
                let member = struct_ty.members.iter().find(|m| {
                    // TODO avoid the hashmap lookup?
                    let bytes_size = self.get(m.tyid).unwrap().ty.bytes_size();
                    (m.offset..m.offset + bytes_size) == byte_range
                });

                if let Some(member) = member {
                    Ok(Selection {
                        tyid: member.tyid,
                        path: smallvec![SelectStep::Member(Rc::clone(&member.name))],
                    })
                } else {
                    Err(SelectError::RangeCrossesBoundaries)
                }
            }
        }
    }
}

// use cases:
//
//  - register types
//
//  - select(type, offset, size) => (type, access path) || invalid || cross-boundaries
//

pub struct Type {
    // TODO alignment, ...
    /// Human-readable name for the type
    pub name: Rc<String>,
    pub ty: Ty,
}

pub enum Ty {
    Int(Int),
    Enum(Enum),
    Struct(Struct),
}
impl Ty {
    fn bytes_size(&self) -> u32 {
        match self {
            Ty::Int(int_ty) => int_ty.size as u32,
            Ty::Enum(enum_ty) => enum_ty.base_type.size as u32,
            Ty::Struct(struct_ty) => struct_ty.size,
        }
    }
}

pub struct Int {
    pub size: u8,
    pub signed: Signedness,
}
pub enum Signedness {
    Signed,
    Unsigned,
}

pub struct Enum {
    pub variants: Vec<EnumVariant>,
    pub base_type: Int,
}
pub struct EnumVariant {
    pub value: i64,
    pub name: Rc<String>,
}

/// Representation fo a structure or union type.
///
/// Unions may be represented by adding overlapping members. All members'
/// occupied byte range must fit into the structure's size.
pub struct Struct {
    pub members: Vec<StructMember>,
    pub size: u32,
}
pub struct StructMember {
    pub offset: u32,
    pub name: Rc<String>,
    pub tyid: TypeID,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Selection {
    pub tyid: TypeID,
    pub path: SmallVec<[SelectStep; 4]>,
}
#[derive(Debug, PartialEq, Eq)]
pub enum SelectStep {
    Index(i64),
    Member(Rc<String>),
}
impl SelectStep {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_struct() {
        let mut types = TypeSet::new();

        let ty_point = types.add(Type {
            name: Rc::new("point".to_owned()),
            ty: Ty::Struct(Struct {
                size: 24,
                members: vec![
                    StructMember {
                        offset: 0,
                        name: Rc::new("x".to_owned()),
                        tyid: TYID_I32,
                    },
                    // a bit of padding
                    StructMember {
                        offset: 8,
                        name: Rc::new("y".to_owned()),
                        tyid: TYID_I32,
                    },
                    StructMember {
                        offset: 12,
                        name: Rc::new("cost".to_owned()),
                        tyid: TYID_I8,
                    },
                ],
            }),
        });

        {
            let s = types.select(ty_point, 0..4).unwrap();
            assert_eq!(s.tyid, TYID_I32);
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
            assert_eq!(s.tyid, TYID_I8);
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
        types.add(Type {
            name: Rc::new("point".to_owned()),
            ty: Ty::Struct(Struct {
                size: 4,
                members: vec![StructMember {
                    offset: 4,
                    name: Rc::new("x".to_owned()),
                    tyid: TYID_I32,
                }],
            }),
        });
    }
}
