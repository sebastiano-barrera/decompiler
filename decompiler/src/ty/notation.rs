mod parse;

use std::{sync::Arc, u64};

use super::{Ty, TypeID, TypeSet};
pub use parse::{parse, ParseError, ParseResult};

#[derive(Debug, PartialEq, Eq)]
pub struct TypeBuilder {
    program: Vec<Step>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Step {
    Ptr,
    Int {
        size_bytes: u8,
    },
    Uint {
        size_bytes: u8,
    },
    Float {
        size_bytes: u8,
    },
    Array {
        count: u32,
    },
    Struct {
        count: u32,
    },
    Func {
        count: i32,
    },
    /// Reference to an existing type ID.
    Ref {
        tyid: TypeID,
    },
    Void,
}

impl Default for TypeBuilder {
    fn default() -> Self {
        Self::empty()
    }
}
impl TypeBuilder {
    pub fn empty() -> Self {
        TypeBuilder {
            program: Vec::new(),
        }
    }

    pub fn build_types(&self, types: &TypeSet) -> super::Result<TypeID> {
        // TODO better strateg for generating TypeIDs. I'm almost embarrassed
        // that this works.
        let tyids = {
            let rtx = types.read_tx()?;
            let rtx = rtx.read();

            let mut tyids = vec![TypeID(0); self.program.len()];

            let mut last_tyid = TypeID(u64::MAX);
            for ndx in 0..self.program.len() {
                while rtx.is_tyid_defined(last_tyid)? {
                    last_tyid.0 -= 1;
                }
                tyids[ndx] = last_tyid;
                last_tyid.0 -= 1;
            }

            tyids
        };

        let mut stack = Vec::new();

        let mut wtx = types.write_tx()?;
        {
            let mut wtx = wtx.write();

            for (step, &tyid) in self.program.iter().zip(&tyids) {
                match step {
                    Step::Void => {
                        wtx.set(tyid, Ty::Void)?;
                    }
                    Step::Ptr => {
                        let subj = stack.pop().unwrap();
                        wtx.set(tyid, Ty::Ptr(subj))?;
                    }
                    Step::Int { size_bytes } => {
                        let ty = Ty::Int(super::Int {
                            size: *size_bytes,
                            signed: super::Signedness::Signed,
                        });
                        wtx.set(tyid, ty)?;
                    }
                    Step::Uint { size_bytes } => {
                        let ty = Ty::Int(super::Int {
                            size: *size_bytes,
                            signed: super::Signedness::Unsigned,
                        });
                        wtx.set(tyid, ty)?;
                    }
                    Step::Float { size_bytes } => {
                        let ty = Ty::Float(super::Float { size: *size_bytes });
                        wtx.set(tyid, ty)?;
                    }
                    Step::Array { count } => {
                        let subj = stack.pop().unwrap();
                        wtx.set(
                            tyid,
                            Ty::Array(super::Array {
                                element_tyid: subj,
                                index_subrange: crate::ty::Subrange {
                                    lo: 0,
                                    hi: Some(*count as i64),
                                },
                            }),
                        )?;
                    }
                    Step::Struct { count } => {
                        // TODO alignment, padding etc.
                        let mut total_size = 0;
                        let mut members = Vec::new();
                        for memb_ndx in 0..*count {
                            let subj = stack.pop().unwrap();
                            members.push(super::StructMember {
                                offset: total_size,
                                name: Arc::new(format!("memb{}", memb_ndx)),
                                tyid: subj,
                            });
                            // TODO unsized types should be rejected as invalid instead
                            total_size += wtx.read().bytes_size(subj)?.unwrap_or(0);
                        }

                        members.reverse();

                        let ty = Ty::Struct(super::Struct {
                            members,
                            size: total_size,
                        });
                        wtx.set(tyid, ty)?;
                    }
                    Step::Func { count } => {
                        let return_tyid = stack.pop().unwrap();

                        let mut param_tyids = Vec::new();
                        for _ in 0..*count {
                            let subj = stack.pop().unwrap();
                            param_tyids.push(subj);
                        }
                        param_tyids.reverse();

                        let ty = Ty::Subroutine(super::Subroutine {
                            return_tyid,
                            param_names: vec![None; param_tyids.len()],
                            param_tyids,
                        });
                        wtx.set(tyid, ty)?;
                    }
                    Step::Ref { tyid: target_tyid } => {
                        wtx.set(tyid, Ty::Alias(*target_tyid))?;
                    }
                }

                stack.push(tyid);
            }
        }

        let top_tyid = stack.pop().unwrap();
        assert!(stack.is_empty());
        wtx.commit()?;
        Ok(top_tyid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_func_stack_program_example() {
        //   func(u8, i32) void
        let builder = TypeBuilder {
            program: vec![
                Step::Int { size_bytes: 4 },
                Step::Uint { size_bytes: 1 },
                Step::Void,
                Step::Func { count: 2 },
            ],
        };

        let types = crate::ty::TypeSet::new();
        let root_tyid = builder.build_types(&types).unwrap();

        let rtx = types.read_tx().unwrap();
        let ty_cow = rtx.read().get(root_tyid).unwrap().unwrap();
        let Ty::Subroutine(crate::ty::Subroutine {
            return_tyid,
            param_names,
            param_tyids,
        }) = ty_cow.as_ref()
        else {
            panic!("not a subroutine! {:?}", ty_cow);
        };

        let mut tyids = vec![root_tyid, *return_tyid];
        tyids.extend(param_tyids.iter().copied());
        for i in 0..tyids.len() {
            for j in i + 1..tyids.len() {
                assert_ne!(tyids[i], tyids[j]);
            }
        }
    }
}
