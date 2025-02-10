use crate::{
    mil::{self, AncestralName, ArithOp, Insn, Reg},
    ty,
};

use super::Builder;

use anyhow::{anyhow, Context};

pub fn read_func_params<'a>(
    bld: &mut Builder<'a>,
    param_types: &[ty::TypeID],
) -> anyhow::Result<()> {
    let types = bld
        .types
        .ok_or(anyhow!("No TypeSet passed to the Builder"))?;

    let mut state = State::default();

    for &param_tyid in param_types.iter() {
        let param_ty = &types.get(param_tyid).unwrap().ty;
        let sz = param_ty.bytes_size() as usize;

        let src_anc = state
            .pull_arg()
            .ok_or_else(|| anyhow!("not enough arg ancestrals!"))?;
        let src = bld.reg_gen.next();
        bld.init_ancestral(src, src_anc, mil::RegType::Bytes(sz));

        match param_ty {
            ty::Ty::Int(_) | ty::Ty::Bool(_) | ty::Ty::Ptr(_) | ty::Ty::Enum(_) => {
                let sz = sz.try_into().unwrap();
                if let Some(dest) = state.pull_integer_reg() {
                    emit_partial_write(bld, src, dest, 0, sz);
                } else {
                    let src_value = bld.reg_gen.next();
                    emit_partial_write(bld, src, src_value, 0, sz);

                    let slot_ofs = state.pull_stack_slot() as i64;
                    let addr = bld.reg_gen.next();
                    bld.emit(addr, Insn::ArithK8(ArithOp::Add, Builder::RSP, slot_ofs));
                    bld.emit(addr, Insn::StoreMem(addr, src_value));
                }
            }

            ty::Ty::Float(_) => {
                let sz = sz.try_into().unwrap();
                if let Some(dest) = state.pull_sse_reg() {
                    emit_partial_write(bld, src, dest, 0, sz);
                } else {
                    // TODO refactor, deduplicate
                    let src_value = bld.reg_gen.next();
                    emit_partial_write(bld, src, src_value, 0, sz);

                    let slot_ofs = state.pull_stack_slot() as i64;
                    let addr = bld.reg_gen.next();
                    bld.emit(addr, Insn::ArithK8(ArithOp::Add, Builder::RSP, slot_ofs));
                    bld.emit(addr, Insn::StoreMem(addr, src_value));
                }
            }

            ty::Ty::Struct(_) => {
                read_struct(bld, &mut state, &types, param_tyid, src)?;
            }

            other @ (ty::Ty::Subroutine(_) | ty::Ty::Unknown(_) | ty::Ty::Void) => {
                panic!("invalid types for function params: {:?}", other)
            }
        }
    }

    Ok(())
}

/// Try reading the struct type identified by `struct_tyid` from machine registers.
///
/// If, for some reason, the x86_64 calling convention does not allow the struct
/// to be passed via machine registers, `None` is returned. In this case, no
/// instructions are emitted, and `state` is not changed.
///
/// In the successful case where the calling convention allows passing via
/// registers, then the necessary instructions are emitted, `state` is updated,
/// and `Some(())` is returned.
fn read_struct<'a>(
    bld: &mut Builder<'a>,
    state: &mut State,
    types: &ty::TypeSet,
    struct_tyid: ty::TypeID,
    src: mil::Reg,
) -> anyhow::Result<()> {
    let ty = &types.get(struct_tyid).unwrap().ty;
    let sz = ty.bytes_size();

    let eb_count = sz.div_ceil(8) as usize;
    assert!(eb_count > 0);
    assert!(eb_count <= 8);

    // TODO increase the limit to 8 to support {SSE, SSEUP, SSEUP, ...}
    if eb_count > 2 {
        return read_struct_from_memory(bld, state, types, struct_tyid, src);
    }

    let mut buf_cls = [RegClass::Unused; 8];
    let pass_mode = classify_struct_member(types, struct_tyid, 0, &mut buf_cls)?;
    if pass_mode != StructPassMode::Reg {
        return read_struct_from_memory(bld, state, types, struct_tyid, src);
    }

    let clss = &buf_cls[..eb_count];
    assert!(clss.iter().all(|&cls| cls != RegClass::Unused));

    let mut buf_reg = [mil::Reg(u16::MAX); 8];
    let Some(regs) = state.try_(|state| {
        for (i, cls) in clss.iter().enumerate() {
            buf_reg[i] = match cls {
                RegClass::Unused => {
                    panic!("reg class left Unused by classify_struct_member")
                }
                RegClass::Integer => state.pull_integer_reg()?,
                RegClass::Sse => state.pull_sse_reg()?,
            };
        }
        Some(&buf_reg[..clss.len()])
    }) else {
        return read_struct_from_memory(bld, state, types, struct_tyid, src);
    };

    // TODO flatten the struct...
    let mut queue = Vec::with_capacity(16);
    queue.push((0u32, src, ty));
    while let Some((offset, src, ty)) = queue.pop() {
        match ty {
            ty::Ty::Int(_)
            | ty::Ty::Bool(_)
            | ty::Ty::Enum(_)
            | ty::Ty::Ptr(_)
            | ty::Ty::Float(_) => {
                let eb_ndx = (offset / 8) as usize;
                let ofs_in_eb = (offset % 8) as u8;
                let size = ty.bytes_size().try_into().unwrap();
                let dest = regs[eb_ndx];
                emit_partial_write(bld, src, dest, ofs_in_eb, size);
            }

            ty::Ty::Struct(struct_ty) => {
                let memb_value = bld.reg_gen.next();

                for memb in &struct_ty.members {
                    let memb_ty = &types.get(memb.tyid).unwrap().ty;
                    bld.emit(
                        memb_value,
                        Insn::StructGetMember {
                            struct_value: src,
                            // TODO Allocate this somewhere where it makes sense
                            name: memb.name.as_ref().clone().leak(),
                            size: memb_ty.bytes_size(),
                        },
                    );
                    queue.push((memb.offset, memb_value, memb_ty));
                }
            }

            ty::Ty::Subroutine(_) | ty::Ty::Unknown(_) | ty::Ty::Void => {
                panic!("invalid type for a struct member: {:?}", ty)
            }
        }
    }

    Ok(())
}

fn read_struct_from_memory<'a>(
    bld: &mut Builder<'a>,
    state: &mut State,
    types: &ty::TypeSet,
    struct_tyid: ty::TypeID,
    src: mil::Reg,
) -> anyhow::Result<()> {
    let addr = bld.reg_gen.next();

    let ty::Ty::Struct(struct_ty) = &types.get(struct_tyid).unwrap().ty else {
        panic!("must be called with a struct")
    };

    let eb_count = struct_ty.size.div_ceil(8) as usize;
    let stack_base = state.pull_stack_slots(eb_count) as i64;

    let memb_value = bld.reg_gen.next();

    for memb in &struct_ty.members {
        let size = types.bytes_size(memb.tyid).unwrap();
        bld.emit(
            memb_value,
            Insn::StructGetMember {
                struct_value: src,
                // TODO Allocate this somewhere where it makes sense
                name: memb.name.as_ref().clone().leak(),
                size,
            },
        );
        bld.emit(
            addr,
            Insn::ArithK8(ArithOp::Add, Builder::RSP, stack_base + memb.offset as i64),
        );
        bld.emit(addr, Insn::StoreMem(addr, memb_value));
    }

    Ok(())
}

fn emit_partial_write(bld: &mut Builder, src: mil::Reg, dest: mil::Reg, offset: u8, size: u8) {
    // dest[offset+size:8] ++ src[0:size] ++ dest[0:offset]

    let t0 = bld.reg_gen.next();
    let t1 = bld.reg_gen.next();

    bld.emit(
        t0,
        mil::Insn::Part {
            src: dest,
            offset: 0,
            size: offset,
        },
    );
    bld.emit(
        t1,
        mil::Insn::Part {
            src,
            offset: 0,
            size,
        },
    );
    bld.emit(t1, mil::Insn::Concat { lo: t0, hi: t1 });
    bld.emit(
        t0,
        mil::Insn::Part {
            src: dest,
            offset: offset + size,
            size: 8 - offset - size,
        },
    );
    bld.emit(dest, mil::Insn::Concat { lo: t1, hi: t0 });
}

static INTEGER_REGS: [Reg; 6] = [
    Builder::RDI,
    Builder::RSI,
    Builder::RDX,
    Builder::RCX,
    Builder::R8,
    Builder::R9,
];

#[derive(Clone)]
struct State {
    int_regs: &'static [Reg],
    sse_regs: &'static [Reg],
    args: &'static [AncestralName],
    stack_offset: usize,
}
impl State {
    fn try_<F, R>(&mut self, action: F) -> Option<R>
    where
        F: FnOnce(&mut Self) -> Option<R>,
    {
        let self_bak = self.clone();
        let ret = action(self);
        if ret.is_none() {
            *self = self_bak;
        }
        ret
    }

    fn pull_integer_reg(&mut self) -> Option<Reg> {
        pull_slice(&mut self.int_regs)
    }

    fn pull_sse_reg(&mut self) -> Option<Reg> {
        pull_slice(&mut self.sse_regs)
    }

    fn pull_arg(&mut self) -> Option<AncestralName> {
        pull_slice(&mut self.args)
    }

    #[inline(always)]
    fn assert_stack_qword_aligned(&self) {
        assert_eq!(self.stack_offset % 8, 0);
    }

    fn pull_stack_slot(&mut self) -> usize {
        self.pull_stack_slots(1)
    }
    fn pull_stack_slots(&mut self, count: usize) -> usize {
        self.assert_stack_qword_aligned();
        let ofs = self.stack_offset;
        self.stack_offset += 8 * count;
        ofs
    }
}

fn pull_slice<T: Clone>(slice: &mut &[T]) -> Option<T> {
    let (next, rest) = slice.split_first()?;
    *slice = rest;
    Some(next.clone())
}

impl Default for State {
    fn default() -> Self {
        State {
            int_regs: &INTEGER_REGS,
            // TODO!
            sse_regs: &[],
            args: &super::ANC_ARGS,
            // we start at 8, because the first eightbyte in the stack is for the return address
            stack_offset: 8,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum StructPassMode {
    Reg,
    Memory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegClass {
    Unused,
    Integer,
    Sse,
}
impl RegClass {
    fn merge_with(&self, other: Self) -> Self {
        match (self, &other) {
            (_, RegClass::Unused) => panic!("invalid merged ScalarClass: Unused"),
            (RegClass::Unused, other) => *other,

            (RegClass::Sse, RegClass::Sse) => RegClass::Sse,

            (RegClass::Integer, _) | (_, RegClass::Integer) => RegClass::Integer,
        }
    }
}

fn classify_struct_member<'a>(
    types: &ty::TypeSet,
    tyid: ty::TypeID,
    offset: u32,
    classes: &mut [RegClass; 8],
) -> anyhow::Result<StructPassMode> {
    let ty = &types.get(tyid).unwrap().ty;
    let sz = ty.bytes_size();

    assert_ne!(sz, 0);

    let eb_start = offset as usize / 8;
    let eb_end = (offset + sz).div_ceil(8) as usize;

    if eb_end as usize >= classes.len() {
        // not enough space for this member in 8 eightbytes
        return Ok(StructPassMode::Memory);
    }

    let Some(align) = types.alignment(tyid) else {
        let name = types.get(tyid).unwrap().name.as_str();
        return Err(anyhow!(
            "struct type is not fully known (name '{}', tyid {:?}, offset {})",
            name,
            tyid,
            offset
        ));
    };
    if offset % align as u32 != 0 {
        // offset is not a multiple of the member's size, i.e. it's "unaligned"
        return Ok(StructPassMode::Memory);
    }

    match &ty {
        ty::Ty::Ptr(_) | ty::Ty::Enum(_) | ty::Ty::Int(_) => {
            assert!(sz <= 8);
            assert_eq!(eb_end, eb_start + 1);

            let slot = &mut classes[eb_start];
            *slot = slot.merge_with(RegClass::Integer);

            Ok(StructPassMode::Reg)
        }
        ty::Ty::Struct(struct_ty) => {
            assert_ne!(struct_ty.members.len(), 0);

            for (memb_ndx, memb) in struct_ty.members.iter().enumerate() {
                let pass_mode = classify_struct_member(types, memb.tyid, memb.offset, classes)
                    .with_context(|| format!("member #{} offset {} tyid", memb_ndx, memb.offset))?;

                if let StructPassMode::Memory = pass_mode {
                    return Ok(StructPassMode::Memory);
                }
            }

            Ok(StructPassMode::Reg)
        }
        ty::Ty::Float(_) => Err(anyhow!("unsupported: function parameters of type float")),

        ty::Ty::Bool(_) | ty::Ty::Subroutine(_) | ty::Ty::Unknown(_) | ty::Ty::Void => {
            panic!("invalid types for struct members")
        }
    }
}

#[cfg(test)]
mod tests {
    use insta::assert_snapshot;
    use ty::TypeID;

    use crate::{ssa, xform};

    use super::*;
    use std::sync::Arc;

    #[test]
    fn param_i32() {
        let types = make_scalars();
        let param_types = &[types.tyid_i32];
        let snap = check_types(&types, param_types);
        assert_snapshot!(snap);
    }

    fn check_types(types: &Types, param_types: &[ty::TypeID]) -> String {
        let mut bld = Builder::new();
        bld.use_types(&types.types);

        read_func_params(&mut bld, param_types).unwrap();

        let v0 = bld.reg_gen.next();
        bld.emit(v0, Insn::Ret(Builder::RDI));

        let prog = bld.build();
        let mut prog = ssa::mil_to_ssa(ssa::ConversionParams { program: prog });
        xform::simplify_half_null_concat(&mut prog);
        let snap = format!("params: {:?}\nprogram:\n{:?}", param_types, prog);
        snap
    }

    struct Types {
        types: ty::TypeSet,
        tyid_i64: ty::TypeID,
        tyid_i32: ty::TypeID,
        tyid_i16: ty::TypeID,
        tyid_i8: ty::TypeID,
        tyid_f32: ty::TypeID,
        tyid_f64: ty::TypeID,
        tyid_f256: ty::TypeID,
    }

    // TOOD share the result (e.g. as a OnceCell)
    fn make_scalars() -> Types {
        use ty::{Int, Signedness, Ty, Type, TypeSet};

        let mut types = TypeSet::new();

        fn mk_int(name: &str, size: u8) -> Type {
            Type {
                name: Arc::new(name.to_owned()),
                ty: Ty::Int(Int {
                    size,
                    signed: Signedness::Signed,
                }),
            }
        }

        let tyid_i8 = types.add(mk_int("i8", 1));
        let tyid_i16 = types.add(mk_int("i16", 2));
        let tyid_i32 = types.add(mk_int("i32", 4));
        let tyid_i64 = types.add(mk_int("i64", 8));

        let tyid_f32 = types.add(Type {
            name: Arc::new("float32".to_owned()),
            ty: Ty::Float(ty::Float { size: 4 }),
        });
        let tyid_f64 = types.add(Type {
            name: Arc::new("float64".to_owned()),
            ty: Ty::Float(ty::Float { size: 8 }),
        });
        let tyid_f256 = types.add(Type {
            name: Arc::new("float256".to_owned()),
            ty: Ty::Float(ty::Float { size: 32 }),
        });

        Types {
            types,
            tyid_i64,
            tyid_i32,
            tyid_i16,
            tyid_i8,
            tyid_f32,
            tyid_f64,
            tyid_f256,
        }
    }

    #[test]
    fn classify_struct_one_int() {
        let mut scas = make_scalars();

        for sca_tyid in [scas.tyid_i8, scas.tyid_i16, scas.tyid_i32, scas.tyid_i64] {
            let struct_tyid = make_sample_struct(&mut scas.types, &[(0, sca_tyid)]);

            let buf = &mut [RegClass::Unused; 8];
            let class = classify_struct_member(&scas.types, struct_tyid, 0, buf).unwrap();

            assert_eq!(class, StructPassMode::Reg);
            assert_eq!(
                &buf[..],
                &[
                    RegClass::Integer,
                    RegClass::Unused,
                    RegClass::Unused,
                    RegClass::Unused,
                    RegClass::Unused,
                    RegClass::Unused,
                    RegClass::Unused,
                    RegClass::Unused,
                ]
            );
        }
    }

    #[test]
    fn classify_struct_ints_fit_1_eb() {
        let mut scas = make_scalars();

        let cases: &[&[_]] = &[
            &[
                (0, scas.tyid_i8),
                (1, scas.tyid_i8),
                (2, scas.tyid_i8),
                (3, scas.tyid_i8),
                (4, scas.tyid_i8),
                (5, scas.tyid_i8),
                (6, scas.tyid_i8),
                (7, scas.tyid_i8),
            ],
            &[(0, scas.tyid_i32), (4, scas.tyid_i32)],
            &[(0, scas.tyid_i32), (4, scas.tyid_i16)],
            &[(0, scas.tyid_i32), (4, scas.tyid_i8)],
            &[(0, scas.tyid_i32), (6, scas.tyid_i16)],
            &[(0, scas.tyid_i32), (7, scas.tyid_i8)],
            &[(0, scas.tyid_i32), (4, scas.tyid_i32)],
            &[(0, scas.tyid_i16), (4, scas.tyid_i32)],
            &[(0, scas.tyid_i8), (4, scas.tyid_i32)],
            &[(0, scas.tyid_i16), (4, scas.tyid_i32)],
            &[(0, scas.tyid_i8), (4, scas.tyid_i32)],
            &[(0, scas.tyid_i64)],
        ];

        for &members in cases {
            let struct_tyid = make_sample_struct(&mut scas.types, members);

            let buf = &mut [RegClass::Unused; 8];
            let class = classify_struct_member(&scas.types, struct_tyid, 0, buf).unwrap();

            assert_eq!(class, StructPassMode::Reg);
            assert_eq!(
                &buf[..],
                &[
                    RegClass::Integer,
                    RegClass::Unused,
                    RegClass::Unused,
                    RegClass::Unused,
                    RegClass::Unused,
                    RegClass::Unused,
                    RegClass::Unused,
                    RegClass::Unused,
                ]
            );
        }
    }

    #[test]
    fn classify_struct_one_float() {
        let mut scas = make_scalars();

        for sca_tyid in [scas.tyid_f32, scas.tyid_f64] {
            let struct_tyid = make_sample_struct(&mut scas.types, &[(0, sca_tyid)]);

            let buf = &mut [RegClass::Unused; 8];
            let class = classify_struct_member(&scas.types, struct_tyid, 0, buf).unwrap();

            assert_eq!(class, StructPassMode::Reg);
            assert_eq!(
                &buf[..],
                &[
                    RegClass::Sse,
                    RegClass::Unused,
                    RegClass::Unused,
                    RegClass::Unused,
                    RegClass::Unused,
                    RegClass::Unused,
                    RegClass::Unused,
                    RegClass::Unused,
                ]
            );
        }
    }

    #[test]
    fn classify_struct_two_ints() {
        let mut scas = make_scalars();

        let cases: &[&[_]] = &[
            &[(0, scas.tyid_i8), (8, scas.tyid_i8)],
            &[(0, scas.tyid_i8), (4, scas.tyid_i8), (8, scas.tyid_i8)],
            &[
                (0, scas.tyid_i8),
                (8, scas.tyid_i8),
                (9, scas.tyid_i8),
                (12, scas.tyid_i8),
            ],
            &[(0, scas.tyid_i16), (8, scas.tyid_i16)],
            &[(0, scas.tyid_i32), (8, scas.tyid_i32)],
            &[(0, scas.tyid_i64), (8, scas.tyid_i64)],
        ];

        for &members in cases {
            let struct_tyid = make_sample_struct(&mut scas.types, members);

            let buf = &mut [RegClass::Unused; 8];
            let class = classify_struct_member(&scas.types, struct_tyid, 0, buf).unwrap();

            assert_eq!(class, StructPassMode::Reg);
            assert_eq!(
                &buf[..],
                &[
                    RegClass::Integer,
                    RegClass::Integer,
                    RegClass::Unused,
                    RegClass::Unused,
                    RegClass::Unused,
                    RegClass::Unused,
                    RegClass::Unused,
                    RegClass::Unused,
                ]
            );
        }
    }

    #[test]
    fn classify_struct_larger() {
        let mut scas = make_scalars();

        let cases: &[&[_]] = &[
            &[(0, scas.tyid_i8), (8, scas.tyid_i8), (64, scas.tyid_i8)],
            &[(0, scas.tyid_i16), (8, scas.tyid_i16), (64, scas.tyid_i16)],
            &[(0, scas.tyid_i32), (8, scas.tyid_i32), (64, scas.tyid_i32)],
            &[(0, scas.tyid_i64), (8, scas.tyid_i64), (64, scas.tyid_i64)],
            // unaligned members
            &[(1, scas.tyid_i16)],
            &[(1, scas.tyid_i32)],
            &[(1, scas.tyid_i64)],
        ];

        for &members in cases {
            println!("{:?}", members);
            let struct_tyid = make_sample_struct(&mut scas.types, members);
            let buf = &mut [RegClass::Unused; 8];
            let class = classify_struct_member(&scas.types, struct_tyid, 0, buf).unwrap();
            assert_eq!(class, StructPassMode::Memory);
        }
    }

    fn make_sample_struct(types: &mut ty::TypeSet, members: &[(u32, TypeID)]) -> TypeID {
        assert!(members.iter().is_sorted_by_key(|(ofs, _)| ofs));

        let struct_sz = {
            let &(ofs, tyid) = members.last().unwrap();
            let sz = types.bytes_size(tyid).unwrap();
            ofs + sz
        };

        let name = Arc::new("SampleStruct".to_owned());
        types.add(ty::Type {
            name,
            ty: ty::Ty::Struct(ty::Struct {
                size: struct_sz,
                members: members
                    .iter()
                    .map(|&(offset, tyid)| ty::StructMember {
                        offset,
                        name: Arc::new(format!("x{}", offset)),
                        tyid,
                    })
                    .collect(),
            }),
        })
    }
}
