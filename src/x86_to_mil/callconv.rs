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

    for (param_ndx, &param_tyid) in param_types.iter().enumerate() {
        let param_buf = &mut [RegClass::Undefined; 8];
        let param_class = classify_param(types, param_tyid, param_buf);

        println!("param {param_ndx}: {param_class:#?}");

        let param_class = match param_class {
            Ok(ParamClass::Registers(clss)) if enough_registers_available(&state, clss) => {
                ParamClass::Registers(clss)
            }
            Ok(ParamClass::Registers(_)) => ParamClass::Memory,
            Ok(ParamClass::Memory) => ParamClass::Memory,
            Err(_) => {
                // stop parsing parameters
                // (a problem in this parameter prevents us from parsing any further parameter type)
                // TODO collect these errors and move them out
                eprintln!("warning: read_func_params stopped after {param_ndx} parameters");
                break;
            }
        };

        let src_anc = state
            .pull_arg()
            .ok_or_else(|| anyhow!("not enough arg ancestrals!"))?;
        let src = bld.reg_gen.next();
        bld.emit(src, Insn::Ancestral(src_anc));
        emit_read_param(bld, src, param_tyid, &param_class, &mut state)?;
    }

    Ok(())
}

fn enough_registers_available(state: &State, clss: &[RegClass]) -> bool {
    let mut int_count = 0;
    let mut sse_count = 0;

    for cls in clss {
        match cls {
            RegClass::Undefined => panic!(),
            RegClass::Integer => {
                int_count += 1;
            }
            RegClass::Sse => {
                sse_count += 1;
            }
        }
    }

    state.available_integer_regs() >= int_count && state.available_sse_regs() >= sse_count
}

fn emit_read_param<'a>(
    bld: &mut Builder<'a>,
    src: mil::Reg,
    param_tyid: ty::TypeID,
    param_class: &ParamClass,
    state: &mut State,
) -> anyhow::Result<()> {
    // TODO redesign this thing so that we don't have to re-do this lookup all the time...
    let types = bld.types.unwrap();
    let ty = &types.get(param_tyid).unwrap().ty;

    let sz = ty.bytes_size() as usize;

    state.assert_stack_qword_aligned();

    match param_class {
        ParamClass::Registers(eb_clss) => match ty {
            ty::Ty::Ptr(_) | ty::Ty::Enum(_) | ty::Ty::Int(_) => {
                assert_eq!(eb_clss.len(), 1);

                let eb_cls = eb_clss[0];
                assert!(matches!(eb_cls, RegClass::Integer));

                let dest = state.pull_integer_reg().unwrap();
                let insn = match sz {
                    1 => Insn::V8WithL1(dest, src),
                    2 => Insn::V8WithL2(dest, src),
                    4 => Insn::V8WithL4(dest, src),
                    8 => Insn::Get8(src),
                    _ => panic!("invalid size for an integer: {sz}"),
                };
                bld.emit(dest, insn);
            }
            ty::Ty::Struct(_) => {
                assert_eq!(eb_clss.len(), sz.div_ceil(8));
                for (eb_ndx, eb_cls) in eb_clss.iter().enumerate() {
                    let offset = eb_ndx * 8;

                    match eb_cls {
                        RegClass::Undefined => panic!("unassigned class for param!"),
                        RegClass::Integer => {
                            // the case where there is no available integer reg is
                            // excluded by `are_registers_available`
                            let dest = state.pull_integer_reg().unwrap();
                            let insn = Insn::StructGet8 {
                                struct_value: src,
                                offset: offset.try_into().unwrap(),
                            };
                            bld.emit(dest, insn);
                        }
                        RegClass::Sse => todo!("SSE registers not yet supported"),
                    }
                }
            }
            ty::Ty::Float(_) => todo!("float parameters not yet supported"),

            ty::Ty::Bool(_) | ty::Ty::Subroutine(_) | ty::Ty::Unknown(_) | ty::Ty::Void => {
                panic!("invalid type for parameter")
            }
        },
        ParamClass::Memory => match ty {
            ty::Ty::Int(_) | ty::Ty::Enum(_) | ty::Ty::Ptr(_) => {
                let slot_sz = 8 * sz.div_ceil(8);

                let slot_ofs = state.pull_stack_slot(slot_sz) as i64;
                let reg = bld.reg_gen.next();
                bld.emit(reg, Insn::ArithK8(ArithOp::Add, Builder::RSP, 8 + slot_ofs));
                bld.emit(reg, Insn::StoreMem(reg, src));
            }

            ty::Ty::Struct(_) => {
                let eb_count = sz.div_ceil(8);
                let reg_addr = bld.reg_gen.next();
                let reg_part = bld.reg_gen.next();

                for eb_ndx in 0..eb_count {
                    let slot_ofs = state.pull_stack_slot(8) as i64;
                    bld.emit(
                        reg_addr,
                        Insn::ArithK8(ArithOp::Add, Builder::RSP, 8 + slot_ofs),
                    );

                    let eb_ndx: u8 = eb_ndx.try_into().unwrap();
                    bld.emit(
                        reg_part,
                        Insn::StructGet8 {
                            struct_value: src,
                            offset: 8 * eb_ndx,
                        },
                    );
                    bld.emit(reg_addr, Insn::StoreMem(reg_addr, reg_part));
                }
            }
            ty::Ty::Float(_) => todo!("float parameters not yet supported"),
            ty::Ty::Bool(_) | ty::Ty::Subroutine(_) | ty::Ty::Unknown(_) | ty::Ty::Void => {
                panic!("invalid type for parameter")
            }
        },
    }

    Ok(())
}

static INTEGER_REGS: [Reg; 6] = [
    Builder::RDI,
    Builder::RSI,
    Builder::RDX,
    Builder::RCX,
    Builder::R8,
    Builder::R9,
];

struct State {
    int_regs: &'static [Reg],
    sse_regs: &'static [Reg],
    args: &'static [AncestralName],
    stack_offset: usize,
}
impl State {
    fn available_integer_regs(&self) -> usize {
        self.int_regs.len()
    }

    fn pull_integer_reg(&mut self) -> Option<Reg> {
        pull_slice(&mut self.int_regs)
    }

    fn available_sse_regs(&self) -> usize {
        self.sse_regs.len()
    }

    fn pull_sse_reg(&mut self) -> Option<Reg> {
        pull_slice(&mut self.sse_regs)
    }

    fn pull_arg(&mut self) -> Option<AncestralName> {
        pull_slice(&mut self.args)
    }

    fn assert_stack_qword_aligned(&self) {
        assert_eq!(self.stack_offset % 8, 0);
    }

    fn pull_stack_slot(&mut self, slot_sz: usize) -> usize {
        let ofs = self.stack_offset;
        self.stack_offset += slot_sz;
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
            stack_offset: 0,
        }
    }
}

#[derive(Debug)]
enum ParamClass<'a> {
    Registers(&'a [RegClass]),
    Memory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegClass {
    Undefined,
    Integer,
    Sse,
}
impl RegClass {
    fn merge_with(&self, other: Self) -> Self {
        match (self, &other) {
            (_, RegClass::Undefined) => panic!("invalid merged ScalarClass: None"),
            (RegClass::Undefined, other) => *other,

            (RegClass::Sse, RegClass::Sse) => RegClass::Sse,

            (RegClass::Integer, _) | (_, RegClass::Integer) => RegClass::Integer,
        }
    }
}

fn classify_param<'a>(
    types: &ty::TypeSet,
    param_tyid: ty::TypeID,
    classes: &'a mut [RegClass; 8],
) -> anyhow::Result<ParamClass<'a>> {
    // scalar types are the same as a struct with a single member
    if classify_struct_member(types, param_tyid, 0, classes)? {
        return Ok(ParamClass::Memory);
    }

    // eightbytes: count of how many were used
    let eb_count = classes
        .iter()
        .take_while(|&&slot| slot != RegClass::Undefined)
        .count();
    Ok(ParamClass::Registers(&classes[..eb_count]))
}

fn classify_struct_member(
    types: &ty::TypeSet,
    tyid: ty::TypeID,
    offset: u32,
    classes: &mut [RegClass; 8],
) -> anyhow::Result<bool> {
    let ty = &types.get(tyid).unwrap().ty;
    let sz = ty.bytes_size();

    if (offset + sz).div_ceil(8) as usize >= classes.len() {
        // not enough space for this member in 8 eightbytes
        return Ok(true);
    }

    let align = types.alignment(tyid).ok_or_else(|| {
        let name = types.get(tyid).unwrap().name.as_str();
        anyhow!(
            "struct type is not fully known (name '{}', tyid {:?}, offset {})",
            name,
            tyid,
            offset
        )
    })? as u32;
    if offset % align != 0 {
        // offset is not a multiple of the member's size, i.e. it's "unaligned"
        return Ok(true);
    }

    match &ty {
        // size is already in `sz`
        ty::Ty::Ptr(_) | ty::Ty::Enum(_) | ty::Ty::Int(_) => {
            let eb_offset = offset / 8;
            let eb_count = sz.div_ceil(8);

            for eb_ndx in 0..eb_count {
                let slot = &mut classes[(eb_offset + eb_ndx) as usize];
                *slot = slot.merge_with(RegClass::Integer);
            }

            Ok(false)
        }
        ty::Ty::Struct(struct_ty) => {
            assert_ne!(struct_ty.size, 0);
            assert_ne!(struct_ty.members.len(), 0);

            for (memb_ndx, memb) in struct_ty.members.iter().enumerate() {
                if classify_struct_member(types, memb.tyid, memb.offset, classes)
                    .with_context(|| format!("member #{} offset {} tyid", memb_ndx, memb.offset))?
                {
                    return Ok(true);
                }
            }

            Ok(false)
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

    use crate::ssa;

    use super::*;
    use std::sync::Arc;

    #[test]
    fn param_i32() {
        let types = make_types();
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
        let prog = ssa::mil_to_ssa(ssa::ConversionParams { program: prog });
        let snap = format!("params: {:?}\nprogram:\n{:?}", param_types, prog);
        snap
    }

    struct Types {
        types: ty::TypeSet,
        tyid_i32: ty::TypeID,
        tyid_i8: ty::TypeID,
        tyid_point: ty::TypeID,
    }

    fn make_types() -> Types {
        use ty::{Int, Signedness, Struct, StructMember, Ty, Type, TypeSet};

        let mut types = TypeSet::new();

        let tyid_i32 = types.add(Type {
            name: Arc::new("i32".to_owned()),
            ty: Ty::Int(Int {
                size: 4,
                signed: Signedness::Signed,
            }),
        });

        let tyid_i8 = types.add(Type {
            name: Arc::new("i8".to_owned()),
            ty: Ty::Int(Int {
                size: 1,
                signed: Signedness::Signed,
            }),
        });

        let tyid_point = types.add(Type {
            name: Arc::new("point".to_owned()),
            ty: Ty::Struct(Struct {
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
            }),
        });

        Types {
            types,
            tyid_i32,
            tyid_i8,
            tyid_point,
        }
    }
}
