use std::cmp::Ordering;

use crate::{
    mil::{self, ArithOp, Insn},
    ty,
};

use super::Builder;

use anyhow::{anyhow, Ok};
use tracing::{event, instrument, span, Level};

#[derive(Clone, Copy)]
enum PassMode {
    Regs([Option<iced_x86::Register>; 2]),
    OneBigSse {
        reg: iced_x86::Register,
        eb_count: u8,
    },
    Memory,
}

pub struct Report {
    pub ok_count: usize,
}

#[instrument(skip_all)]
pub fn unpack_params(
    bld: &mut Builder,
    param_types: &[ty::TypeID],
    ret_tyid: ty::TypeID,
) -> anyhow::Result<Report> {
    let mut report = Report { ok_count: 0 };
    let mut state = ParamPassing::default();

    prepare_for_return_value(bld, ret_tyid, &mut state)?;

    for (ndx, &param_tyid) in param_types.iter().enumerate() {
        let param_ty = bld.types.get(param_tyid).unwrap();

        let span = span!(Level::INFO, "param", ndx, tyid=?param_tyid, ty=?param_ty);
        let _enter = span.enter();

        let res = unpack_param(bld, &mut state, param_tyid);
        match res {
            Result::Err(err) => {
                event!(
                    Level::ERROR,
                    ?err,
                    ok_count = report.ok_count,
                    total_count = param_types.len(),
                    "parameter type could not be classified (remaining parameters can't be mapped)"
                );
                break;
            }
            Result::Ok(()) => {
                report.ok_count += 1;
            }
        }
    }

    Ok(report)
}

#[instrument(skip_all)]
fn unpack_param(
    bld: &mut Builder,
    state: &mut ParamPassing,
    tyid: ty::TypeID,
) -> anyhow::Result<()> {
    let mut eb_set = EightbytesSet::new_regs();
    classify_eightbytes(&mut eb_set, &bld.types, tyid, 0)?;

    let mode = eightbytes_to_pass_mode(eb_set, state);
    // .unwrap(): classify_eightbytes already checks that size is known
    let sz = bld.types.bytes_size(tyid).unwrap();

    let arg_value = bld.tmp_gen();

    {
        let arg_ndx = state.pull_arg();
        let arg_insn_ndx = bld.emit(
            arg_value,
            Insn::FuncArgument {
                index: arg_ndx,
                reg_type: mil::RegType::Bytes(sz),
            },
        );
        bld.set_value_type(arg_insn_ndx, tyid);
    }

    match mode {
        PassMode::Regs(regs) => {
            assert!(regs[0].is_some());

            // TODO assert that the type is 1-16 bytes long, based on clss
            for (ndx, reg) in regs.into_iter().enumerate() {
                let Some(reg) = reg else {
                    continue;
                };

                let offset = (8 * ndx).try_into().unwrap();
                // write the eightbyte from the source value (opaque) to the
                // destination reg, rounded up to 8 bytes
                let arg_value = match sz.cmp(&8) {
                    Ordering::Equal => arg_value,
                    Ordering::Less => {
                        let v0 = bld.tmp_gen();
                        bld.emit(
                            v0,
                            mil::Insn::Widen {
                                reg: arg_value,
                                target_size: 8,
                                sign: false,
                            },
                        );
                        v0
                    }
                    Ordering::Greater => {
                        let v0 = bld.tmp_gen();
                        bld.emit(
                            v0,
                            mil::Insn::Part {
                                src: arg_value,
                                offset,
                                size: 8,
                            },
                        );
                        v0
                    }
                };
                bld.emit_write_machine_reg(reg, 8, arg_value);
            }
        }
        PassMode::OneBigSse { reg, eb_count } => {
            assert_eq!(sz.div_ceil(8), eb_count as usize);
            bld.emit(
                arg_value,
                mil::Insn::Widen {
                    reg: arg_value,
                    target_size: 32, // 4 eightbytes, which is the largest
                    sign: false,
                },
            );
            bld.emit_write_machine_reg(reg, 32, arg_value);
        }
        PassMode::Memory => {
            let eb_count = sz.div_ceil(8);
            let sz: u16 = sz.try_into().unwrap();

            let addr = bld.tmp_gen();
            let eb = bld.tmp_gen();

            for eb_ndx in 0..eb_count {
                // relies on RSP never being assigned by any instruction emitted in this module
                let eb_offset = state.pull_stack_eightbyte(1) as i64;
                let offset: u16 = (eb_ndx * 8).try_into().unwrap();

                // the stack slot really always is 8 bytes, so: take an 8-bytes chunk of the
                // value or widen the "tail" part

                // how much of the value we're "storing" in the stack slot
                let part_size = (sz - offset).min(8);
                let eb_init_insn = if part_size < 8 {
                    Insn::Widen {
                        reg: arg_value,
                        target_size: 8,
                        sign: false,
                    }
                } else if part_size == 8 {
                    Insn::Part {
                        src: arg_value,
                        offset,
                        size: 8,
                    }
                } else {
                    unreachable!("part_size can't be > 8");
                };

                bld.emit(eb, eb_init_insn);
                bld.emit(addr, Insn::ArithK(ArithOp::Add, Builder::RSP, eb_offset));
                bld.emit(addr, Insn::StoreMem { addr, value: eb });
            }
        }
    }

    Ok(())
}

/// Determine whether a "hidden argument" is passed to a function with return
/// type identified by `ret_tyid`, and if so, allocate it into `state` (so that
/// the rest of the parameter passing can be "decoded" transparently, with the
/// hidden argument already accounted for).
///
/// # Context
///
/// This function helps this code follow rule #2 from the following excerpt of
/// "System V ABI 1.0 - AMD64 Supplement" (page 28):
///
/// > The returning of values is done according to the following algorithm:
/// >
/// > 1. Classify the return type with the classification algorithm.
/// >
/// > 2. If the type has class MEMORY, then the caller provides space for the
/// > return value and passes the address of this storage in %rdi as if it were
/// > the first argument to the function. In effect, this address becomes a
/// > “hidden” first argument. This storage must not overlap any data visible to
/// > the callee through other names than this argument.
/// >
/// > On return %rax will contain the address that has been passed in by the caller in %rdi.
/// >
/// > [... continues with more rules, but those aren't relevant for this function...]
fn prepare_for_return_value(
    bld: &mut Builder,
    ret_tyid: ty::TypeID,
    state: &mut ParamPassing,
) -> anyhow::Result<()> {
    // We need to check whether the return type is allocated to memory or to a
    // register. In case it's memory (stack space, typically), the address is
    // going to be passed in as RDI, so we have to skip *that* for parameters.
    match &*bld.types.get(ret_tyid).unwrap() {
        ty::Ty::Void => {
            // we're fine: no storage is used for this value, so `state` is already OK
        }
        ret_ty @ (ty::Ty::Bool(_) | ty::Ty::Subroutine(_)) => {
            return Err(anyhow::anyhow!(
                "invalid type for a function return value: {:?}",
                ret_ty
            ));
        }
        ty::Ty::Unknown => {
            // nothing better to do in this case...
            return Err(anyhow::anyhow!(
                "unknown return type (remaining parameters can't be mapped)"
            ));
        }
        _ => {
            let mut eb_set = EightbytesSet::new_regs();
            classify_eightbytes(&mut eb_set, &bld.types, ret_tyid, 0)?;
            // pass a copy of state: in this step, we just want to predict the
            // outcome of pack_return_value, we don't want to actually pull
            // regs yet
            let pass_mode = eightbytes_to_pass_mode(eb_set, &mut state.clone());
            match pass_mode {
                PassMode::OneBigSse { .. } | PassMode::Regs(_) => {}
                PassMode::Memory => {
                    // ok, *now* we know that the hidden argument is there
                    let _ = state.pull_integer_reg().unwrap();
                }
            }
        }
    }

    Ok(())
}

#[instrument(skip_all)]
pub fn pack_return_value(bld: &mut Builder, ret_tyid: ty::TypeID) -> anyhow::Result<mil::Reg> {
    let ret_ty = &*bld.types.get(ret_tyid).unwrap();
    let ret_val = bld.tmp_gen();
    match &*ret_ty {
        ty::Ty::Void => {
            bld.emit(ret_val, Insn::Void);
        }
        ty::Ty::Unknown => {
            // nothing better to do in this case...
            bld.emit(ret_val, Insn::UndefinedBytes { size: 0 });
        }
        ty::Ty::Bool(_) | ty::Ty::Subroutine(_) => {
            panic!("invalid type for a function return value: {:?}", ret_ty);
        }
        _ => {
            let mut eb_set = EightbytesSet::new_regs();
            classify_eightbytes(&mut eb_set, &bld.types, ret_tyid, 0)?;
            // for return values, no more than 2 registers should be used; if the type
            // is larger than that, it goes to memory
            let eb_set = eb_set.limit_regs(2);

            // .unwrap(): classify_eightbytes already checks that size is known
            let sz = bld.types.bytes_size(ret_tyid).unwrap();
            assert!(sz > 0);

            bld.emit(ret_val, Insn::Void);
            let tmp = bld.tmp_gen();

            match eb_set {
                EightbytesSet::Regs { clss } => {
                    // different than those for parameter passing
                    let mut int_regs = [Builder::RAX, Builder::RDX].as_slice().iter();
                    let mut sse_regs = [Builder::ZMM0, Builder::ZMM1].as_slice().iter();

                    // TODO assert that the type is 1-16 bytes long, based on clss
                    let mut clss = clss.used().iter().peekable();
                    while let Some(cls) = clss.next() {
                        let part = match *cls {
                            RegClass::Integer => {
                                *int_regs.next().expect("bug: not enough int regs!")
                            }
                            RegClass::Sse => {
                                let sse_reg = *sse_regs.next().expect("bug: not enough sse regs!");
                                let mut eb_count = 1;
                                while clss.next_if_eq(&&RegClass::SseUp).is_some() {
                                    eb_count += 1;
                                }

                                bld.emit(
                                    tmp,
                                    Insn::Part {
                                        src: sse_reg,
                                        offset: 0,
                                        size: 8 * eb_count,
                                    },
                                );
                                tmp
                            }
                            RegClass::SseUp => unreachable!(),
                            RegClass::Unused => {
                                for &cls in clss {
                                    assert_eq!(cls, RegClass::Unused);
                                }
                                break;
                            }
                        };
                        bld.emit(
                            ret_val,
                            Insn::Concat {
                                lo: ret_val,
                                hi: part,
                            },
                        );
                    }
                }
                EightbytesSet::Memory => {
                    // From the spec: AMD64 ABI Draft 0.99.6 – July 2, 2012 - page 22
                    //
                    // > If the type has class MEMORY, then the caller provides space
                    // > for the return value and passes the address of this storage
                    // > in %rdi as if it were the first argument to the function. In
                    // > effect, this address becomes a “hidden” first argument. This
                    // > storage must not overlap any data visible to the callee through
                    // > other names than this argument.
                    // >
                    // > On return %rax will contain the address that has been passed in
                    // > by the caller in %rdi.

                    let eb_count = sz.div_ceil(8);

                    let addr = bld.tmp_gen();
                    for eb_ndx in 0..eb_count {
                        let eb_offset = (8 * eb_ndx).try_into().unwrap();
                        bld.emit(addr, Insn::ArithK(ArithOp::Add, Builder::RAX, eb_offset));
                        let eb = bld.tmp_gen();
                        bld.emit(eb, Insn::LoadMem { addr, size: 8 });
                        bld.emit(
                            ret_val,
                            Insn::Concat {
                                lo: ret_val,
                                hi: eb,
                            },
                        );
                    }
                }
            };

            // arguments are rounded up to multiples of 8 bytes, and assembly code out
            // there tends to work on 8 bytes registers; but it's useful to show the
            // 'small integer' version of the value
            bld.emit(
                ret_val,
                Insn::Part {
                    src: ret_val,
                    offset: 0,
                    size: sz.try_into().unwrap(),
                },
            );
        }
    }

    Ok(ret_val)
}

#[instrument(skip_all)]
pub fn pack_params(
    bld: &mut Builder,
    subr_tyid: ty::TypeID,
) -> anyhow::Result<(Report, Vec<mil::Reg>)> {
    let Result::Ok(subr_ty) = super::check_subroutine_type(bld.types, subr_tyid) else {
        return Err(anyhow::anyhow!("could not narrow down to subroutine type"));
    };
    let param_types = &subr_ty.param_tyids;
    let ret_tyid = subr_ty.return_tyid;

    let mut report = Report { ok_count: 0 };
    let mut state = ParamPassing::default();
    prepare_for_return_value(bld, ret_tyid, &mut state)?;

    let param_count = param_types.len();
    let mut param_regs: Vec<mil::Reg> = Vec::with_capacity(param_count);
    for (ndx, &param_tyid) in param_types.iter().enumerate() {
        let param_ty = bld.types.get(param_tyid).unwrap();

        let span = span!(Level::INFO, "param", ndx, tyid=?param_tyid, ty=?param_ty);
        let _enter = span.enter();

        match pack_param(bld, &mut state, param_tyid) {
            Result::Ok(param_dest) => {
                param_regs.push(param_dest);
                report.ok_count += 1;
            }
            Result::Err(err) => {
                event!(
                    Level::ERROR,
                    ok_count = param_regs.len(),
                    total_count = param_types.len(),
                    ?err,
                    "parameter type could not be classified (remaining parameters can't be mapped)"
                );
                break;
            }
        }
    }

    assert_eq!(param_regs.len(), report.ok_count);
    Ok((report, param_regs))
}

fn pack_param(
    bld: &mut Builder,
    state: &mut ParamPassing,
    tyid: ty::TypeID,
) -> anyhow::Result<mil::Reg> {
    let mut eb_set = EightbytesSet::new_regs();
    classify_eightbytes(&mut eb_set, bld.types, tyid, 0)?;
    let mode = eightbytes_to_pass_mode(eb_set, state);
    let arg_value = bld.tmp_gen();

    let sz = bld
        .types
        .bytes_size(tyid)
        .ok_or_else(|| anyhow!("type has no size?"))?;
    let eb_count = sz.div_ceil(8);

    bld.emit(arg_value, Insn::Void);

    match mode {
        PassMode::Regs(regs) => {
            assert!(regs[0].is_some());

            // TODO assert that the type is 1-16 bytes long, based on clss
            let mut regs_eb_count = 0;
            for (ndx, reg) in regs.into_iter().enumerate() {
                let Some(reg) = reg else {
                    continue;
                };

                let eb_value = bld.emit_read_machine_reg(reg);
                regs_eb_count += 1;

                if ndx > 0 {
                    // Combine with previous parts
                    bld.emit(
                        arg_value,
                        mil::Insn::Concat {
                            lo: arg_value,
                            hi: eb_value,
                        },
                    );
                } else {
                    bld.emit(arg_value, mil::Insn::Get(eb_value));
                }
            }

            assert_eq!(regs_eb_count, eb_count);
        }
        PassMode::OneBigSse { reg, eb_count } => {
            assert_eq!(sz.div_ceil(8), eb_count as usize);
            let sse_value = bld.emit_read_machine_reg(reg);
            bld.emit(arg_value, Insn::Get(sse_value));
        }
        PassMode::Memory => {
            assert!(eb_count > 0);
            assert!(sz > 0);

            // read all eightbytes in a single 'operation'
            let addr = bld.tmp_gen();
            let eb_offset = state.pull_stack_eightbyte(eb_count) as i64;
            bld.emit(addr, Insn::ArithK(ArithOp::Add, Builder::RSP, eb_offset));
            bld.emit(
                arg_value,
                Insn::LoadMem {
                    addr,
                    size: (8 * eb_count).try_into().unwrap(),
                },
            );
        }
    }

    // Trim to the correct size if needed
    if sz < (8 * eb_count) {
        bld.emit(
            arg_value,
            mil::Insn::Part {
                src: arg_value,
                offset: 0,
                size: sz.try_into().unwrap(),
            },
        );
    }

    let index = bld.last_index_of_value(arg_value).unwrap();
    bld.set_value_type(index, tyid);

    Ok(arg_value)
}

pub fn unpack_return_value(
    bld: &mut Builder,
    ret_tyid: ty::TypeID,
    ret_val: mil::Reg,
) -> anyhow::Result<()> {
    match &*bld.types.get(ret_tyid).unwrap() {
        // no register changed as a result of a call
        ty::Ty::Void => Ok(()),
        ty::Ty::Unknown => {
            // don't know anything better that could be done in this case...
            for mreg in [Builder::RAX, Builder::RDX] {
                bld.emit(mreg, Insn::UndefinedBytes { size: 8 });
            }
            for mreg in [Builder::ZMM0, Builder::ZMM1] {
                bld.emit(mreg, Insn::UndefinedBytes { size: 64 });
            }
            Ok(())
        }
        ret_ty @ (ty::Ty::Bool(_) | ty::Ty::Subroutine(_)) => {
            panic!("invalid type for a function return value: {:?}", ret_ty);
        }
        _ => {
            let mut eb_set = EightbytesSet::new_regs();
            classify_eightbytes(&mut eb_set, bld.types, ret_tyid, 0)?;
            // for return values, no more than 2 registers should be used; if the type
            // is larger than that, it goes to memory
            let eb_set = eb_set.limit_regs(2);

            // .unwrap(): classify_eightbytes already checks that size is known
            let sz = bld.types.bytes_size(ret_tyid).unwrap();
            assert!(sz > 0);

            match eb_set {
                EightbytesSet::Regs { clss } => {
                    // different than those for parameter passing
                    let mut int_regs = [Builder::RAX, Builder::RDX].as_slice().iter();
                    let mut sse_regs = [Builder::ZMM0, Builder::ZMM1].as_slice().iter();

                    // TODO assert that the type is 1-16 bytes long, based on clss
                    let mut clss = clss.used().iter().peekable();
                    let mut eb_dest_ofs = 0;
                    while let Some(cls) = clss.next() {
                        let (eb_dest, eb_count) = match *cls {
                            RegClass::Integer => {
                                let eb_dest = *int_regs.next().expect("bug: not enough int regs!");
                                (eb_dest, 1)
                            }
                            RegClass::Sse => {
                                let sse_reg = *sse_regs.next().expect("bug: not enough sse regs!");
                                let mut eb_count = 1;
                                while let Some(RegClass::SseUp) = clss.peek() {
                                    eb_count += 1;
                                }
                                (sse_reg, eb_count)
                            }
                            RegClass::SseUp => unreachable!(),
                            RegClass::Unused => {
                                for &cls in clss {
                                    assert_eq!(cls, RegClass::Unused);
                                }
                                break;
                            }
                        };
                        bld.emit(
                            eb_dest,
                            Insn::Part {
                                src: ret_val,
                                offset: eb_dest_ofs,
                                size: eb_count * 8,
                            },
                        );

                        eb_dest_ofs += eb_count * 8;
                    }

                    Ok(())
                }
                EightbytesSet::Memory => {
                    // From the spec: AMD64 ABI Draft 0.99.6 – July 2, 2012 - page 22
                    //
                    // > If the type has class MEMORY, then the caller provides space
                    // > for the return value and passes the address of this storage
                    // > in %rdi as if it were the first argument to the function. In
                    // > effect, this address becomes a “hidden” first argument. This
                    // > storage must not overlap any data visible to the callee through
                    // > other names than this argument.
                    // >
                    // > On return %rax will contain the address that has been passed in
                    // > by the caller in %rdi.

                    let eb_count = sz.div_ceil(8);
                    let tmp = bld.tmp_gen();
                    let ret_val = if sz < eb_count * 8 {
                        bld.emit(
                            tmp,
                            Insn::Widen {
                                reg: ret_val,
                                target_size: (eb_count * 8).try_into().unwrap(),
                                sign: false,
                            },
                        );
                        tmp
                    } else {
                        ret_val
                    };

                    bld.emit(
                        tmp,
                        Insn::StoreMem {
                            addr: Builder::RAX,
                            value: ret_val,
                        },
                    );

                    Ok(())
                }
            }
        }
    }
}

fn eightbytes_to_pass_mode(eb_set: EightbytesSet, state_saved: &mut ParamPassing) -> PassMode {
    // we manipulate state and try to get enough registers as needed. if we
    // can't we ought to use PassMode::Memory. we save the resulting state back
    // to state_saved only if we manage to allocate all registers
    let mut state = state_saved.clone();

    // as per spec: AMD64 ABI Draft 0.99.6 – July 2, 2012 - page 19
    // > If the size of the aggregate exceeds two eightbytes and the first
    // > eightbyte isn’t SSE or any other eightbyte isn’t SSEUP, the whole
    // > argument is passed in memory.
    let pass_mode = match eb_set {
        EightbytesSet::Regs { clss } => {
            let clss = clss.used();
            assert!(clss.iter().all(|cls| *cls != RegClass::Unused));
            assert!(!clss.is_empty());

            // [Sse, SseUp...]
            if clss[0] == RegClass::Sse && clss[1..].iter().all(|&cls| cls == RegClass::SseUp) {
                let Some(reg) = state.pull_sse_reg() else {
                    return PassMode::Memory;
                };
                PassMode::OneBigSse {
                    reg,
                    eb_count: clss.len().try_into().unwrap(),
                }
            } else if clss.len() <= 2 {
                let mut regs = [None; 2];

                for (slot, cls) in regs.iter_mut().zip(clss) {
                    let reg = match cls {
                        RegClass::Integer => state.pull_integer_reg(),
                        RegClass::Sse => state.pull_sse_reg(),
                        RegClass::SseUp => panic!("SseUp invalid in this position"),
                        RegClass::Unused => unreachable!(),
                    };
                    let Some(reg) = reg else {
                        return PassMode::Memory;
                    };
                    *slot = Some(reg);
                }

                PassMode::Regs(regs)
            } else {
                PassMode::Memory
            }
        }
        EightbytesSet::Memory => PassMode::Memory,
    };

    *state_saved = state;
    pass_mode
}

fn eightbytes_range(offset: usize, size: usize) -> (u8, u8) {
    let eb_first_ndx = offset / 8;
    let eb_last_ndx = (offset + size - 1) / 8;
    (
        eb_first_ndx.try_into().unwrap(),
        eb_last_ndx.try_into().unwrap(),
    )
}

fn classify_eightbytes(
    eb_set: &mut EightbytesSet,
    types: &ty::TypeSet,
    tyid: ty::TypeID,
    offset: usize,
) -> anyhow::Result<()> {
    let ty = types.get(tyid).unwrap();

    if let ty::Ty::Alias(ref_tyid) = &*ty {
        return classify_eightbytes(eb_set, types, *ref_tyid, offset);
    }

    let sz = types
        .bytes_size(tyid)
        .ok_or_else(|| anyhow!("type has no size?"))?;
    let alignment: usize = types
        .alignment(tyid)
        .ok_or_else(|| anyhow!("type has no alignment?"))?
        .into();
    if (offset % alignment) != 0 {
        // unaligned member of a struct
        // (things that aren't struct members have offset == 0)
        *eb_set = EightbytesSet::Memory;
        return Ok(());
    }

    match &*ty {
        ty::Ty::Int(_) | ty::Ty::Bool(_) | ty::Ty::Ptr(_) | ty::Ty::Enum(_) => {
            let (eb_first_ndx, eb_last_ndx) = eightbytes_range(offset, sz);
            // should only fail for unaligned types, which we've already excluded
            assert_eq!(eb_first_ndx, eb_last_ndx);
            eb_set.merge(eb_first_ndx, RegClass::Integer);
        }

        ty::Ty::Float(_) => {
            let (eb_first_ndx, eb_last_ndx) = eightbytes_range(offset, sz);
            eb_set.merge(eb_first_ndx, RegClass::Sse);
            for eb_ndx in (eb_first_ndx + 1)..=eb_last_ndx {
                eb_set.merge(eb_ndx, RegClass::SseUp);
            }
        }

        ty::Ty::Struct(struct_ty) => {
            for memb in struct_ty.members.iter() {
                classify_eightbytes(eb_set, types, memb.tyid, memb.offset)?;
                if eb_set == &EightbytesSet::Memory {
                    return Ok(());
                }
            }
        }
        ty::Ty::Array(array_ty) => {
            if let ty::Subrange {
                lo: 0,
                hi: Some(count),
            } = array_ty.index_subrange
            {
                if count < 0 {
                    // TODO nicer way to exclude this case altogether?
                    return Err(anyhow!("invalid array index subrange (count < 0)"));
                }
                let count: usize = count.try_into().unwrap();

                let element_size = types
                    .bytes_size(array_ty.element_tyid)
                    .ok_or_else(|| anyhow!("array element type has no size"))?;
                for i in 0..count {
                    classify_eightbytes(eb_set, types, array_ty.element_tyid, i * element_size)?;
                    if eb_set == &EightbytesSet::Memory {
                        return Ok(());
                    }
                }
            } else {
                // C/C++ don't do this; not sure how to handle parameter passing
                // in this case
                return Err(anyhow!(
                    "array parameter has indices not in 0..N range; discarding"
                ));
            }
        }

        ty::Ty::Subroutine(_) | ty::Ty::Unknown | ty::Ty::Void | ty::Ty::Flag => {
            panic!("invalid type for a function param: {:?}", ty)
        }
        ty::Ty::Alias(_) => unreachable!(),
    }
    Ok(())
}

#[derive(PartialEq, Eq, Debug)]
enum EightbytesSet {
    Memory,
    Regs { clss: Classes },
}
#[derive(PartialEq, Eq, Debug)]
struct Classes([RegClass; 4]);

impl EightbytesSet {
    fn new_regs() -> EightbytesSet {
        EightbytesSet::Regs {
            clss: Classes([RegClass::Unused; 4]),
        }
    }

    fn limit_regs(self, count: usize) -> Self {
        match self {
            EightbytesSet::Memory => EightbytesSet::Memory,
            EightbytesSet::Regs {
                clss: Classes(clss),
            } => {
                for slot in &clss[count..] {
                    if *slot != RegClass::Unused {
                        return EightbytesSet::Memory;
                    }
                }

                self
            }
        }
    }

    fn merge(&mut self, ndx: u8, other: RegClass) {
        // we can't quite be sure from the beginning of how many eightbytes
        // are going to be used for a struct, because we could have up to
        // 4 eightbytes in "one big ssa reg" (Sse, SseUp, ...), or only up to 2
        // for an Integer/Sse mix.
        //
        // but, if we learn that a type requires more than 4 eightbytes, there
        // is no more ambiguity, and no need to track anything. we'd do this
        // separately in the conversion to PassMode, but this simplifies a few
        // things
        if ndx >= 4 {
            *self = EightbytesSet::Memory;
            return;
        }

        match self {
            EightbytesSet::Memory => {}
            EightbytesSet::Regs {
                clss: Classes(clss),
            } => {
                let cls = &mut clss[ndx as usize];
                *cls = match (*cls, other) {
                    (_, RegClass::Unused) => panic!("invalid merged ScalarClass: Unused"),
                    (RegClass::Unused, other) => other,

                    (RegClass::Integer, _) | (_, RegClass::Integer) => RegClass::Integer,

                    (RegClass::Sse, RegClass::Sse) => RegClass::Sse,
                    (RegClass::SseUp, RegClass::SseUp) => RegClass::SseUp,

                    (RegClass::Sse, RegClass::SseUp) | (RegClass::SseUp, RegClass::Sse) => {
                        panic!("bug: invalid merge: {:?} <- {:?}", *cls, other)
                    }
                };
            }
        };
    }
}

impl Classes {
    fn used(&self) -> &[RegClass] {
        // make sure Unused are all at the end of the slice
        // TODO replace with newtype?
        let Classes(clss) = self;
        let unused_count = clss
            .iter()
            .rev()
            .take_while(|&&cls| cls == RegClass::Unused)
            .count();
        let used_count = clss.len() - unused_count;
        &clss[0..used_count]
    }
}

static INTEGER_REGS: [iced_x86::Register; 6] = [
    iced_x86::Register::RDI,
    iced_x86::Register::RSI,
    iced_x86::Register::RDX,
    iced_x86::Register::RCX,
    iced_x86::Register::R8,
    iced_x86::Register::R9,
];

static SSE_REGS: [iced_x86::Register; 16] = [
    iced_x86::Register::ZMM0,
    iced_x86::Register::ZMM1,
    iced_x86::Register::ZMM2,
    iced_x86::Register::ZMM3,
    iced_x86::Register::ZMM4,
    iced_x86::Register::ZMM5,
    iced_x86::Register::ZMM6,
    iced_x86::Register::ZMM7,
    iced_x86::Register::ZMM8,
    iced_x86::Register::ZMM9,
    iced_x86::Register::ZMM10,
    iced_x86::Register::ZMM11,
    iced_x86::Register::ZMM12,
    iced_x86::Register::ZMM13,
    iced_x86::Register::ZMM14,
    iced_x86::Register::ZMM15,
];

#[derive(Clone)]
struct ParamPassing {
    int_regs: std::slice::Iter<'static, iced_x86::Register>,
    sse_regs: std::slice::Iter<'static, iced_x86::Register>,
    args_processed_count: u16,
    stack_eb_ndx: usize,
}
impl ParamPassing {
    fn pull_integer_reg(&mut self) -> Option<iced_x86::Register> {
        self.int_regs.next().copied()
    }

    fn pull_sse_reg(&mut self) -> Option<iced_x86::Register> {
        self.sse_regs.next().copied()
    }

    fn pull_arg(&mut self) -> u16 {
        let arg_ndx = self.args_processed_count;
        self.args_processed_count += 1;
        arg_ndx
    }

    /// Get the "current" offset into the stack, and advance it `eb_count`
    /// eightbytes forward.
    ///
    /// The first value returned is 8, as `dword ptr [rsp]` is the return
    /// address (and no parameter is stored there).
    ///
    /// Returns: the stack offset, expressed in byte offest from the value
    /// of RSP at the beginning of the function. Always aligned to 8.
    fn pull_stack_eightbyte(&mut self, eb_count: usize) -> usize {
        let ofs = self.stack_eb_ndx * 8;
        self.stack_eb_ndx += eb_count;
        ofs
    }
}

impl Default for ParamPassing {
    fn default() -> Self {
        ParamPassing {
            int_regs: INTEGER_REGS.as_slice().iter(),
            sse_regs: SSE_REGS.as_slice().iter(),
            args_processed_count: 0,
            stack_eb_ndx: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegClass {
    Unused,
    Integer,
    Sse,
    SseUp,
}

#[cfg(test)]
mod tests {
    use ty::TypeID;

    use super::*;
    use std::sync::Arc;

    struct Types {
        tyid_i64: ty::TypeID,
        tyid_i32: ty::TypeID,
        tyid_i16: ty::TypeID,
        tyid_i8: ty::TypeID,
        tyid_f32: ty::TypeID,
        tyid_f64: ty::TypeID,
    }

    struct TypeIdGen {
        counter: usize,
    }

    impl TypeIdGen {
        fn new() -> Self {
            TypeIdGen { counter: 0 }
        }

        fn next_id(&mut self) -> TypeID {
            let tyid = TypeID(self.counter);
            self.counter += 1;
            tyid
        }
    }

    // TOOD share the result (e.g. as a OnceCell)
    fn make_scalars(types: &mut ty::TypeSet, tyid_gen: &mut TypeIdGen) -> Types {
        use ty::{Int, Signedness, Ty, TypeSet};

        let mut mk_int = |types: &mut TypeSet, name: &str, size: u8| {
            let tyid = tyid_gen.next_id();
            let ty = Ty::Int(Int {
                size,
                signed: Signedness::Signed,
            });
            types.set(tyid, ty);
            types.set_name(tyid, name.to_owned());
            tyid
        };

        let tyid_i8 = mk_int(types, "i8", 1);
        let tyid_i16 = mk_int(types, "i16", 2);
        let tyid_i32 = mk_int(types, "i32", 4);
        let tyid_i64 = mk_int(types, "i64", 8);

        let tyid_f32 = tyid_gen.next_id();
        types.set(tyid_f32, Ty::Float(ty::Float { size: 4 }));
        types.set_name(tyid_f32, "float32".to_owned());

        let tyid_f64 = tyid_gen.next_id();
        types.set(tyid_f64, Ty::Float(ty::Float { size: 8 }));

        types.set_name(tyid_f64, "float64".to_owned());

        Types {
            tyid_i64,
            tyid_i32,
            tyid_i16,
            tyid_i8,
            tyid_f32,
            tyid_f64,
        }
    }

    #[test]
    fn classify_struct_one_int() {
        let mut types = ty::TypeSet::new();
        let mut tyid_gen = TypeIdGen::new();
        let scas = make_scalars(&mut types, &mut tyid_gen);

        for sca_tyid in [scas.tyid_i8, scas.tyid_i16, scas.tyid_i32, scas.tyid_i64] {
            let struct_tyid = tyid_gen.next_id();
            make_sample_struct(&mut types, struct_tyid, &[(0, sca_tyid)]);

            let mut buf = EightbytesSet::new_regs();
            classify_eightbytes(&mut buf, &types, struct_tyid, 0).unwrap();

            assert_eq!(
                buf,
                EightbytesSet::Regs {
                    clss: Classes([
                        RegClass::Integer,
                        RegClass::Unused,
                        RegClass::Unused,
                        RegClass::Unused,
                    ])
                }
            );
        }
    }

    #[test]
    fn classify_struct_ints_fit_1_eb() {
        let mut types = ty::TypeSet::new();
        let mut tyid_gen = TypeIdGen::new();
        let scas = make_scalars(&mut types, &mut tyid_gen);

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
            let struct_tyid = tyid_gen.next_id();
            make_sample_struct(&mut types, struct_tyid, members);

            let mut buf = EightbytesSet::new_regs();
            classify_eightbytes(&mut buf, &types, struct_tyid, 0).unwrap();

            assert_eq!(
                buf,
                EightbytesSet::Regs {
                    clss: Classes([
                        RegClass::Integer,
                        RegClass::Unused,
                        RegClass::Unused,
                        RegClass::Unused,
                    ])
                }
            );
        }
    }

    #[test]
    fn classify_struct_one_float() {
        let mut types = ty::TypeSet::new();
        let mut tyid_gen = TypeIdGen::new();
        let scas = make_scalars(&mut types, &mut tyid_gen);

        for sca_tyid in [scas.tyid_f32, scas.tyid_f64] {
            let struct_tyid = tyid_gen.next_id();
            make_sample_struct(&mut types, struct_tyid, &[(0, sca_tyid)]);

            let mut buf = EightbytesSet::new_regs();
            classify_eightbytes(&mut buf, &types, struct_tyid, 0).unwrap();

            assert_eq!(
                &buf,
                &EightbytesSet::Regs {
                    clss: Classes([
                        RegClass::Sse,
                        RegClass::Unused,
                        RegClass::Unused,
                        RegClass::Unused,
                    ])
                }
            );
        }
    }

    #[test]
    fn classify_struct_two_ints() {
        let mut types = ty::TypeSet::new();
        let mut tyid_gen = TypeIdGen::new();
        let scas = make_scalars(&mut types, &mut tyid_gen);

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
            let struct_tyid = tyid_gen.next_id();
            make_sample_struct(&mut types, struct_tyid, members);

            let mut buf = EightbytesSet::new_regs();
            classify_eightbytes(&mut buf, &types, struct_tyid, 0).unwrap();

            assert_eq!(
                buf,
                EightbytesSet::Regs {
                    clss: Classes([
                        RegClass::Integer,
                        RegClass::Integer,
                        RegClass::Unused,
                        RegClass::Unused,
                    ])
                }
            );
        }
    }

    #[test]
    fn classify_struct_larger() {
        let mut types = ty::TypeSet::new();
        let mut tyid_gen = TypeIdGen::new();
        let scas = make_scalars(&mut types, &mut tyid_gen);

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
            let struct_tyid = tyid_gen.next_id();
            make_sample_struct(&mut types, struct_tyid, members);
            let mut buf = EightbytesSet::new_regs();
            classify_eightbytes(&mut buf, &types, struct_tyid, 0).unwrap();
            assert_eq!(buf, EightbytesSet::Memory);
        }
    }

    fn make_sample_struct(types: &mut ty::TypeSet, tyid: TypeID, members: &[(usize, TypeID)]) {
        assert!(members.iter().is_sorted_by_key(|(ofs, _)| ofs));

        let struct_sz = {
            let &(ofs, tyid) = members.last().unwrap();
            let sz = types.bytes_size(tyid).unwrap();
            ofs + sz
        };
        let ty_struct = ty::Ty::Struct(ty::Struct {
            size: struct_sz,
            members: members
                .iter()
                .map(|&(offset, tyid)| ty::StructMember {
                    offset,
                    name: Arc::new(format!("x{}", offset)),
                    tyid,
                })
                .collect(),
        });
        types.set(tyid, ty_struct);
        types.set_name(tyid, "SampleStruct".to_owned());
    }
}
