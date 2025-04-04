use crate::{
    mil::{self, AncestralName, ArithOp, Insn},
    ty,
};

use super::Builder;

use anyhow::anyhow;

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
    pub errors: Vec<anyhow::Error>,
}

pub fn read_func_params<'a>(
    bld: &mut Builder<'a>,
    param_types: &[ty::TypeID],
    ret_tyid: ty::TypeID,
) -> anyhow::Result<Report> {
    let mut report = Report {
        ok_count: 0,
        errors: Vec::new(),
    };

    let types = bld
        .types
        .ok_or(anyhow!("No TypeSet passed to the Builder"))?;

    let mut state = ParamPassing::default();

    // We need to check whether the return type is allocated to memory or to a
    // register. In case it's memory (stack space, typically), the address is
    // going to be passed in as RDI, so we have to skip *that* for parameters.
    {
        let ret_ty = &types.get(ret_tyid).unwrap().ty;
        if let ty::Ty::Void | ty::Ty::Bool(_) | ty::Ty::Subroutine(_) = ret_ty {
            panic!("invalid type for a function return value: {:?}", ret_ty);
        }
        if let ty::Ty::Unknown(_) = ret_ty {
            // nothing better to do in this case...
            report.errors.push(anyhow!(
            "unknown return type (can't guarantee mapping of parameters to register/stack slots)"
        ));
            return Ok(report);
        }

        let mut eb_set = EightbytesSet::new_regs();
        classify_eightbytes(&mut eb_set, types, ret_tyid, 0)?;
        // pass a copy of state: we just want to predict the outcome of
        // read_return_value, we don't want to actually pull regs yet
        let pass_mode = eightbytes_to_pass_mode(eb_set, &mut state.clone());
        match pass_mode {
            PassMode::Regs(_) => {}
            // one-big-sse is the same as memory
            PassMode::OneBigSse { .. } | PassMode::Memory => {
                let _ = state.pull_integer_reg().unwrap();
            }
        }
    }

    for &param_tyid in param_types.iter() {
        let param_ty = &types.get(param_tyid).unwrap().ty;

        if let ty::Ty::Void | ty::Ty::Bool(_) | ty::Ty::Subroutine(_) = param_ty {
            panic!("invalid type for a function parameter: {:?}", param_ty);
        }

        let mut eb_set = EightbytesSet::new_regs();
        let res = classify_eightbytes(&mut eb_set, types, param_tyid, 0);
        if let Err(err) = res {
            report.errors.push(err);
            // because each argument uses a variable number of integer regs, ssa
            // regs, and stack slots, we can't be sure of how to map ANY of the
            // remaining parameters
            break;
        }

        let pass_mode = eightbytes_to_pass_mode(eb_set, &mut state);
        let sz = types.bytes_size(param_tyid).unwrap();

        let param_anc = state
            .pull_arg()
            .ok_or_else(|| anyhow!("not enough arg ancestrals!"))?;
        let param_src = bld.reg_gen.next();
        bld.init_ancestral(param_src, param_anc, mil::RegType::Bytes(sz as usize));

        pass_param(bld, &mut state, types, param_tyid, param_src, pass_mode);

        report.ok_count += 1;
    }

    Ok(report)
}

fn pass_param<'a>(
    bld: &mut Builder<'a>,
    state: &mut ParamPassing,
    types: &ty::TypeSet,
    tyid: ty::TypeID,
    arg_value: mil::Reg,
    mode: PassMode,
) {
    let sz = types.bytes_size(tyid).unwrap();

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
                let arg_value = if sz < 8 {
                    let v0 = bld.reg_gen.next();
                    bld.emit(
                        v0,
                        mil::Insn::Widen {
                            reg: arg_value,
                            target_size: 8,
                            sign: false,
                        },
                    );
                    v0
                } else if sz > 8 {
                    let v0 = bld.reg_gen.next();
                    bld.emit(
                        v0,
                        mil::Insn::Part {
                            src: arg_value,
                            offset,
                            size: 8,
                        },
                    );
                    v0
                } else {
                    arg_value
                };

                bld.emit_write_machine_reg(reg, 8, arg_value);
            }
        }
        PassMode::OneBigSse { reg, eb_count } => {
            assert_eq!(sz.div_ceil(8), eb_count as u32);
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

            let addr = bld.reg_gen.next();
            let eb = bld.reg_gen.next();

            for eb_ndx in 0..eb_count {
                // relies on RSP never being assigned by any instruction emitted in this module
                let eb_offset = state.pull_stack_eightbyte() as i64;
                let offset = (eb_ndx * 8).try_into().unwrap();
                bld.emit(
                    eb,
                    Insn::Part {
                        src: arg_value,
                        offset,
                        size: (sz - offset).min(8),
                    },
                );
                bld.emit(addr, Insn::ArithK(ArithOp::Add, Builder::RSP, eb_offset));
                bld.emit(addr, Insn::StoreMem(addr, eb));
            }
        }
    }
}

pub fn read_return_value<'a>(
    bld: &mut Builder<'a>,
    ret_tyid: ty::TypeID,
) -> anyhow::Result<mil::Reg> {
    let types = bld
        .types
        .ok_or(anyhow!("No TypeSet passed to the Builder"))?;

    let ret_ty = &types.get(ret_tyid).unwrap().ty;

    if let ty::Ty::Void | ty::Ty::Bool(_) | ty::Ty::Subroutine(_) = ret_ty {
        panic!("invalid type for a function return value: {:?}", ret_ty);
    }
    if let ty::Ty::Unknown(_) = ret_ty {
        // nothing better to do in this case...
        let ret_val = bld.reg_gen.next();
        bld.emit(ret_val, Insn::Undefined);
        return Ok(ret_val);
    }

    let mut eb_set = EightbytesSet::new_regs();
    classify_eightbytes(&mut eb_set, types, ret_tyid, 0)?;
    // for return values, no more than 2 registers should be used; if the type
    // is larger than that, it goes to memory
    let eb_set = eb_set.limit_regs(2);

    let sz = types.bytes_size(ret_tyid).unwrap();

    let ret_val = bld.reg_gen.next();
    bld.emit(ret_val, Insn::Void);
    let tmp = bld.reg_gen.next();

    match eb_set {
        EightbytesSet::Regs { clss } => {
            // different than those for parameter passing
            let mut int_regs = [Builder::RAX, Builder::RDX].as_slice().into_iter();
            let mut sse_regs = [Builder::ZMM0, Builder::ZMM1].as_slice().into_iter();

            // TODO assert that the type is 1-16 bytes long, based on clss
            let mut clss = clss.used().iter().peekable();
            while let Some(cls) = clss.next() {
                let part = match *cls {
                    RegClass::Integer => *int_regs.next().expect("bug: not enough int regs!"),
                    RegClass::Sse => {
                        let sse_reg = *sse_regs.next().expect("bug: not enough sse regs!");
                        let mut eb_count = 1;
                        while let Some(RegClass::SseUp) = clss.peek() {
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
                        while let Some(cls) = clss.next() {
                            assert_eq!(*cls, RegClass::Unused);
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

            let addr = bld.reg_gen.next();
            for eb_ndx in 0..eb_count {
                let eb_offset = (8 * eb_ndx).try_into().unwrap();
                bld.emit(addr, Insn::ArithK(ArithOp::Add, Builder::RAX, eb_offset));
                let eb = bld.reg_gen.next();
                bld.emit(eb, Insn::LoadMem { reg: addr, size: 8 });
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
    Ok(ret_val)
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
            assert!(clss.len() >= 1);

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

fn eightbytes_range(offset: u32, size: u32) -> (u8, u8) {
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
    offset: u32,
) -> anyhow::Result<()> {
    let ty = &types.get(tyid).unwrap().ty;

    if let ty::Ty::Alias(ref_tyid) = ty {
        return classify_eightbytes(eb_set, types, *ref_tyid, offset);
    }

    let sz: u32 = types
        .bytes_size(tyid)
        .ok_or_else(|| anyhow!("type has no size?"))?
        .try_into()
        .unwrap();
    let alignment: u32 = types
        .alignment(tyid)
        .ok_or_else(|| anyhow!("type has no alignment?"))?
        .into();
    if (offset % alignment) != 0 {
        // unaligned member of a struct
        // (things that aren't struct members have offset == 0)
        *eb_set = EightbytesSet::Memory;
        return Ok(());
    }

    match ty {
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
                eb_set.merge(eb_ndx.try_into().unwrap(), RegClass::SseUp);
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

        ty::Ty::Subroutine(_) | ty::Ty::Unknown(_) | ty::Ty::Void => {
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
                return self;
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
    args: std::slice::Iter<'static, AncestralName>,
    stack_eb_ndx: usize,
}
impl ParamPassing {
    fn pull_integer_reg(&mut self) -> Option<iced_x86::Register> {
        self.int_regs.next().copied()
    }

    fn pull_sse_reg(&mut self) -> Option<iced_x86::Register> {
        self.sse_regs.next().copied()
    }

    fn pull_arg(&mut self) -> Option<AncestralName> {
        self.args.next().copied()
    }

    /// Get the "current" offset into the stack, and advance it of one eightbyte.
    ///
    /// The first value returned is 8, as `dword ptr [rsp]` is the return
    /// address (and no parameter is stored there).
    ///
    /// Returns: the stack offset, expressed in byte offest from the value
    /// of RSP at the beginning of the function. Always aligned to 8.
    fn pull_stack_eightbyte(&mut self) -> usize {
        let ofs = self.stack_eb_ndx * 8;
        self.stack_eb_ndx += 1;
        ofs
    }
}

impl Default for ParamPassing {
    fn default() -> Self {
        ParamPassing {
            int_regs: INTEGER_REGS.as_slice().into_iter(),
            sse_regs: SSE_REGS.as_slice().into_iter(),
            args: super::ANC_ARGS.as_slice().into_iter(),
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
    use insta::assert_snapshot;
    use ty::TypeID;

    use crate::ssa;

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
        bld.use_types(
            &types.types,
            ty::Subroutine {
                return_tyid: types.tyid_void,
                param_names: vec![None; param_types.len()],
                param_tyids: param_types.to_vec(),
            },
        )
        .unwrap();

        read_func_params(&mut bld, param_types, types.tyid_void).unwrap();

        let v0 = bld.reg_gen.next();
        bld.emit(v0, Insn::Ret(Builder::RDI));

        let prog = bld.build();
        let prog = ssa::mil_to_ssa(ssa::ConversionParams { program: prog });
        let snap = format!("params: {:?}\nprogram:\n{:?}", param_types, prog);
        snap
    }

    struct Types {
        types: ty::TypeSet,
        tyid_void: ty::TypeID,
        tyid_i64: ty::TypeID,
        tyid_i32: ty::TypeID,
        tyid_i16: ty::TypeID,
        tyid_i8: ty::TypeID,
        tyid_f32: ty::TypeID,
        tyid_f64: ty::TypeID,
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

        let tyid_void = types.add(Type {
            name: Arc::new("void".to_owned()),
            ty: Ty::Void,
        });
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

        Types {
            types,
            tyid_void,
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
        let mut scas = make_scalars();

        for sca_tyid in [scas.tyid_i8, scas.tyid_i16, scas.tyid_i32, scas.tyid_i64] {
            let struct_tyid = make_sample_struct(&mut scas.types, &[(0, sca_tyid)]);

            let mut buf = EightbytesSet::new_regs();
            classify_eightbytes(&mut buf, &scas.types, struct_tyid, 0).unwrap();

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

            let mut buf = EightbytesSet::new_regs();
            classify_eightbytes(&mut buf, &scas.types, struct_tyid, 0).unwrap();

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
        let mut scas = make_scalars();

        for sca_tyid in [scas.tyid_f32, scas.tyid_f64] {
            let struct_tyid = make_sample_struct(&mut scas.types, &[(0, sca_tyid)]);

            let mut buf = EightbytesSet::new_regs();
            classify_eightbytes(&mut buf, &scas.types, struct_tyid, 0).unwrap();

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

            let mut buf = EightbytesSet::new_regs();
            classify_eightbytes(&mut buf, &scas.types, struct_tyid, 0).unwrap();

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
            let mut buf = EightbytesSet::new_regs();
            classify_eightbytes(&mut buf, &scas.types, struct_tyid, 0).unwrap();
            assert_eq!(buf, EightbytesSet::Memory);
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
