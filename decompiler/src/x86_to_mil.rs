use std::ops::DerefMut;
use std::sync::{Arc, RwLock};

use crate::mil::{self, AncestralName, Control, RegType};
use crate::ty;
use crate::util::{ToWarnings, Warnings};
use iced_x86::{Formatter, IntelFormatter};
use iced_x86::{OpKind, Register};

use anyhow::{Context as _, Result};
use tracing::{event, span, Level};

pub mod callconv;

pub struct Builder {
    pb: mil::ProgramBuilder,
}

impl Builder {
    pub fn new(
        // this may become a simple &ty::TypeSet and be passed directly to
        // Self::translate
        types: Arc<RwLock<ty::TypeSet>>,
    ) -> Self {
        let mut bld = Builder {
            pb: mil::ProgramBuilder::new(Self::R_TMP_FIRST, types),
        };

        bld.init_ancestral(Self::RSP, mil::ANC_STACK_BOTTOM, RegType::Bytes(8));

        // ensure all registers are initialized at least once. most of these
        // instructions (if all goes well, all of them) get "deleted" (masked)
        // if the program is valid and the decompilation correct.  if not, this
        // allows the program to still be decompiled into something (albeit,
        // with some "holes")

        bld.init_ancestral(Self::CF, ANC_CF, RegType::Bool);
        bld.init_ancestral(Self::PF, ANC_PF, RegType::Bool);
        bld.init_ancestral(Self::AF, ANC_AF, RegType::Bool);
        bld.init_ancestral(Self::ZF, ANC_ZF, RegType::Bool);
        bld.init_ancestral(Self::SF, ANC_SF, RegType::Bool);
        bld.init_ancestral(Self::TF, ANC_TF, RegType::Bool);
        bld.init_ancestral(Self::IF, ANC_IF, RegType::Bool);
        bld.init_ancestral(Self::DF, ANC_DF, RegType::Bool);
        bld.init_ancestral(Self::OF, ANC_OF, RegType::Bool);
        bld.init_ancestral(Self::RBP, ANC_RBP, RegType::Bytes(8));
        bld.init_ancestral(Self::RSP, ANC_RSP, RegType::Bytes(8));
        bld.init_ancestral(Self::RIP, ANC_RIP, RegType::Bytes(8));
        bld.init_ancestral(Self::RDI, ANC_RDI, RegType::Bytes(8));
        bld.init_ancestral(Self::RSI, ANC_RSI, RegType::Bytes(8));
        bld.init_ancestral(Self::RAX, ANC_RAX, RegType::Bytes(8));
        bld.init_ancestral(Self::RBX, ANC_RBX, RegType::Bytes(8));
        bld.init_ancestral(Self::RCX, ANC_RCX, RegType::Bytes(8));
        bld.init_ancestral(Self::RDX, ANC_RDX, RegType::Bytes(8));
        bld.init_ancestral(Self::R8, ANC_R8, RegType::Bytes(8));
        bld.init_ancestral(Self::R9, ANC_R9, RegType::Bytes(8));
        bld.init_ancestral(Self::R10, ANC_R10, RegType::Bytes(8));
        bld.init_ancestral(Self::R11, ANC_R11, RegType::Bytes(8));
        bld.init_ancestral(Self::R12, ANC_R12, RegType::Bytes(8));
        bld.init_ancestral(Self::R13, ANC_R13, RegType::Bytes(8));
        bld.init_ancestral(Self::R14, ANC_R14, RegType::Bytes(8));
        bld.init_ancestral(Self::R15, ANC_R15, RegType::Bytes(8));
        bld.init_ancestral(Self::ZMM0, ANC_ZMM0, RegType::Bytes(64));
        bld.init_ancestral(Self::ZMM1, ANC_ZMM1, RegType::Bytes(64));
        bld.init_ancestral(Self::ZMM2, ANC_ZMM2, RegType::Bytes(64));
        bld.init_ancestral(Self::ZMM3, ANC_ZMM3, RegType::Bytes(64));
        bld.init_ancestral(Self::ZMM4, ANC_ZMM4, RegType::Bytes(64));
        bld.init_ancestral(Self::ZMM5, ANC_ZMM5, RegType::Bytes(64));
        bld.init_ancestral(Self::ZMM6, ANC_ZMM6, RegType::Bytes(64));
        bld.init_ancestral(Self::ZMM7, ANC_ZMM7, RegType::Bytes(64));
        bld.init_ancestral(Self::ZMM8, ANC_ZMM8, RegType::Bytes(64));
        bld.init_ancestral(Self::ZMM9, ANC_ZMM9, RegType::Bytes(64));
        bld.init_ancestral(Self::ZMM10, ANC_ZMM10, RegType::Bytes(64));
        bld.init_ancestral(Self::ZMM11, ANC_ZMM11, RegType::Bytes(64));
        bld.init_ancestral(Self::ZMM12, ANC_ZMM12, RegType::Bytes(64));
        bld.init_ancestral(Self::ZMM13, ANC_ZMM13, RegType::Bytes(64));
        bld.init_ancestral(Self::ZMM14, ANC_ZMM14, RegType::Bytes(64));
        bld.init_ancestral(Self::ZMM15, ANC_ZMM15, RegType::Bytes(64));

        bld
    }

    pub fn build(self) -> mil::Program {
        self.pb.build()
    }

    fn init_ancestral(&mut self, reg: mil::Reg, anc_name: AncestralName, rt: RegType) {
        self.emit(reg, mil::Insn::Ancestral(anc_name));
        self.pb.set_ancestral_type(anc_name, rt);
    }

    pub fn tmp_gen(&mut self) -> mil::Reg {
        self.pb.tmp_gen()
    }

    pub fn translate(
        mut self,
        insns: impl Iterator<Item = iced_x86::Instruction>,
        func_ty: Option<&ty::Subroutine>,
    ) -> Result<(mil::Program, Warnings)> {
        use iced_x86::{OpKind, Register};

        let types = Arc::clone(&self.pb.types());
        let mut types = types.write().unwrap();
        let types = types.deref_mut();
        let mut formatter = IntelFormatter::new();
        let mut warnings = Warnings::default();

        if let Some(func_ty) = func_ty {
            let param_count = func_ty.param_tyids.len();
            let res = callconv::unpack_params(
                &mut self,
                &types,
                &func_ty.param_tyids,
                func_ty.return_tyid,
            )
            .context("while applying the calling convention for parameters");

            match res {
                Ok(report) => {
                    if report.ok_count < param_count {
                        event!(
                            Level::WARN,
                            ok_count = report.ok_count,
                            total_count = param_count,
                            "partial parameter unpacking"
                        );
                    }
                }
                Err(err) => {
                    warnings.add(err.into());
                    // fine to continue with partially non-decoded arguments/return values
                }
            }
        }

        for insn in insns {
            // Temporary abstract registers
            //    These are used in the mil program to compute 'small stuff' (memory
            //    offsets, some arithmetic).  Never reused across different
            //    instructions.  "Generated" via self.pb (tmp_gen, tmp_reset)
            self.pb.tmp_reset();
            self.pb.set_input_addr(insn.ip());

            let mut output = String::new();
            formatter.format(&insn, &mut output);
            let _ = span!(Level::TRACE, "translating instruction", insn = output).enter();

            use iced_x86::Mnemonic as M;
            match insn.mnemonic() {
                /*
                    To be implemented:
                    * movaps Rxmm, Mem
                    * movaps Rxmm, Rxmm
                    * movups Mem, Rxmm
                    * movd   Rxmm, Rint
                    * movss  Rxmm, Mem
                    * movzx  Rint, Mem
                */
                M::Nop => {}

                M::Push => {
                    assert_eq!(insn.op_count(), 1);

                    let (value, sz) = self.emit_read(&insn, 0);

                    self.emit(
                        Self::RSP,
                        mil::Insn::ArithK(mil::ArithOp::Add, Self::RSP, -(sz as i64)),
                    );
                    let v0 = self.pb.tmp_gen();
                    self.emit(
                        v0,
                        mil::Insn::StoreMem {
                            addr: Self::RSP,
                            value,
                        },
                    );
                }
                M::Pop => {
                    assert_eq!(insn.op_count(), 1);

                    let v0 = self.pb.tmp_gen();

                    let sz = Self::op_size(&insn, 0);
                    if sz != 8 && sz != 2 {
                        panic!("assertion failed: pop dest size must be either 8 or 2 bytes");
                    }
                    self.emit(
                        v0,
                        mil::Insn::LoadMem {
                            addr: Self::RSP,
                            size: sz.into(),
                        },
                    );

                    self.emit_write(&insn, 0, v0, sz);
                    self.emit(
                        Self::RSP,
                        mil::Insn::ArithK(mil::ArithOp::Add, Self::RSP, sz as i64),
                    );
                }
                M::Leave => {
                    self.emit(Self::RSP, mil::Insn::Get(Self::RBP));
                    self.emit(
                        Self::RBP,
                        mil::Insn::LoadMem {
                            addr: Self::RSP,
                            size: 8,
                        },
                    );
                    self.emit(
                        Self::RSP,
                        mil::Insn::ArithK(mil::ArithOp::Add, Self::RSP, 8),
                    );
                }
                M::Ret => {
                    let ret_val = func_ty
                        .ok_or_else(|| anyhow::anyhow!("no function type"))
                        .and_then(|func_ty| {
                            callconv::pack_return_value(&mut self, &types, func_ty.return_tyid)
                                .context("decoding return value")
                        })
                        .or_warn(&mut warnings)
                        .unwrap_or(Self::RAX);

                    let v0 = self.pb.tmp_gen();
                    self.emit(v0, mil::Insn::SetReturnValue(ret_val));
                    self.emit(v0, mil::Insn::Control(Control::Ret));
                }

                // assuming that the instruction is correct and correctly
                // decoded by iced_x86, the same code should serve all these
                // variants of mov
                M::Mov | M::Movsd | M::Movaps | M::Movups => {
                    let (value, sz) = self.emit_read(&insn, 1);
                    self.emit_write(&insn, 0, value, sz);
                }

                M::Add => {
                    let (a, a_sz) = self.emit_read(&insn, 0);
                    let (b, b_sz) = self.emit_read(&insn, 1);
                    self.emit_arith(a, a_sz, b, b_sz, mil::ArithOp::Add);
                    self.emit_write(&insn, 0, a, a_sz);
                    self.emit_set_flags_arith(a);
                }
                M::Sub => {
                    let (a, a_sz) = self.emit_read(&insn, 0);
                    let (b, b_sz) = self.emit_read(&insn, 1);
                    assert_eq!(a_sz, b_sz, "sub: operands must be the same size");
                    self.emit_arith(a, a_sz, b, b_sz, mil::ArithOp::Sub);
                    self.emit_write(&insn, 0, a, a_sz);
                    self.emit_set_flags_arith(a);
                }
                M::Xor | M::Pxor => {
                    let (a, a_sz) = self.emit_read(&insn, 0);
                    let (b, b_sz) = self.emit_read(&insn, 1);
                    self.emit_bit_op(a_sz, b_sz, a, b, mil::ArithOp::BitXor, insn);
                }
                M::Or => {
                    let (a, a_sz) = self.emit_read(&insn, 0);
                    let (b, b_sz) = self.emit_read(&insn, 1);
                    self.emit_bit_op(a_sz, b_sz, a, b, mil::ArithOp::BitOr, insn);
                }
                M::And => {
                    let (a, a_sz) = self.emit_read(&insn, 0);
                    let (b, b_sz) = self.emit_read(&insn, 1);
                    self.emit_bit_op(a_sz, b_sz, a, b, mil::ArithOp::BitAnd, insn);
                }

                M::Inc => {
                    let (a, a_sz) = self.emit_read(&insn, 0);
                    self.emit(a, mil::Insn::ArithK(mil::ArithOp::Add, a, 1));
                    self.emit_write(&insn, 0, a, a_sz);

                    self.emit(Self::OF, mil::Insn::False);
                    self.emit(Self::CF, mil::Insn::False);
                    // TODO implement: AF
                    self.emit(Self::SF, mil::Insn::SignOf(a));
                    self.emit(Self::ZF, mil::Insn::IsZero(a));
                    let v0 = self.pb.tmp_gen();
                    self.emit(
                        v0,
                        mil::Insn::Part {
                            src: a,
                            offset: 0,
                            size: 1,
                        },
                    );
                    self.emit(Self::PF, mil::Insn::Parity(v0));
                }

                M::Test => {
                    let (a, a_sz) = self.emit_read(&insn, 0);
                    let (b, b_sz) = self.emit_read(&insn, 1);
                    let v0 = self.pb.tmp_gen();
                    self.emit_arith(a, a_sz, b, b_sz, mil::ArithOp::BitAnd);
                    self.emit(Self::SF, mil::Insn::SignOf(a));
                    self.emit(Self::ZF, mil::Insn::IsZero(a));
                    self.emit(
                        v0,
                        mil::Insn::Part {
                            src: a,
                            offset: 0,
                            size: 1,
                        },
                    );
                    self.emit(Self::PF, mil::Insn::Parity(a));
                    self.emit(Self::CF, mil::Insn::False);
                    self.emit(Self::OF, mil::Insn::False);
                }

                M::Cmp => {
                    let (a, a_sz) = self.emit_read(&insn, 0);
                    let (b, b_sz) = self.emit_read(&insn, 1);
                    assert_eq!(a_sz, b_sz, "cmp: operands must be the same size");
                    // just put the result in a tmp reg, then ignore it (other than for the
                    // flags)
                    self.emit_arith(a, a_sz, b, b_sz, mil::ArithOp::Sub);
                    self.emit_set_flags_arith(a);
                }

                M::Lea => match insn.op1_kind() {
                    OpKind::Memory => {
                        let v0 = self.pb.tmp_gen();
                        self.emit_compute_address_into(&insn, v0);
                        assert_eq!(insn.op0_kind(), OpKind::Register);
                        let value_size = insn.op0_register().size().try_into().unwrap();
                        self.emit_write(&insn, 0, v0, value_size);
                    }
                    OpKind::MemorySegSI
                    | OpKind::MemorySegESI
                    | OpKind::MemorySegRSI
                    | OpKind::MemorySegDI
                    | OpKind::MemorySegEDI
                    | OpKind::MemorySegRDI
                    | OpKind::MemoryESDI
                    | OpKind::MemoryESEDI
                    | OpKind::MemoryESRDI => {
                        panic!("lea: segment-relative memory operands are not yet supported")
                    }
                    _ => {
                        panic!("lea: invalid operand: second operand must be of type memory")
                    }
                },

                M::Shl => {
                    self.emit_shift(insn, mil::ArithOp::Shl);
                }
                M::Shr => {
                    self.emit_shift(insn, mil::ArithOp::Shr);
                }

                M::Imul => {
                    match insn.op_count() {
                        1 => {
                            let src_b = self.emit_read_value(&insn, 0);
                            let a_size = match insn.op0_kind() {
                                OpKind::Register => insn.op0_register().size(),
                                OpKind::Memory => insn.memory_size().size(),
                                kind => panic!("imul: invalid operand: {:?}", kind),
                            };
                            let (dest_hi, dest_lo) = match a_size {
                                1 => (Register::AH, Register::AL),
                                2 => (Register::DX, Register::AX),
                                4 => (Register::EDX, Register::EAX),
                                8 => (Register::RDX, Register::RAX),
                                _ => panic!("imul: invalid operand size: {}", a_size),
                            };

                            let v0 = self.emit_read_machine_reg(dest_lo);
                            self.emit(v0, mil::Insn::Arith(mil::ArithOp::Mul, v0, src_b));
                            self.emit(Self::OF, mil::Insn::OverflowOf(v0));
                            self.emit(Self::CF, mil::Insn::Get(Self::OF));
                            self.emit(Self::SF, mil::Insn::UndefinedBool);
                            self.emit(Self::ZF, mil::Insn::UndefinedBool);
                            self.emit(Self::AF, mil::Insn::UndefinedBool);
                            self.emit(Self::PF, mil::Insn::UndefinedBool);

                            let a_size = a_size.try_into().unwrap();
                            let result_hi = self.pb.tmp_gen();
                            self.emit(
                                result_hi,
                                mil::Insn::Part {
                                    src: v0,
                                    offset: a_size,
                                    size: a_size,
                                },
                            );
                            self.emit_write_machine_reg(dest_hi, a_size, result_hi);
                            self.emit(
                                v0,
                                mil::Insn::Part {
                                    src: v0,
                                    offset: 0,
                                    size: a_size,
                                },
                            );
                            self.emit_write_machine_reg(dest_lo, a_size, v0);
                        }

                        op_count @ (2 | 3) => {
                            let a = self.emit_read_value(&insn, 0);
                            let result_size = insn
                                .try_op_register(0)
                                .expect("imul: register as first operand")
                                .size()
                                .try_into()
                                .unwrap();
                            let b = self.emit_read_value(&insn, 1);
                            let result = self.pb.tmp_gen();
                            self.emit(result, mil::Insn::Arith(mil::ArithOp::Mul, a, b));

                            if op_count == 3 {
                                let imm64 = match insn.op_kind(2) {
                                    OpKind::Immediate8 => insn.immediate8() as i8 as i64,
                                    OpKind::Immediate16 => insn.immediate16() as i16 as i64,
                                    OpKind::Immediate32 => insn.immediate32() as i32 as i64,
                                    OpKind::Immediate64 => insn.immediate64() as i64,
                                    other => panic!("imul: invalid 3rd operand: {other:?}"),
                                };
                                let k = self.pb.tmp_gen();
                                self.emit(
                                    k,
                                    mil::Insn::Const {
                                        value: imm64,
                                        size: 8,
                                    },
                                );
                                self.emit(result, mil::Insn::Arith(mil::ArithOp::Mul, result, k));
                            }

                            self.emit_write(&insn, 0, result, result_size);
                            self.emit(Self::OF, mil::Insn::OverflowOf(result));
                            self.emit(Self::CF, mil::Insn::Get(Self::OF));
                            self.emit(Self::SF, mil::Insn::UndefinedBool);
                            self.emit(Self::ZF, mil::Insn::UndefinedBool);
                            self.emit(Self::AF, mil::Insn::UndefinedBool);
                            self.emit(Self::PF, mil::Insn::UndefinedBool);
                        }

                        other => panic!("imul: invalid operands count: {}", other),
                    };
                }

                M::Call => {
                    let (callee, sz) = self.emit_read(&insn, 0);
                    assert_eq!(
                        sz, 8,
                        "invalid call instruction: operand must be 8 bytes, not {}",
                        sz
                    );

                    let v1 = self.tmp_gen();
                    match self.resolve_call(&insn, &types) {
                        Some((subr_tyid, subr_ty, param_values)) => {
                            event!(Level::TRACE, ?subr_tyid, ?param_values, "resolved call");

                            self.emit_call(callee, param_values, subr_tyid, v1);

                            if let Err(err) = callconv::unpack_return_value(
                                &mut self,
                                &types,
                                subr_ty.return_tyid,
                                v1,
                            ) {
                                event!(Level::ERROR, ?err, "could not unpack return value");
                            }
                        }
                        None => {
                            // just a dumb approximation of a likely case
                            event!(Level::ERROR, "call unresolved, using a default fallback");
                            let param_values = vec![Self::RDI, Self::RSI, Self::RDX, Self::RCX];
                            self.emit_call(
                                callee,
                                param_values,
                                types.tyid_shared_unknown_unsized(),
                                v1,
                            );
                            self.emit(v1, mil::Insn::Get(Self::RAX));
                        }
                    }
                }

                //
                // Jumps
                //
                M::Jmp => {
                    // refactor with emit_jmpif?
                    let v0 = self.pb.tmp_gen();
                    match insn.op0_kind() {
                        OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => {
                            let target = insn.near_branch_target();
                            self.emit(v0, mil::Insn::Control(Control::JmpExt(target)));
                        }
                        _ => {
                            let addr_reg = self.emit_compute_address(&insn);
                            self.emit(addr_reg, mil::Insn::SetJumpTarget(addr_reg));
                            self.emit(v0, mil::Insn::Control(Control::JmpIndirect));
                        }
                    }
                }
                M::Ja => {
                    let cond = self.emit_cmp_a();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Je => {
                    let cond = self.emit_cmp_e();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jne => {
                    let cond = self.emit_cmp_ne();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jb => {
                    let cond = self.emit_cmp_b();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jl => {
                    let cond = self.emit_cmp_l();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jle => {
                    let cond = self.emit_cmp_le();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jae => {
                    let cond = self.emit_cmp_ae();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jbe => {
                    let cond = self.emit_cmp_be();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jcxz => {
                    let cond = self.emit_cmp_cxz();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jecxz => {
                    let cond = self.emit_cmp_ecxz();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jg => {
                    let cond = self.emit_cmp_g();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jge => {
                    let cond = self.emit_cmp_ge();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jno => {
                    let cond = self.emit_cmp_no();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jnp => {
                    let cond = self.emit_cmp_np();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jns => {
                    let cond = self.emit_cmp_ns();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jo => {
                    let cond = self.emit_cmp_o();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jp => {
                    let cond = self.emit_cmp_p();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jrcxz => {
                    let cond = self.emit_cmp_rcxz();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Js => {
                    let cond = self.emit_cmp_s();
                    self.emit_jmpif(insn, 0, cond);
                }

                M::Seta => {
                    let v0 = self.emit_cmp_a();
                    self.emit_flag_to_byte(v0);
                }
                M::Setae => {
                    let v0 = self.emit_cmp_ae();
                    self.emit_flag_to_byte(v0);
                }
                M::Setb => {
                    let v0 = self.emit_cmp_b();
                    self.emit_flag_to_byte(v0);
                }
                M::Setbe => {
                    let v0 = self.emit_cmp_be();
                    self.emit_flag_to_byte(v0);
                }
                M::Sete => {
                    let v0 = self.emit_cmp_e();
                    self.emit_flag_to_byte(v0);
                }
                M::Setg => {
                    let v0 = self.emit_cmp_g();
                    self.emit_flag_to_byte(v0);
                }
                M::Setge => {
                    let v0 = self.emit_cmp_ge();
                    self.emit_flag_to_byte(v0);
                }
                M::Setl => {
                    let v0 = self.emit_cmp_l();
                    self.emit_flag_to_byte(v0);
                }
                M::Setle => {
                    let v0 = self.emit_cmp_le();
                    self.emit_flag_to_byte(v0);
                }
                M::Setne => {
                    let v0 = self.emit_cmp_ne();
                    self.emit_flag_to_byte(v0);
                }
                M::Setno => {
                    let v0 = self.emit_cmp_no();
                    self.emit_flag_to_byte(v0);
                }
                M::Setnp => {
                    let v0 = self.emit_cmp_np();
                    self.emit_flag_to_byte(v0);
                }
                M::Setns => {
                    let v0 = self.emit_cmp_ns();
                    self.emit_flag_to_byte(v0);
                }
                M::Seto => {
                    let v0 = self.emit_cmp_o();
                    self.emit_flag_to_byte(v0);
                }
                M::Setp => {
                    let v0 = self.emit_cmp_p();
                    self.emit_flag_to_byte(v0);
                }

                M::Cbw => {
                    let al = self.pb.tmp_gen();
                    self.emit(
                        al,
                        mil::Insn::Part {
                            src: Self::RAX,
                            offset: 0,
                            size: 1,
                        },
                    );
                    self.emit(
                        Self::RAX,
                        mil::Insn::Widen {
                            reg: al,
                            target_size: 2,
                            sign: true,
                        },
                    );
                }
                M::Cwde => {
                    let ax = self.pb.tmp_gen();
                    self.emit(
                        ax,
                        mil::Insn::Part {
                            src: Self::RAX,
                            offset: 0,
                            size: 2,
                        },
                    );
                    self.emit(
                        Self::RAX,
                        mil::Insn::Widen {
                            reg: ax,
                            target_size: 4,
                            sign: true,
                        },
                    );
                }
                M::Cdqe => {
                    let eax = self.pb.tmp_gen();
                    self.emit(
                        eax,
                        mil::Insn::Part {
                            src: Self::RAX,
                            offset: 0,
                            size: 4,
                        },
                    );
                    self.emit(
                        Self::RAX,
                        mil::Insn::Widen {
                            reg: eax,
                            target_size: 8,
                            sign: true,
                        },
                    );
                }

                _ => {
                    let mut output = String::new();
                    formatter.format(&insn, &mut output);
                    let description = format!("unsupported: {}", output);
                    let v0 = self.pb.tmp_gen();
                    self.emit(v0, mil::Insn::NotYetImplemented(description.leak()));
                }
            }
        }

        let mil = self.build();
        Ok((mil, warnings))
    }

    fn resolve_call<'t>(
        &mut self,
        insn: &iced_x86::Instruction,
        types: &'t ty::TypeSet,
    ) -> Option<(ty::TypeID, &'t ty::Subroutine, Vec<mil::Reg>)> {
        let target_tyid = match insn.op0_kind() {
            OpKind::NearBranch64 => {
                let return_pc = insn.next_ip();
                let target = insn.near_branch_target();
                types.resolve_call(ty::CallSiteKey { return_pc, target })
            }
            _ => None,
        };
        let Some(subr_tyid) = target_tyid else {
            event!(Level::WARN, "no type hints for this callsite");
            return None;
        };

        let subr_ty = match check_subroutine_type(types, subr_tyid) {
            Ok(subr_ty) => subr_ty,
            Err(err) => {
                event!(
                    Level::ERROR,
                    ?err,
                    "could not narrow down to subroutine type"
                );
                return None;
            }
        };

        let param_values = match self.pack_params(&types, subr_ty) {
            Ok(params) => params,
            Err(err) => {
                event!(Level::ERROR, ?err, "could not pack params");
                return None;
            }
        };

        Some((subr_tyid, subr_ty, param_values))
    }

    fn emit_call(
        &mut self,
        callee: mil::Reg,
        param_values: Vec<mil::Reg>,
        target_tyid: ty::TypeID,
        ret_reg: mil::Reg,
    ) {
        let first_arg = if param_values.is_empty() {
            None
        } else {
            for (ndx, arg) in param_values.into_iter().rev().enumerate() {
                self.emit(
                    ret_reg,
                    mil::Insn::CArg {
                        value: arg,
                        next_arg: if ndx > 0 { Some(ret_reg) } else { None },
                    },
                );
            }
            Some(ret_reg)
        };
        self.emit(ret_reg, mil::Insn::Call { callee, first_arg });
        self.reset_all_flags();
    }

    fn reset_all_flags(&mut self) {
        self.emit(Self::CF, mil::Insn::UndefinedBool);
        self.emit(Self::PF, mil::Insn::UndefinedBool);
        self.emit(Self::AF, mil::Insn::UndefinedBool);
        self.emit(Self::ZF, mil::Insn::UndefinedBool);
        self.emit(Self::SF, mil::Insn::UndefinedBool);
        self.emit(Self::TF, mil::Insn::UndefinedBool);
        self.emit(Self::IF, mil::Insn::UndefinedBool);
        self.emit(Self::DF, mil::Insn::UndefinedBool);
        self.emit(Self::OF, mil::Insn::UndefinedBool);
    }

    fn pack_params(
        &mut self,
        types: &ty::TypeSet,
        subr_ty: &ty::Subroutine,
    ) -> Result<Vec<mil::Reg>> {
        let (_report, param_values) =
            callconv::pack_params(self, types, &subr_ty.param_tyids, subr_ty.return_tyid)
                .context("while applying calling convention")?;
        assert_eq!(_report.ok_count, param_values.len());
        Ok(param_values)
    }

    fn emit_bit_op(
        &mut self,
        a_sz: u16,
        b_sz: u16,
        a: mil::Reg,
        b: mil::Reg,
        arith_op: mil::ArithOp,
        insn: iced_x86::Instruction,
    ) {
        assert_eq!(a_sz, b_sz, "bit op: operands must be the same size");
        self.emit_arith(a, a_sz, b, b_sz, arith_op);
        self.emit_write(&insn, 0, a, a_sz);

        self.emit(Self::OF, mil::Insn::False);
        self.emit(Self::CF, mil::Insn::False);
        // TODO implement: AF
        self.emit(Self::SF, mil::Insn::SignOf(a));
        self.emit(Self::ZF, mil::Insn::IsZero(a));
        let v0 = self.pb.tmp_gen();
        self.emit(
            v0,
            mil::Insn::Part {
                src: a,
                offset: 0,
                size: 1,
            },
        );
        self.emit(Self::PF, mil::Insn::Parity(v0));
    }

    fn emit_arith(&mut self, a: mil::Reg, a_sz: u16, b: mil::Reg, b_sz: u16, op: mil::ArithOp) {
        // integer or xmm/ymm registers
        assert!([1, 2, 4, 8, 16, 32].contains(&a_sz));
        assert!([1, 2, 4, 8, 16, 32].contains(&b_sz));
        let sz = a_sz.max(b_sz);
        self.extend_zero(a, a_sz, sz);
        self.extend_zero(b, b_sz, sz);
        self.emit(a, mil::Insn::Arith(op, a, b));
    }

    fn emit_shift(&mut self, insn: iced_x86::Instruction, arith_op: mil::ArithOp) {
        let (value, sz) = self.emit_read(&insn, 0);
        let (bits_count, bits_count_size) = self.emit_read(&insn, 1);
        self.emit_arith(value, sz, bits_count, bits_count_size, arith_op);
        self.emit_write(&insn, 0, value, sz);

        // TODO implement flag cahnges: CF, OF
        // (these are more complex than others, as they depend on the exact value
        // of the bit count)
        self.emit(Self::SF, mil::Insn::SignOf(value));
        self.emit(Self::ZF, mil::Insn::IsZero(value));
        let v0 = self.pb.tmp_gen();
        self.emit(
            v0,
            mil::Insn::Part {
                src: value,
                offset: 0,
                size: 1,
            },
        );
        self.emit(Self::PF, mil::Insn::Parity(v0));
        // ignored: AF
    }

    fn emit_jmpif(&mut self, insn: iced_x86::Instruction, op_ndx: u32, cond: mil::Reg) {
        match insn.op_kind(op_ndx) {
            OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => {
                let target = insn.near_branch_target();
                let v0 = self.pb.tmp_gen();
                self.emit(v0, mil::Insn::SetJumpCondition(cond));
                self.emit(v0, mil::Insn::Control(Control::JmpExtIf(target)));
            }
            _ => {
                todo!("indirect jmpif");
            }
        }
    }

    /// Emit Insns that set the x86_64 flags as it happens after an arithmetic x86_64
    /// instruction.
    ///
    /// Argument `a` is the destination register of the arthmetic operation.  It is both used
    /// to refer to the result, or to otherwise identify the arithmetic operation itself.
    fn emit_set_flags_arith(&mut self, a: mil::Reg) {
        self.emit(Self::OF, mil::Insn::OverflowOf(a));
        self.emit(Self::CF, mil::Insn::CarryOf(a));
        // TODO implement: AF
        self.emit(Self::SF, mil::Insn::SignOf(a));
        self.emit(Self::ZF, mil::Insn::IsZero(a));
        let v0 = self.pb.tmp_gen();
        self.emit(
            v0,
            mil::Insn::Part {
                src: a,
                offset: 0,
                size: 1,
            },
        );
        self.emit(Self::PF, mil::Insn::Parity(v0));
    }

    fn op_size(insn: &iced_x86::Instruction, op_ndx: u32) -> u16 {
        match insn.op_kind(op_ndx) {
            OpKind::Register => insn.op_register(op_ndx).size().try_into().unwrap(),
            OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => 8,
            OpKind::FarBranch16 | OpKind::FarBranch32 => {
                todo!("not supported: far branch operands")
            }
            OpKind::Immediate8 => 1,
            OpKind::Immediate8_2nd => 1,
            OpKind::Immediate16 => 2,
            OpKind::Immediate32 => 4,
            OpKind::Immediate64 => 8,
            // these are sign-extended (to different sizes). the conversion to u64 keeps the same bits,
            // so I think we don't lose any info (semantic or otherwise)
            OpKind::Immediate8to16 => 2,
            OpKind::Immediate8to32 => 4,
            OpKind::Immediate8to64 => 8,
            OpKind::Immediate32to64 => 8,

            OpKind::MemorySegSI
            | OpKind::MemorySegESI
            | OpKind::MemorySegRSI
            | OpKind::MemorySegDI
            | OpKind::MemorySegEDI
            | OpKind::MemorySegRDI
            | OpKind::MemoryESDI
            | OpKind::MemoryESEDI
            | OpKind::MemoryESRDI => todo!("not supported: segment-relative memory operands"),

            OpKind::Memory => insn.memory_size().size().try_into().unwrap(),
        }
    }

    /// Emit MIL instructions for reading the given operand.
    ///
    /// The operand to be read is taken in as an instruction and an index; the operand to be
    /// read is the nth operand of the instruction.
    ///
    /// Return value: the register that stores the read value (in the MIL text), and the
    /// value's size in bytes (either 1, 2, 4, or 8).
    fn emit_read_value(&mut self, insn: &iced_x86::Instruction, op_ndx: u32) -> mil::Reg {
        let v0 = self.pb.tmp_gen();

        match insn.op_kind(op_ndx) {
            OpKind::Register => {
                let reg = insn.op_register(op_ndx);
                self.emit_read_machine_reg(reg)
            }
            OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => self.emit(
                v0,
                mil::Insn::Const {
                    value: insn.near_branch_target() as i64,
                    size: 8,
                },
            ),
            OpKind::FarBranch16 | OpKind::FarBranch32 => {
                todo!("not supported: far branch operands")
            }

            OpKind::Immediate8 => self.emit(
                v0,
                mil::Insn::Const {
                    value: insn.immediate8() as _,
                    size: 1,
                },
            ),
            OpKind::Immediate8_2nd => self.emit(
                v0,
                mil::Insn::Const {
                    value: insn.immediate8_2nd() as _,
                    size: 1,
                },
            ),
            OpKind::Immediate16 => self.emit(
                v0,
                mil::Insn::Const {
                    value: insn.immediate16() as _,
                    size: 2,
                },
            ),
            OpKind::Immediate32 => self.emit(
                v0,
                mil::Insn::Const {
                    value: insn.immediate32() as _,
                    size: 4,
                },
            ),
            OpKind::Immediate64 => self.emit(
                v0,
                mil::Insn::Const {
                    value: insn.immediate64() as _,
                    size: 8,
                },
            ),
            OpKind::Immediate8to16 => self.emit(
                v0,
                mil::Insn::Const {
                    value: insn.immediate8to16() as _,
                    size: 2,
                },
            ),
            OpKind::Immediate8to32 => self.emit(
                v0,
                mil::Insn::Const {
                    value: insn.immediate8to32() as _,
                    size: 4,
                },
            ),
            OpKind::Immediate8to64 => self.emit(
                v0,
                mil::Insn::Const {
                    value: insn.immediate8to64() as _,
                    size: 8,
                },
            ),
            OpKind::Immediate32to64 => self.emit(
                v0,
                mil::Insn::Const {
                    value: insn.immediate32to64() as _,
                    size: 8,
                },
            ),

            OpKind::MemorySegSI
            | OpKind::MemorySegESI
            | OpKind::MemorySegRSI
            | OpKind::MemorySegDI
            | OpKind::MemorySegEDI
            | OpKind::MemorySegRDI
            | OpKind::MemoryESDI
            | OpKind::MemoryESEDI
            | OpKind::MemoryESRDI => todo!("not supported: segment-relative memory operands"),

            OpKind::Memory => {
                use iced_x86::MemorySize;

                let addr = self.emit_compute_address(insn);

                // we're ignoring what's in the memory structure (for non-Offset
                // size types). assuming that the rest of the assembly handles
                // it correctly, we should be able to recover the correct
                // information later anyway.
                let memory_size = insn.memory_size();
                self.emit(
                    v0,
                    mil::Insn::LoadMem {
                        addr,
                        size: memory_size.size().try_into().unwrap(),
                    },
                );

                match memory_size {
                    MemorySize::WordOffset => {
                        self.emit(v0, mil::Insn::LoadMem { addr: v0, size: 2 });
                    }
                    MemorySize::DwordOffset => {
                        self.emit(v0, mil::Insn::LoadMem { addr: v0, size: 4 });
                    }
                    MemorySize::QwordOffset => {
                        self.emit(v0, mil::Insn::LoadMem { addr: v0, size: 8 });
                    }
                    _ => {}
                }

                v0
            }
        }
    }

    /// Read a register of any size, emitting mil::Insn::Part as necessary
    fn emit_read_machine_reg(&mut self, reg: Register) -> mil::Reg {
        let full_reg = reg.full_register();
        let value = Builder::xlat_reg(full_reg);
        if reg == full_reg {
            value
        } else {
            let dest = self.pb.tmp_gen();
            self.emit(
                dest,
                mil::Insn::Part {
                    src: value,
                    offset: 0,
                    size: reg.size().try_into().unwrap(),
                },
            )
        }
    }

    fn emit_read(&mut self, insn: &iced_x86::Instruction, op_ndx: u32) -> (mil::Reg, u16) {
        let value = self.emit_read_value(insn, op_ndx);
        let sz = Self::op_size(insn, op_ndx);
        (value, sz)
    }

    fn emit_write(
        &mut self,
        insn: &iced_x86::Instruction,
        dest_op_ndx: u32,
        value: mil::Reg,
        value_size: u16,
    ) {
        match insn.op_kind(dest_op_ndx) {
            OpKind::Register => {
                let dest = insn.op_register(dest_op_ndx);
                let dest_size: u16 = dest.size().try_into().unwrap();
                self.extend_zero(value, value_size, dest_size);
                self.emit_write_machine_reg(dest, value_size, value);
            }

            OpKind::NearBranch16
            | OpKind::NearBranch32
            | OpKind::NearBranch64
            | OpKind::FarBranch16
            | OpKind::FarBranch32
            | OpKind::Immediate8
            | OpKind::Immediate8_2nd
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64 => {
                panic!("invalid mov dest operand: {:?}", insn.op0_kind())
            }

            OpKind::MemorySegSI
            | OpKind::MemorySegESI
            | OpKind::MemorySegRSI
            | OpKind::MemorySegDI
            | OpKind::MemorySegEDI
            | OpKind::MemorySegRDI
            | OpKind::MemoryESDI
            | OpKind::MemoryESEDI
            | OpKind::MemoryESRDI => {
                todo!("mov: segment-relative memory destination operands are not supported")
            }

            OpKind::Memory => {
                assert_eq!(
                    value_size as usize,
                    insn.memory_size().size(),
                    "destination memory operand is not the same size as the value"
                );

                let addr = self.pb.tmp_gen();
                self.emit_compute_address_into(insn, addr);
                assert_ne!(value, addr);

                self.emit(addr, mil::Insn::StoreMem { addr, value });
            }
        }
    }

    fn extend_zero(&mut self, value: mil::Reg, value_size: u16, dest_size: u16) {
        assert!(
            dest_size >= value_size,
            "dest ({dest_size} bytes) can't fit source ({value_size} bytes)"
        );
        if dest_size > value_size {
            self.emit(
                value,
                mil::Insn::Widen {
                    reg: value,
                    target_size: dest_size,
                    sign: false,
                },
            );
        }
    }

    fn emit_write_machine_reg(&mut self, dest_reg: Register, value_size: u16, value: mil::Reg) {
        let full_dest_reg = dest_reg.full_register();
        let full_size = full_dest_reg.size().try_into().unwrap();
        let full_dest = Builder::xlat_reg(full_dest_reg);

        if value_size == full_size {
            self.emit(full_dest, mil::Insn::Get(value));
            return;
        }

        assert!(value_size < full_size);
        let unchanged_part = self.pb.tmp_gen();
        self.emit(
            unchanged_part,
            mil::Insn::Part {
                src: full_dest,
                offset: value_size,
                size: full_size - value_size,
            },
        );
        self.emit(
            full_dest,
            mil::Insn::Concat {
                hi: unchanged_part,
                lo: value,
            },
        );
    }

    fn emit_compute_address(&mut self, insn: &iced_x86::Instruction) -> mil::Reg {
        let v0 = self.pb.tmp_gen();
        self.emit_compute_address_into(insn, v0);
        v0
    }
    fn emit_compute_address_into(&mut self, insn: &iced_x86::Instruction, dest: mil::Reg) {
        assert_eq!(
            insn.segment_prefix(),
            Register::None,
            "emit_compute_address_into: segment-relative memory address operands are not supported",
        );

        self.pb.push(
            dest,
            mil::Insn::Const {
                value: insn.memory_displacement64() as i64,
                size: 8,
            },
        );

        match insn.memory_base() {
            Register::None => {}
            base => {
                // TODO make this recursive and use read_operand instead of xlat_reg?
                // addresses are 64-bit, so we use 8 bytes instructions
                self.pb.push(
                    dest,
                    mil::Insn::Arith(mil::ArithOp::Add, dest, Self::xlat_reg(base)),
                );
            }
        }

        match insn.memory_index() {
            Register::None => {}
            index_reg => {
                let v1 = self.pb.tmp_gen();
                let scale = insn.memory_index_scale() as i64;
                let reg = Self::xlat_reg(index_reg);
                // addresses are 64-bit in this architecture, so instructions are all with 8 bytes results
                let scaled = mil::Insn::ArithK(mil::ArithOp::Mul, reg, scale);
                self.pb.push(v1, scaled);
                self.pb
                    .push(dest, mil::Insn::Arith(mil::ArithOp::Add, dest, v1));
            }
        }
    }

    fn emit_cmp_a(&mut self) -> mil::Reg {
        // jmp if !SF && !ZF (also jnbe)
        let v0 = self.pb.tmp_gen();
        let v1 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::Not(Self::SF));
        self.emit(v1, mil::Insn::Not(Self::ZF));
        self.emit(v0, mil::Insn::Bool(mil::BoolOp::And, v0, v1));
        v0
    }

    fn emit_cmp_e(&mut self) -> mil::Reg {
        Self::ZF
    }

    fn emit_cmp_ne(&mut self) -> mil::Reg {
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::Not(Self::ZF));
        v0
    }

    fn emit_cmp_b(&mut self) -> mil::Reg {
        // also Jc, Jnae
        Self::CF
    }

    fn emit_cmp_l(&mut self) -> mil::Reg {
        // jmp if SF != OF
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::Cmp(mil::CmpOp::EQ, Self::SF, Self::OF));
        self.emit(v0, mil::Insn::Not(v0));
        v0
    }

    fn emit_cmp_le(&mut self) -> mil::Reg {
        // jmp if ZF=1 or SF != OF
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::Cmp(mil::CmpOp::EQ, Self::SF, Self::OF));
        self.emit(v0, mil::Insn::Not(v0));
        self.emit(v0, mil::Insn::Bool(mil::BoolOp::Or, v0, Self::ZF));
        v0
    }

    fn emit_cmp_ae(&mut self) -> mil::Reg {
        // also jnb, jnc
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::Not(Self::CF));
        v0
    }

    fn emit_cmp_be(&mut self) -> mil::Reg {
        // also jna
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::Bool(mil::BoolOp::Or, Self::CF, Self::ZF));
        v0
    }

    fn emit_cmp_cxz(&mut self) -> mil::Reg {
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::IsZero(Self::RCX));
        v0
    }

    fn emit_cmp_ecxz(&mut self) -> mil::Reg {
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::IsZero(Self::RCX));
        v0
    }

    fn emit_cmp_g(&mut self) -> mil::Reg {
        let v0 = self.pb.tmp_gen();
        let v1 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::Not(Self::ZF));
        self.emit(v1, mil::Insn::Cmp(mil::CmpOp::EQ, Self::SF, Self::OF));
        self.emit(v0, mil::Insn::Bool(mil::BoolOp::And, v0, v1));
        v0
    }

    fn emit_cmp_ge(&mut self) -> mil::Reg {
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::Cmp(mil::CmpOp::EQ, Self::SF, Self::OF));
        v0
    }

    fn emit_cmp_no(&mut self) -> mil::Reg {
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::Not(Self::OF));
        v0
    }

    fn emit_cmp_np(&mut self) -> mil::Reg {
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::Not(Self::PF));
        v0
    }

    fn emit_cmp_ns(&mut self) -> mil::Reg {
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::Not(Self::SF));
        v0
    }

    fn emit_cmp_o(&mut self) -> mil::Reg {
        Self::OF
    }

    fn emit_cmp_p(&mut self) -> mil::Reg {
        Self::PF
    }

    fn emit_cmp_rcxz(&mut self) -> mil::Reg {
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::IsZero(Self::RCX));
        v0
    }

    fn emit_cmp_s(&mut self) -> mil::Reg {
        Self::SF
    }

    fn emit_flag_to_byte(&mut self, reg: mil::Reg) -> mil::Reg {
        self.emit(
            reg,
            mil::Insn::Widen {
                reg,
                target_size: 1,
                sign: false,
            },
        );
        reg
    }

    // flags
    const CF: mil::Reg = mil::Reg(2); // Carry flag
    const PF: mil::Reg = mil::Reg(3); // Parity flag      true=even false=odd
    const AF: mil::Reg = mil::Reg(4); // Auxiliary Carry
    const ZF: mil::Reg = mil::Reg(5); // Zero flag        true=zero false=non-zero
    const SF: mil::Reg = mil::Reg(6); // Sign flag        true=neg false=pos
    const TF: mil::Reg = mil::Reg(7); // Trap flag
    const IF: mil::Reg = mil::Reg(8); // Interrupt enable true=enabled false=disabled
    const DF: mil::Reg = mil::Reg(9); // Direction flag   true=down false=up
    const OF: mil::Reg = mil::Reg(10); // Overflow flag    true=overflow false=no-overflow

    // general purpose regs
    const RBP: mil::Reg = mil::Reg(11);
    const RSP: mil::Reg = mil::Reg(12);
    const RIP: mil::Reg = mil::Reg(13);
    const RDI: mil::Reg = mil::Reg(14);
    const RSI: mil::Reg = mil::Reg(15);
    const RAX: mil::Reg = mil::Reg(16);
    const RBX: mil::Reg = mil::Reg(17);
    const RCX: mil::Reg = mil::Reg(18);
    const RDX: mil::Reg = mil::Reg(19);
    const R8: mil::Reg = mil::Reg(20);
    const R9: mil::Reg = mil::Reg(21);
    const R10: mil::Reg = mil::Reg(22);
    const R11: mil::Reg = mil::Reg(23);
    const R12: mil::Reg = mil::Reg(24);
    const R13: mil::Reg = mil::Reg(25);
    const R14: mil::Reg = mil::Reg(26);
    const R15: mil::Reg = mil::Reg(27);

    const ZMM0: mil::Reg = mil::Reg(28);
    const ZMM1: mil::Reg = mil::Reg(29);
    const ZMM2: mil::Reg = mil::Reg(30);
    const ZMM3: mil::Reg = mil::Reg(31);
    const ZMM4: mil::Reg = mil::Reg(32);
    const ZMM5: mil::Reg = mil::Reg(33);
    const ZMM6: mil::Reg = mil::Reg(34);
    const ZMM7: mil::Reg = mil::Reg(35);
    const ZMM8: mil::Reg = mil::Reg(36);
    const ZMM9: mil::Reg = mil::Reg(37);
    const ZMM10: mil::Reg = mil::Reg(38);
    const ZMM11: mil::Reg = mil::Reg(39);
    const ZMM12: mil::Reg = mil::Reg(40);
    const ZMM13: mil::Reg = mil::Reg(41);
    const ZMM14: mil::Reg = mil::Reg(42);
    const ZMM15: mil::Reg = mil::Reg(43);

    const R_TMP_FIRST: mil::Reg = mil::Reg(45);

    /// Translate a *full* register name
    fn xlat_reg(reg: iced_x86::Register) -> mil::Reg {
        match reg.full_register() {
            Register::None => panic!("invalid register: none"),
            Register::RBP => Self::RBP,
            Register::RSP => Self::RSP,
            Register::RIP => Self::RIP,
            Register::RDI => Self::RDI,
            Register::RSI => Self::RSI,
            Register::RAX => Self::RAX,
            Register::RBX => Self::RBX,
            Register::RCX => Self::RCX,
            Register::RDX => Self::RDX,
            Register::R8 => Self::R8,
            Register::R9 => Self::R9,
            Register::R10 => Self::R10,
            Register::R11 => Self::R11,
            Register::R12 => Self::R12,
            Register::R13 => Self::R13,
            Register::R14 => Self::R14,
            Register::R15 => Self::R15,
            Register::ZMM0 => Self::ZMM0,
            Register::ZMM1 => Self::ZMM1,
            Register::ZMM2 => Self::ZMM2,
            Register::ZMM3 => Self::ZMM3,
            Register::ZMM4 => Self::ZMM4,
            Register::ZMM5 => Self::ZMM5,
            Register::ZMM6 => Self::ZMM6,
            Register::ZMM7 => Self::ZMM7,
            Register::ZMM8 => Self::ZMM8,
            Register::ZMM9 => Self::ZMM9,
            Register::ZMM10 => Self::ZMM10,
            Register::ZMM11 => Self::ZMM11,
            Register::ZMM12 => Self::ZMM12,
            Register::ZMM13 => Self::ZMM13,
            Register::ZMM14 => Self::ZMM14,
            Register::ZMM15 => Self::ZMM15,
            _ => {
                panic!(
                    "unsupported register: {:?} (full: {:?})",
                    reg,
                    reg.full_register()
                )
            }
        }
    }

    fn emit(&mut self, dest: mil::Reg, insn: mil::Insn) -> mil::Reg {
        self.pb.push(dest, insn)
    }
}

pub fn check_subroutine_type(types: &ty::TypeSet, tyid: ty::TypeID) -> Result<&ty::Subroutine> {
    match &types.get_through_alias(tyid).expect("invalid type ID") {
        ty::Ty::Subroutine(subr_ty) => Ok(subr_ty),
        _ => anyhow::bail!("not a subroutine type (ID: {:?})", tyid),
    }
}

define_ancestral_name!(ANC_CF, "CF");
define_ancestral_name!(ANC_PF, "PF");
define_ancestral_name!(ANC_AF, "AF");
define_ancestral_name!(ANC_ZF, "ZF");
define_ancestral_name!(ANC_SF, "SF");
define_ancestral_name!(ANC_TF, "TF");
define_ancestral_name!(ANC_IF, "IF");
define_ancestral_name!(ANC_DF, "DF");
define_ancestral_name!(ANC_OF, "OF");
define_ancestral_name!(ANC_RBP, "RBP");
define_ancestral_name!(ANC_RSP, "RSP");
define_ancestral_name!(ANC_RIP, "RIP");
define_ancestral_name!(ANC_RDI, "RDI");
define_ancestral_name!(ANC_RSI, "RSI");
define_ancestral_name!(ANC_RAX, "RAX");
define_ancestral_name!(ANC_RBX, "RBX");
define_ancestral_name!(ANC_RCX, "RCX");
define_ancestral_name!(ANC_RDX, "RDX");
define_ancestral_name!(ANC_R8, "R8");
define_ancestral_name!(ANC_R9, "R9");
define_ancestral_name!(ANC_R10, "R10");
define_ancestral_name!(ANC_R11, "R11");
define_ancestral_name!(ANC_R12, "R12");
define_ancestral_name!(ANC_R13, "R13");
define_ancestral_name!(ANC_R14, "R14");
define_ancestral_name!(ANC_R15, "R15");
define_ancestral_name!(ANC_ZMM0, "ZMM0");
define_ancestral_name!(ANC_ZMM1, "ZMM1");
define_ancestral_name!(ANC_ZMM2, "ZMM2");
define_ancestral_name!(ANC_ZMM3, "ZMM3");
define_ancestral_name!(ANC_ZMM4, "ZMM4");
define_ancestral_name!(ANC_ZMM5, "ZMM5");
define_ancestral_name!(ANC_ZMM6, "ZMM6");
define_ancestral_name!(ANC_ZMM7, "ZMM7");
define_ancestral_name!(ANC_ZMM8, "ZMM8");
define_ancestral_name!(ANC_ZMM9, "ZMM9");
define_ancestral_name!(ANC_ZMM10, "ZMM10");
define_ancestral_name!(ANC_ZMM11, "ZMM11");
define_ancestral_name!(ANC_ZMM12, "ZMM12");
define_ancestral_name!(ANC_ZMM13, "ZMM13");
define_ancestral_name!(ANC_ZMM14, "ZMM14");
define_ancestral_name!(ANC_ZMM15, "ZMM15");

define_ancestral_name!(ANC_ARG0, "arg0");
define_ancestral_name!(ANC_ARG1, "arg1");
define_ancestral_name!(ANC_ARG2, "arg2");
define_ancestral_name!(ANC_ARG3, "arg3");
define_ancestral_name!(ANC_ARG4, "arg4");
define_ancestral_name!(ANC_ARG5, "arg5");
define_ancestral_name!(ANC_ARG6, "arg6");
define_ancestral_name!(ANC_ARG7, "arg7");
const ANC_ARGS: [mil::AncestralName; 8] = [
    ANC_ARG0, ANC_ARG1, ANC_ARG2, ANC_ARG3, ANC_ARG4, ANC_ARG5, ANC_ARG6, ANC_ARG7,
];
