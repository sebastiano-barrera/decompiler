use crate::mil::{self, AncestralName, RegType};
use crate::ty;
use iced_x86::{Formatter, IntelFormatter};
use iced_x86::{OpKind, Register};

use anyhow::{anyhow, Result};
use smallvec::SmallVec;

pub fn translate(insns: impl Iterator<Item = iced_x86::Instruction>) -> Result<mil::Program> {
    Builder::new().translate(insns)
}

pub struct Builder<'a> {
    pb: mil::ProgramBuilder,
    reg_gen: RegGen,
    types: Option<&'a ty::TypeSet>,
}

impl<'a> Builder<'a> {
    pub fn new() -> Self {
        let mut bld = Builder {
            pb: mil::ProgramBuilder::new(),
            reg_gen: Self::reset_reg_gen(),
            types: None,
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

        bld
    }

    pub fn build(self) -> mil::Program {
        self.pb.build()
    }

    pub fn use_types(&mut self, types: &'a ty::TypeSet) {
        self.types = Some(types);
    }

    fn init_ancestral(&mut self, reg: mil::Reg, anc_name: AncestralName, rt: RegType) {
        self.emit(reg, mil::Insn::Ancestral(anc_name));
        self.pb.set_ancestral_type(anc_name, rt);
    }

    pub fn translate(
        mut self,
        insns: impl Iterator<Item = iced_x86::Instruction>,
    ) -> Result<mil::Program> {
        use iced_x86::{OpKind, Register};

        let mut formatter = IntelFormatter::new();

        // ensure that all possible temporary registers are initialized at least
        // once. this in turn ensures that all phi nodes always have a valid
        // input value for all predecessors. this is useful because a variable
        // (register) may only be initialized in one path, while the program
        // relies on an ancestral value in the other path.
        for reg_ndx in Self::R_TMP_FIRST.0..=Self::R_TMP_LAST.0 {
            let reg = mil::Reg(reg_ndx);
            self.emit(reg, mil::Insn::Undefined);
        }

        for insn in insns {
            // Temporary abstract registers
            //    These are used in the mil program to compute 'small stuff' (memory
            //    offsets, some arithmetic).  Never reused across different
            //    instructions.  "Generated" via self.reg_gen (RegGen)
            self.reg_gen = Self::reset_reg_gen();
            self.pb.set_input_addr(insn.ip());

            let mut output = String::new();
            formatter.format(&insn, &mut output);

            use iced_x86::Mnemonic as M;
            match insn.mnemonic() {
                M::Nop => {}

                M::Push => {
                    assert_eq!(insn.op_count(), 1);

                    let (value, sz) = self.emit_read(&insn, 0);
                    let v0 = self.reg_gen.next();

                    self.emit(
                        Self::RSP,
                        mil::Insn::ArithK(mil::ArithOp::Add, Self::RSP, -(sz as i64)),
                    );
                    self.emit(v0, mil::Insn::StoreMem(Self::RSP, value));
                }
                M::Pop => {
                    assert_eq!(insn.op_count(), 1);

                    let v0 = self.reg_gen.next();

                    let sz = Self::op_size(&insn, 0);
                    if sz != 8 && sz != 2 {
                        panic!("assertion failed: pop dest size must be either 8 or 2 bytes");
                    }
                    self.emit(
                        v0,
                        mil::Insn::LoadMem {
                            reg: Self::RSP,
                            size: sz as i32,
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
                            reg: Self::RSP,
                            size: 8,
                        },
                    );
                    self.emit(
                        Self::RSP,
                        mil::Insn::ArithK(mil::ArithOp::Add, Self::RSP, 8),
                    );
                }
                M::Ret => {
                    let v0 = self.reg_gen.next();
                    self.emit(v0, mil::Insn::Ret(Self::RAX));
                }

                M::Mov => {
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
                M::Xor => {
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
                    let v0 = self.reg_gen.next();
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
                    let v0 = self.reg_gen.next();
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
                        let v0 = self.reg_gen.next();
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
                    let (value, sz) = self.emit_read(&insn, 0);
                    let (bits_count, bits_count_size) = self.emit_read(&insn, 1);
                    self.emit_arith(value, sz, bits_count, bits_count_size, mil::ArithOp::Shl);
                    self.emit_write(&insn, 0, value, sz);

                    // TODO implement flag cahnges: CF, OF
                    // (these are more complex than others, as they depend on the exact value
                    // of the bit count)
                    self.emit(Self::SF, mil::Insn::SignOf(value));
                    self.emit(Self::ZF, mil::Insn::IsZero(value));
                    let v0 = self.reg_gen.next();
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

                M::Call => {
                    if insn.op0_kind() == OpKind::NearBranch64 {
                        let target = insn.near_branch_target();
                        eprintln!("#call: to address 0x{:x}", target);
                        if let Some(ts) = self.types {
                            if let Some(tyid) = ts.get_known_object(target) {
                                let typ = ts.get(tyid).unwrap();
                                eprintln!("#call: resolved call to: {:?} = {:?}", tyid, typ);
                            }
                        }
                    }

                    let (callee, sz) = self.emit_read(&insn, 0);
                    assert_eq!(
                        sz, 8,
                        "invalid call instruction: operand must be 8 bytes, not {}",
                        sz
                    );
                    let ret_reg = Self::xlat_reg(Register::RAX);
                    self.emit(ret_reg, mil::Insn::Call(callee));

                    // TODO Use function type info to use the proper number of arguments (also
                    // allow different calling conventions)
                    // For now, we always assume exactly 4 arguments, using the sysv amd64 call
                    // conv.
                    for arch_reg in [Register::RDI, Register::RSI, Register::RDX, Register::RCX] {
                        let value = Self::xlat_reg(arch_reg);
                        let v1 = self.reg_gen.next();
                        self.emit(v1, mil::Insn::CArg(value));
                    }

                    self.emit(Self::CF, mil::Insn::Undefined);
                    self.emit(Self::PF, mil::Insn::Undefined);
                    self.emit(Self::AF, mil::Insn::Undefined);
                    self.emit(Self::ZF, mil::Insn::Undefined);
                    self.emit(Self::SF, mil::Insn::Undefined);
                    self.emit(Self::TF, mil::Insn::Undefined);
                    self.emit(Self::IF, mil::Insn::Undefined);
                    self.emit(Self::DF, mil::Insn::Undefined);
                    self.emit(Self::OF, mil::Insn::Undefined);
                }

                //
                // Jumps
                //
                M::Jmp => {
                    // refactor with emit_jmpif?
                    match insn.op0_kind() {
                        OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => {
                            let target = insn.near_branch_target();
                            let v0 = self.reg_gen.next();
                            self.emit(v0, mil::Insn::JmpExt(target));
                        }
                        _ => {
                            let addr = self.emit_compute_address(&insn);
                            self.emit(addr, mil::Insn::JmpInd(addr));
                        }
                    }
                }
                M::Ja => {
                    // jmp if !SF && !ZF
                    // also jnbe
                    let v0 = self.reg_gen.next();
                    let v1 = self.reg_gen.next();
                    self.emit(v0, mil::Insn::Not(Self::SF));
                    self.emit(v1, mil::Insn::Not(Self::ZF));
                    self.emit(v0, mil::Insn::Bool(mil::BoolOp::And, v0, v1));
                    self.emit_jmpif(insn, 0, v0);
                }
                M::Je => {
                    self.emit_jmpif(insn, 0, Self::ZF);
                }
                M::Jne => {
                    let v0 = self.reg_gen.next();
                    self.emit(v0, mil::Insn::Not(Self::ZF));
                    self.emit_jmpif(insn, 0, v0);
                }
                M::Jb => {
                    // also Jc, Jnae
                    self.emit_jmpif(insn, 0, Self::CF);
                }
                M::Jl => {
                    // jmp if SF != OF
                    let v0 = self.reg_gen.next();
                    self.emit(v0, mil::Insn::Cmp(mil::CmpOp::EQ, Self::SF, Self::OF));
                    self.emit(v0, mil::Insn::Not(v0));
                    self.emit_jmpif(insn, 0, v0);
                }
                M::Jle => {
                    // jmp if ZF=1 or SF ≠ OF
                    let v0 = self.reg_gen.next();
                    self.emit(v0, mil::Insn::Cmp(mil::CmpOp::EQ, Self::SF, Self::OF));
                    self.emit(v0, mil::Insn::Not(v0));
                    self.emit(v0, mil::Insn::Bool(mil::BoolOp::Or, v0, Self::ZF));
                    self.emit_jmpif(insn, 0, v0);
                }
                M::Jae => {
                    // also jnb, jnc
                    let v0 = self.reg_gen.next();
                    self.emit(v0, mil::Insn::Not(Self::CF));
                    self.emit_jmpif(insn, 0, v0);
                }
                M::Jbe => {
                    // also jna
                    let v0 = self.reg_gen.next();
                    self.emit(v0, mil::Insn::Bool(mil::BoolOp::Or, Self::CF, Self::ZF));
                    self.emit_jmpif(insn, 0, v0);
                }
                M::Jcxz => {
                    let v0 = self.reg_gen.next();
                    self.emit(v0, mil::Insn::IsZero(Self::RCX));
                    self.emit_jmpif(insn, 0, v0);
                }
                M::Jecxz => {
                    let v0 = self.reg_gen.next();
                    self.emit(v0, mil::Insn::IsZero(Self::RCX));
                    self.emit_jmpif(insn, 0, v0);
                }
                M::Jg => {
                    let v0 = self.reg_gen.next();
                    let v1 = self.reg_gen.next();
                    self.emit(v0, mil::Insn::Not(Self::ZF));
                    self.emit(v1, mil::Insn::Cmp(mil::CmpOp::EQ, Self::SF, Self::OF));
                    self.emit(v0, mil::Insn::Bool(mil::BoolOp::And, v0, v1));
                    self.emit_jmpif(insn, 0, v0);
                }
                M::Jge => {
                    let v0 = self.reg_gen.next();
                    self.emit(v0, mil::Insn::Cmp(mil::CmpOp::EQ, Self::SF, Self::OF));
                    self.emit_jmpif(insn, 0, v0);
                }
                M::Jno => {
                    let v0 = self.reg_gen.next();
                    self.emit(v0, mil::Insn::Not(Self::OF));
                    self.emit_jmpif(insn, 0, v0);
                }
                M::Jnp => {
                    let v0 = self.reg_gen.next();
                    self.emit(v0, mil::Insn::Not(Self::PF));
                    self.emit_jmpif(insn, 0, v0);
                }
                M::Jns => {
                    let v0 = self.reg_gen.next();
                    self.emit(v0, mil::Insn::Not(Self::SF));
                    self.emit_jmpif(insn, 0, v0);
                }
                M::Jo => {
                    self.emit_jmpif(insn, 0, Self::OF);
                }
                M::Jp => {
                    self.emit_jmpif(insn, 0, Self::PF);
                }
                M::Jrcxz => {
                    let v0 = self.reg_gen.next();
                    self.emit(v0, mil::Insn::IsZero(Self::RCX));
                    self.emit_jmpif(insn, 0, v0);
                }
                M::Js => {
                    self.emit_jmpif(insn, 0, Self::SF);
                }

                _ => {
                    let mut output = String::new();
                    formatter.format(&insn, &mut output);
                    let description = format!("unsupported: {}", output);
                    let v0 = self.reg_gen.next();
                    self.emit(v0, mil::Insn::TODO(description.leak()));
                }
            }
        }

        Ok(self.build())
    }

    fn emit_bit_op(
        &mut self,
        a_sz: u8,
        b_sz: u8,
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
        let v0 = self.reg_gen.next();
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

    fn emit_arith(&mut self, a: mil::Reg, a_sz: u8, b: mil::Reg, b_sz: u8, op: mil::ArithOp) {
        assert!([1, 2, 4, 8].contains(&a_sz));
        assert!([1, 2, 4, 8].contains(&b_sz));
        let sz = a_sz.max(b_sz);
        self.widen(a, sz);
        self.widen(b, sz);
        self.emit(a, mil::Insn::Arith(op, a, b));
    }

    fn widen(&mut self, src: mil::Reg, tgt_sz: u8) {
        self.emit(
            src,
            mil::Insn::Widen {
                reg: src,
                target_size: tgt_sz,
            },
        );
    }

    fn emit_jmpif(&mut self, insn: iced_x86::Instruction, op_ndx: u32, cond: mil::Reg) {
        match insn.op_kind(op_ndx) {
            OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => {
                let target = insn.near_branch_target();
                let v0 = self.reg_gen.next();
                self.emit(v0, mil::Insn::JmpExtIf { cond, addr: target });
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
        let v0 = self.reg_gen.next();
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

    fn op_size(insn: &iced_x86::Instruction, op_ndx: u32) -> u8 {
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
        let v0 = self.reg_gen.next();

        match insn.op_kind(op_ndx) {
            OpKind::Register => {
                let reg = insn.op_register(op_ndx);
                let full_reg = Builder::xlat_reg(reg.full_register());
                match reg.size() {
                    1 => self.emit(
                        v0,
                        mil::Insn::Part {
                            src: full_reg,
                            offset: 0,
                            size: 1,
                        },
                    ),
                    2 => self.emit(
                        v0,
                        mil::Insn::Part {
                            src: full_reg,
                            offset: 0,
                            size: 2,
                        },
                    ),
                    4 => self.emit(
                        v0,
                        mil::Insn::Part {
                            src: full_reg,
                            offset: 0,
                            size: 4,
                        },
                    ),
                    8 => full_reg,
                    other => panic!("invalid register size: {other}"),
                }
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
                // Instruction::memory_size()
                //
                // Instruction::memory_displacement64()
                // Instruction::memory_base()
                // Instruction::memory_index()
                // Instruction::memory_index_scale()
                //
                // Instruction::memory_segment()
                // Instruction::segment_prefix()

                let addr = self.emit_compute_address(insn);

                use iced_x86::MemorySize;
                match insn.memory_size() {
                    MemorySize::UInt8 | MemorySize::Int8 => {
                        self.emit(v0, mil::Insn::LoadMem { reg: addr, size: 1 })
                    }
                    MemorySize::UInt16 | MemorySize::Int16 => {
                        self.emit(v0, mil::Insn::LoadMem { reg: addr, size: 2 })
                    }
                    MemorySize::UInt32 | MemorySize::Int32 => {
                        self.emit(v0, mil::Insn::LoadMem { reg: addr, size: 4 })
                    }
                    MemorySize::UInt64 | MemorySize::Int64 => {
                        self.emit(v0, mil::Insn::LoadMem { reg: addr, size: 8 })
                    }
                    MemorySize::QwordOffset => {
                        self.emit(v0, mil::Insn::LoadMem { reg: addr, size: 8 });
                        self.emit(v0, mil::Insn::LoadMem { reg: v0, size: 8 })
                    }
                    other => todo!("unsupported size for memory operand: {:?}", other),
                }
            }
        }
    }

    fn emit_read(&mut self, insn: &iced_x86::Instruction, op_ndx: u32) -> (mil::Reg, u8) {
        let value = self.emit_read_value(insn, op_ndx);
        let sz = Self::op_size(insn, op_ndx);
        (value, sz)
    }

    fn emit_write(
        &mut self,
        insn: &iced_x86::Instruction,
        dest_op_ndx: u32,
        value: mil::Reg,
        value_size: u8,
    ) {
        match insn.op_kind(dest_op_ndx) {
            OpKind::Register => {
                let dest = insn.op_register(dest_op_ndx);
                assert_eq!(
                    dest.size(),
                    value_size as usize,
                    "mov: src and dest must have same size"
                );

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

                let addr = self.reg_gen.next();
                self.emit_compute_address_into(insn, addr);
                assert_ne!(value, addr);

                let v0 = self.reg_gen.next();
                self.emit(v0, mil::Insn::StoreMem(addr, value));
            }
        }
    }

    fn emit_write_machine_reg(&mut self, dest: Register, value_size: u8, value: mil::Reg) {
        let full_dest = Builder::xlat_reg(dest.full_register());

        if value_size == 8 {
            self.emit(full_dest, mil::Insn::Get(value));
            return;
        }

        assert!(value_size < 8);
        self.emit(
            full_dest,
            mil::Insn::Part {
                src: full_dest,
                offset: value_size,
                size: 8 - value_size,
            },
        );
        self.emit(
            full_dest,
            mil::Insn::Concat {
                hi: full_dest,
                lo: value,
            },
        );
    }

    fn emit_compute_address(&mut self, insn: &iced_x86::Instruction) -> mil::Reg {
        let v0 = self.reg_gen.next();
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
                let v1 = self.reg_gen.next();
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

    pub fn read_func_args(&mut self, arg_types: &[ty::TypeID]) -> Result<()> {
        use ty::Ty;

        if arg_types.len() > ANC_ARGS.len() {
            eprintln!(
                "warning: only {} arguments will be handled ({} were described)",
                ANC_ARGS.len(),
                arg_types.len()
            );
        }

        let mut stack_pos = 0;
        let rsp = Builder::xlat_reg(Register::RSP);
        let mut read_to_stack = move |b: &mut Builder, src| {
            let addr = b.reg_gen.next();
            b.emit(
                addr,
                mil::Insn::ArithK(mil::ArithOp::Sub, rsp, 8 * stack_pos),
            );
            b.emit(addr, mil::Insn::StoreMem(addr, src));
            stack_pos += 1;
        };

        let mut integer_regs = [
            Register::RDI,
            Register::RSI,
            Register::RDX,
            Register::RCX,
            Register::R8,
            Register::R9,
        ]
        .into_iter();
        let mut read_to_integer = move |b: &mut Builder, src, sz| match integer_regs.next() {
            Some(reg) => {
                b.emit_write_machine_reg(reg, sz, src);
            }
            None => read_to_stack(b, src),
        };

        let types = self.types.ok_or(anyhow!("TypeSet not provided"))?;

        for (&arg_tyid, arg_anc) in arg_types.iter().zip(&ANC_ARGS) {
            let arg_ty = &types
                .get(arg_tyid)
                .ok_or(anyhow!("undefined type: {arg_tyid:?}"))?
                .ty;
            assert_ne!(arg_ty.bytes_size(), 0);

            let arg = self.reg_gen.next();
            self.emit(arg, mil::Insn::Ancestral(*arg_anc));

            match arg_ty {
                Ty::Enum(ty::Enum {
                    base_type: int_ty, ..
                })
                | Ty::Int(int_ty) => {
                    if int_ty.size < 8 {
                        read_to_integer(self, arg, int_ty.size);
                    } else {
                        assert_eq!(int_ty.size % 8, 0);
                        let eightb_count = int_ty.size / 8;
                        for eightb_ndx in 0..eightb_count {
                            let eightb = self.reg_gen.next();
                            self.emit(
                                eightb,
                                mil::Insn::StructGet8 {
                                    struct_value: arg,
                                    offset: 8 * eightb_ndx,
                                },
                            );
                            read_to_integer(self, eightb, 8);
                        }
                    }
                }
                Ty::Ptr(_) => {
                    read_to_integer(self, arg, 8);
                }
                Ty::Float(_) => todo!("SSE registers and float function parameters"),
                Ty::Struct(struct_ty) => {
                    let arg_struct_cls = classify_struct(struct_ty, types)?;
                    match arg_struct_cls {
                        ArgStructClass::Memory { eightb_count } => {
                            for eightb_ndx in 0..eightb_count {
                                let eightb = self.reg_gen.next();
                                self.emit(
                                    eightb,
                                    mil::Insn::StructGet8 {
                                        struct_value: arg,
                                        offset: (8 * eightb_ndx).try_into().unwrap(),
                                    },
                                );
                                read_to_stack(self, eightb);
                            }
                        }
                        ArgStructClass::Registers(classes) => {
                            for (ndx, cls) in classes.iter().enumerate() {
                                match cls {
                                    ArgClass::None => {}
                                    ArgClass::Integer => {
                                        let eightb = self.reg_gen.next();
                                        self.emit(
                                            eightb,
                                            mil::Insn::StructGet8 {
                                                struct_value: arg,
                                                offset: (8 * ndx).try_into().unwrap(),
                                            },
                                        );
                                        read_to_integer(self, eightb, 8);
                                    }
                                }
                            }
                        }
                    }
                }

                Ty::Unknown(_) => {
                    return Err(anyhow!("Function argument's type can't be `Unknown`"))
                }
                Ty::Subroutine(_) | Ty::Bool(_) | Ty::Void => {
                    panic!("invalid type for a function argument: {arg_ty:?}")
                }
            }
        }

        Ok(())
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

    const R_TMP_FIRST: mil::Reg = mil::Reg(28);
    const R_TMP_LAST: mil::Reg = mil::Reg(38);

    fn reset_reg_gen() -> RegGen {
        RegGen::new(Self::R_TMP_FIRST, Self::R_TMP_LAST)
    }

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
            _ => panic!(
                "unsupported register: {:?} (full: {:?})",
                reg,
                reg.full_register()
            ),
        }
    }

    fn emit(&mut self, dest: mil::Reg, insn: mil::Insn) -> mil::Reg {
        self.pb.push(dest, insn)
    }
}

#[inline(always)]
fn div_ceil(size: usize, quot: usize) -> usize {
    (size + quot - 1) / quot
}

#[derive(Clone, Copy)]
enum ArgClass {
    None,
    Integer,
}
enum ArgStructClass {
    Memory { eightb_count: usize },
    Registers(SmallVec<[ArgClass; 8]>),
}
fn classify_struct(struct_ty: &ty::Struct, types: &ty::TypeSet) -> Result<ArgStructClass> {
    // TODO Remove heap alloc
    let eightb_count = div_ceil(struct_ty.size as usize, 8);
    if eightb_count > 8 {
        return Ok(ArgStructClass::Memory { eightb_count });
    }

    let mut buf = [ArgClass::None; 8];
    let classes = &mut buf[..eightb_count];

    let mut queue: SmallVec<[_; 8]> = smallvec::smallvec![(0, struct_ty)];
    while let Some((struct_ofs, struct_ty)) = queue.pop() {
        for memb in &struct_ty.members {
            let typ = types.get(memb.tyid).unwrap();
            let sz = typ.ty.bytes_size() as usize;
            let memb_ofs = struct_ofs + memb.offset as usize;

            if memb_ofs % sz != 0 {
                // unaligned struct member
                return Ok(ArgStructClass::Memory { eightb_count });
            }

            match &typ.ty {
                ty::Ty::Int(_) | ty::Ty::Enum(_) | ty::Ty::Ptr(_) => {
                    let memb_eightb_1 = memb_ofs / 8;
                    let memb_eightb_n = (memb_ofs + sz) / 8;
                    for i in memb_eightb_1..=memb_eightb_n {
                        classes[i] = ArgClass::Integer;
                    }
                }
                ty::Ty::Struct(substruct_ty) => {
                    queue.push((memb_ofs, substruct_ty));
                    continue;
                }
                ty::Ty::Float(_) => {
                    return Err(anyhow!("not yet implemented: float struct members"))
                }
                ty::Ty::Unknown(_) => {
                    return Err(anyhow!(
                        "struct is partially known; can't determine parameter passing"
                    ))
                }
                ty @ (ty::Ty::Subroutine(_) | ty::Ty::Bool(_) | ty::Ty::Void) => {
                    panic!("invalid type for struct member: {ty:?}")
                }
            };
        }
    }

    Ok(ArgStructClass::Registers(SmallVec::from_slice(classes)))
}

struct RegGen {
    next: mil::Reg,
    last: mil::Reg,
}
impl RegGen {
    fn new(first: mil::Reg, last: mil::Reg) -> Self {
        assert!(first.0 <= last.0);
        RegGen { next: first, last }
    }

    fn next(&mut self) -> mil::Reg {
        let ret = self.next;
        self.next.0 += 1;

        assert!(ret.0 <= self.last.0);
        ret
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
