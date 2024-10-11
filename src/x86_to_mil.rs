use crate::mil;
use iced_x86::{Formatter, IntelFormatter};
use iced_x86::{OpKind, Register};

use anyhow::Result;

pub fn translate(insns: impl Iterator<Item = iced_x86::Instruction>) -> Result<mil::Program> {
    Builder::new().translate(insns)
}

struct Builder {
    pb: mil::ProgramBuilder,
}

impl Builder {
    fn new() -> Builder {
        Builder {
            pb: mil::ProgramBuilder::new(),
        }
    }

    fn build(self) -> mil::Program {
        self.pb.build()
    }

    fn translate(
        mut self,
        insns: impl Iterator<Item = iced_x86::Instruction>,
    ) -> std::result::Result<mil::Program, anyhow::Error> {
        use iced_x86::{OpKind, Register};

        // Temporary abstract registers
        //    Abstract registers used in the mil program to compute 'small stuff' (memory
        //    offsets, some arithmetic).  Generally used only in the context of a single
        //    instruction.
        const V0: mil::Reg = Builder::V0;
        const V1: mil::Reg = Builder::V1;

        let mut formatter = IntelFormatter::new();

        self.emit(Self::RSP, mil::Insn::Ancestral(mil::Ancestral::StackBot));

        for insn in insns {
            self.pb.set_input_addr(insn.ip());

            let mut output = String::new();
            formatter.format(&insn, &mut output);
            eprintln!("converting: {}", output);

            use iced_x86::Mnemonic as M;
            match insn.mnemonic() {
                M::Push => {
                    assert_eq!(insn.op_count(), 1);

                    let (value, sz) = self.emit_read(&insn, 0);

                    self.emit(Self::RSP, mil::Insn::AddK(Self::RSP, -(sz as i64)));
                    self.emit(V0, mil::Insn::StoreMem(Self::RSP, value));
                }
                M::Pop => {
                    assert_eq!(insn.op_count(), 1);

                    let sz = Self::op_size(&insn, 0);
                    match sz {
                        8 => self.emit(V0, mil::Insn::LoadMem8(Self::RSP)),
                        2 => self.emit(V0, mil::Insn::LoadMem2(Self::RSP)),
                        _ => panic!("assertion failed: pop dest size must be either 8 or 2 bytes"),
                    };

                    self.emit_write(&insn, 0, V0, sz);
                    self.emit(Self::RSP, mil::Insn::AddK(Self::RSP, sz as i64));
                }
                M::Leave => {
                    self.emit(Self::RSP, mil::Insn::Get(Self::RBP));
                    self.emit(Self::RBP, mil::Insn::LoadMem8(Self::RSP));
                    self.emit(Self::RSP, mil::Insn::AddK(Self::RSP, 8));
                }
                M::Ret => {
                    self.emit(V0, mil::Insn::Ret(Self::RAX));
                }

                M::Mov => {
                    let (value, sz) = self.emit_read(&insn, 1);
                    self.emit_write(&insn, 0, value, sz);
                }

                M::Add => {
                    let (a, a_sz) = self.emit_read(&insn, 0);
                    let (b, b_sz) = self.emit_read(&insn, 1);
                    assert_eq!(a_sz, b_sz, "add: operands must be the same size");
                    self.emit(a, mil::Insn::Add(a, b));
                    self.emit_write(&insn, 0, a, a_sz);
                    self.emit_set_flags_arith(a);
                }
                M::Sub => {
                    let (a, a_sz) = self.emit_read(&insn, 0);
                    let (b, b_sz) = self.emit_read(&insn, 1);
                    assert_eq!(a_sz, b_sz, "sub: operands must be the same size");
                    self.emit(a, mil::Insn::Sub(a, b));
                    self.emit_write(&insn, 0, a, a_sz);
                    self.emit_set_flags_arith(a);
                }

                M::Test => {
                    let (a, _) = self.emit_read(&insn, 0);
                    let (b, _) = self.emit_read(&insn, 1);
                    self.emit(V0, mil::Insn::BitAnd(a, b));
                    self.emit(Self::SF, mil::Insn::SignOf(V0));
                    self.emit(Self::ZF, mil::Insn::IsZero(V0));
                    self.emit(V1, mil::Insn::L1(V0));
                    self.emit(Self::PF, mil::Insn::Parity(V1));
                    self.emit(Self::CF, mil::Insn::Const1(0));
                    self.emit(Self::OF, mil::Insn::Const1(0));
                }

                M::Cmp => {
                    let (a, a_sz) = self.emit_read(&insn, 0);
                    let (b, b_sz) = self.emit_read(&insn, 1);
                    assert_eq!(a_sz, b_sz, "cmp: operands must be the same size");
                    // just put the result in a tmp reg, then ignore it (other than for the
                    // flags)
                    self.emit(V0, mil::Insn::Sub(a, b));
                    self.emit_set_flags_arith(V0);
                }

                M::Lea => {
                    let (dest, _) = self.emit_read(&insn, 0);
                    match insn.op1_kind() {
                        OpKind::Memory => {
                            self.emit_compute_address_into(&insn, dest);
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
                    }
                }

                M::Shl => {
                    let (value, sz) = self.emit_read(&insn, 0);
                    let bits_count = self.emit_read_value(&insn, 1);
                    self.emit(value, mil::Insn::Shl(value, bits_count));
                    self.emit_write(&insn, 0, value, sz);

                    // TODO implement flag cahnges: CF, OF
                    // (these are more complex than others, as they depend on the exact value
                    // of the bit count)
                    self.emit(Self::SF, mil::Insn::SignOf(value));
                    self.emit(Self::ZF, mil::Insn::IsZero(value));
                    self.emit(Self::V0, mil::Insn::L1(value));
                    self.emit(Self::PF, mil::Insn::Parity(Self::V0));
                    // ignored: AF
                }

                M::Call => {
                    // TODO Use function type info to use the proper number of arguments (also
                    // allow different calling conventions)
                    // For now, we always assume exactly 4 arguments, using the sysv amd64 call
                    // conv.
                    self.emit(V1, mil::Insn::CArgEnd);
                    for arch_reg in [Register::RDI, Register::RSI, Register::RDX, Register::RCX] {
                        let value = Self::xlat_reg(arch_reg);
                        self.emit(V1, mil::Insn::CArg { value, prev: V1 });
                    }

                    let (callee, sz) = self.emit_read(&insn, 0);
                    assert_eq!(
                        sz, 8,
                        "invalid call instruction: operand must be 8 bytes, not {}",
                        sz
                    );
                    assert_ne!(callee, V1, "callee and arg start can't share a register");
                    let ret_reg = Self::xlat_reg(Register::RAX);
                    self.emit(ret_reg, mil::Insn::Call { callee, arg0: V1 });

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

                M::Jmp => {
                    // refactor with emit_jmpif?
                    match insn.op0_kind() {
                        OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => {
                            let target = insn.near_branch_target();
                            self.emit(Self::V0, mil::Insn::JmpK(target));
                        }
                        _ => {
                            todo!("indirect jmp");
                        }
                    }
                }
                M::Je => {
                    self.emit_jmpif(insn, 0, Self::ZF);
                }
                M::Jb => {
                    self.emit_jmpif(insn, 0, Self::CF);
                }
                M::Jle => {
                    // ZF=1 or SF ≠ OF
                    self.emit(Self::V0, mil::Insn::Eq(Self::SF, Self::OF));
                    self.emit(Self::V0, mil::Insn::Not(Self::V0));
                    self.emit(Self::V0, mil::Insn::BitOr(Self::V0, Self::ZF));
                    self.emit_jmpif(insn, 0, Self::V0);
                }
                _ => {
                    let mut output = String::new();
                    formatter.format(&insn, &mut output);
                    let description = format!("unsupported: {}", output);
                    self.emit(V0, mil::Insn::TODO(description.leak()));
                }
            }
        }

        Ok(self.build())
    }

    fn emit_jmpif(&mut self, insn: iced_x86::Instruction, op_ndx: u32, cond: mil::Reg) {
        match insn.op_kind(op_ndx) {
            OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => {
                let target = insn.near_branch_target();
                self.emit(Self::V0, mil::Insn::JmpIfK { cond, target });
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
        // ignored: AF
        self.emit(Self::SF, mil::Insn::SignOf(a));
        self.emit(Self::ZF, mil::Insn::IsZero(a));
        self.emit(Self::V0, mil::Insn::L1(a));
        self.emit(Self::PF, mil::Insn::Parity(Self::V0));
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
        const V0: mil::Reg = Builder::V0;
        // let v0 = Self::V0;

        match insn.op_kind(op_ndx) {
            OpKind::Register => {
                let reg = insn.op_register(op_ndx);
                let full_reg = Builder::xlat_reg(reg.full_register());
                match reg.size() {
                    1 => self.emit(V0, mil::Insn::L1(full_reg)),
                    2 => self.emit(V0, mil::Insn::L2(full_reg)),
                    4 => self.emit(V0, mil::Insn::L4(full_reg)),
                    8 => full_reg,
                    other => panic!("invalid register size: {other}"),
                }
            }
            OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => {
                self.emit(V0, mil::Insn::Const8(insn.near_branch_target()))
            }
            OpKind::FarBranch16 | OpKind::FarBranch32 => {
                todo!("not supported: far branch operands")
            }

            OpKind::Immediate8 => self.emit(V0, mil::Insn::Const1(insn.immediate8())),
            OpKind::Immediate8_2nd => self.emit(V0, mil::Insn::Const1(insn.immediate8_2nd())),
            OpKind::Immediate16 => self.emit(V0, mil::Insn::Const2(insn.immediate16())),
            OpKind::Immediate32 => self.emit(V0, mil::Insn::Const4(insn.immediate32())),
            OpKind::Immediate64 => self.emit(V0, mil::Insn::Const8(insn.immediate64())),
            // these are sign-extended (to different sizes). the conversion to u64 keeps the same bits,
            // so I think we don't lose any info (semantic or otherwise)
            OpKind::Immediate8to16 => {
                self.emit(V0, mil::Insn::Const2(insn.immediate8to16() as u16))
            }
            OpKind::Immediate8to32 => {
                self.emit(V0, mil::Insn::Const4(insn.immediate8to32() as u32))
            }
            OpKind::Immediate8to64 => {
                self.emit(V0, mil::Insn::Const8(insn.immediate8to64() as u64))
            }
            OpKind::Immediate32to64 => {
                self.emit(V0, mil::Insn::Const8(insn.immediate32to64() as u64))
            }

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
                        self.emit(V0, mil::Insn::LoadMem1(addr))
                    }
                    MemorySize::UInt16 | MemorySize::Int16 => {
                        self.emit(V0, mil::Insn::LoadMem2(addr))
                    }
                    MemorySize::UInt32 | MemorySize::Int32 => {
                        self.emit(V0, mil::Insn::LoadMem4(addr))
                    }
                    MemorySize::UInt64 | MemorySize::Int64 => {
                        self.emit(V0, mil::Insn::LoadMem8(addr))
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

                let full_dest = Builder::xlat_reg(dest.full_register());
                let modifier = match value_size {
                    1 => mil::Insn::WithL1(full_dest, value),
                    2 => mil::Insn::WithL2(full_dest, value),
                    4 => mil::Insn::WithL4(full_dest, value),
                    8 => mil::Insn::Get(value),
                    _ => panic!("invalid dest size"),
                };
                self.emit(full_dest, modifier);
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

                let addr = Self::V1;
                self.emit_compute_address_into(insn, addr);
                assert_ne!(value, addr);

                self.emit(Self::V0, mil::Insn::StoreMem(addr, value));
            }
        }
    }

    fn emit_compute_address(&mut self, insn: &iced_x86::Instruction) -> mil::Reg {
        self.emit_compute_address_into(insn, Self::V0);
        Self::V0
    }
    fn emit_compute_address_into(&mut self, insn: &iced_x86::Instruction, dest: mil::Reg) {
        assert_eq!(
            insn.segment_prefix(),
            Register::None,
            "emit_compute_address_into: segment-relative memory address operands are not supported",
        );

        self.pb
            .push(dest, mil::Insn::Const8(insn.memory_displacement64()));

        match insn.memory_base() {
            Register::None => {}
            base => {
                // TODO make this recursive and use read_operand instead of xlat_reg?
                self.pb
                    .push(dest, mil::Insn::Add(dest, Self::xlat_reg(base)));
            }
        }

        match insn.memory_index() {
            Register::None => {}
            index_reg => {
                let scale = insn.memory_index_scale();
                self.pb.push(
                    Self::V1,
                    mil::Insn::MulK32(Self::xlat_reg(index_reg), scale),
                );
                self.pb.push(dest, mil::Insn::Add(dest, Self::V1));
            }
        }
    }

    // TODO there must be a better way...
    // temporary registers, to represent interemediate steps
    const V0: mil::Reg = mil::Reg::Nor(0);
    const V1: mil::Reg = mil::Reg::Nor(1);

    // flags
    const CF: mil::Reg = mil::Reg::Nor(2); // Carry flag
    const PF: mil::Reg = mil::Reg::Nor(3); // Parity flag      true=even false=odd
    const AF: mil::Reg = mil::Reg::Nor(4); // Auxiliary Carry
    const ZF: mil::Reg = mil::Reg::Nor(5); // Zero flag        true=zero false=non-zero
    const SF: mil::Reg = mil::Reg::Nor(6); // Sign flag        true=neg false=pos
    const TF: mil::Reg = mil::Reg::Nor(7); // Trap flag
    const IF: mil::Reg = mil::Reg::Nor(8); // Interrupt enable true=enabled false=disabled
    const DF: mil::Reg = mil::Reg::Nor(9); // Direction flag   true=down false=up
    const OF: mil::Reg = mil::Reg::Nor(10); // Overflow flag    true=overflow false=no-overflow

    // general purpose regs
    const RBP: mil::Reg = mil::Reg::Nor(11);
    const RSP: mil::Reg = mil::Reg::Nor(12);
    const RIP: mil::Reg = mil::Reg::Nor(13);
    const RDI: mil::Reg = mil::Reg::Nor(14);
    const RSI: mil::Reg = mil::Reg::Nor(15);
    const RAX: mil::Reg = mil::Reg::Nor(16);
    const RBX: mil::Reg = mil::Reg::Nor(17);
    const RCX: mil::Reg = mil::Reg::Nor(18);
    const RDX: mil::Reg = mil::Reg::Nor(19);
    const R8: mil::Reg = mil::Reg::Nor(20);
    const R9: mil::Reg = mil::Reg::Nor(21);
    const R10: mil::Reg = mil::Reg::Nor(22);
    const R11: mil::Reg = mil::Reg::Nor(23);
    const R12: mil::Reg = mil::Reg::Nor(24);
    const R13: mil::Reg = mil::Reg::Nor(25);
    const R14: mil::Reg = mil::Reg::Nor(26);
    const R15: mil::Reg = mil::Reg::Nor(27);

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
