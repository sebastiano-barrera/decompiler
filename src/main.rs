#![allow(unused)]
use std::io::Read;
use std::{fs::File, path::PathBuf};

use iced_x86::{Decoder, Formatter, IntelFormatter, OpKind};

use anyhow::Result;

struct CliOptions {
    elf_filename: PathBuf,
    function_name: String,
}

impl CliOptions {
    fn parse<S: AsRef<str>>(mut opts: impl Iterator<Item = S>) -> Option<Self> {
        let elf_filename = opts.next()?;
        let elf_filename = PathBuf::from(elf_filename.as_ref());

        let function_name = opts.next()?.as_ref().to_owned();

        Some(CliOptions {
            elf_filename,
            function_name,
        })
    }
}

fn main() {
    let mut args = std::env::args();
    let program_name = args.next().unwrap();
    let opts = match CliOptions::parse(args) {
        Some(opts) => opts,
        None => {
            eprintln!("usage: {} EXEC FUNCTION", program_name);
            eprintln!("      EXEC = path to the executable (only ELF is supported)");
            eprintln!("  FUNCTION = name of the function to analyze (e.g. 'main')");
            return;
        }
    };

    // TODO Replace with memory mapping? (but it requires locking, see memmap2's docs)
    // https://docs.rs/memmap2/0.9.5/memmap2/struct.Mmap.html#safety
    let contents = {
        let mut contents = Vec::new();
        let mut elf = File::open(opts.elf_filename).expect("could not open executable");
        elf.read_to_end(&mut contents).expect("read error");
        contents
    };

    let object = goblin::Object::parse(&contents).expect("elf parse error");
    let elf = match object {
        goblin::Object::Elf(elf) => elf,
        _ => {
            eprintln!("unsupported executable format: {:?}", object);
            return;
        }
    };

    let func_sym = elf
        .syms
        .iter()
        .find(|sym| elf.strtab.get_at(sym.st_name) == Some(&opts.function_name))
        .expect("symbol not found");

    if !func_sym.is_function() {
        eprintln!("symbol `{}` is not a function", opts.function_name);
        return;
    }

    let func_addr = func_sym.st_value as usize;
    let func_size = func_sym.st_size as usize;
    let func_end = func_addr + func_size;

    let text_section = elf
        .section_headers
        .iter()
        .find(|sec| sec.is_executable() && elf.shdr_strtab.get_at(sec.sh_name) == Some(".text"))
        .expect("no .text section?!");

    let vm_range = text_section.vm_range();
    if vm_range.start > func_addr || vm_range.end < func_end {
        eprintln!(
            "function memory range (0x{:x}-0x{:x}) out of .text section vm range (0x{:x}-0x{:x})",
            func_addr, func_end, vm_range.start, vm_range.end
        );
    }

    // function's offset into the file
    let func_section_ofs = func_addr - vm_range.start;
    let func_fofs = text_section.sh_offset as usize + func_section_ofs as usize;
    let func_text = &contents[func_fofs..func_fofs + func_size];
    println!(
        "{} 0x{:x}+{} (file 0x{:x})",
        opts.function_name, func_addr, func_size, func_fofs,
    );

    let decoder = Decoder::with_ip(
        64,
        func_text,
        func_addr.try_into().unwrap(),
        iced_x86::DecoderOptions::NONE,
    );
    let mut formatter = IntelFormatter::new();
    let mut instr_strbuf = String::new();
    for instr in decoder {
        print!("{:16x}: ", instr.ip());
        let ofs = instr.ip() as usize - func_addr;
        let len = instr.len();
        for i in 0..8 {
            if i < len {
                print!("{:02x} ", func_text[ofs + i]);
            } else {
                print!("   ");
            }
        }

        instr_strbuf.clear();
        formatter.format(&instr, &mut instr_strbuf);
        println!("{}", instr_strbuf);
    }

    println!();
    let mut decoder = Decoder::with_ip(
        64,
        func_text,
        func_addr.try_into().unwrap(),
        iced_x86::DecoderOptions::NONE,
    );
    let prog = x86_to_mil::translate(decoder.iter()).unwrap();
    println!("mil program = ");
    prog.dump();
}

mod x86_to_mil {
    use crate::mil;
    use iced_x86::{Formatter, IntelFormatter};
    use iced_x86::{OpKind, Register};

    use anyhow::Result;

    pub fn translate(
        mut insns: impl Iterator<Item = iced_x86::Instruction>,
    ) -> Result<mil::Program> {
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
            use iced_x86::{Instruction, OpKind, Register};

            // Temporary abstract registers
            //    Abstract registers used in the mil program to compute 'small stuff' (memory
            //    offsets, some arithmetic).  Generally used only in the context of a single
            //    instruction.
            const V0: mil::Reg = Builder::V0;
            const V1: mil::Reg = Builder::V1;

            let mut formatter = IntelFormatter::new();

            for insn in insns {
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
                            _ => panic!(
                                "assertion failed: pop dest size must be either 8 or 2 bytes"
                            ),
                        };

                        self.emit_write(&insn, 0, V0, sz);
                        self.emit(Self::RSP, mil::Insn::AddK(Self::RSP, sz as i64));
                    }
                    M::Leave => {
                        self.emit(Self::RSP, mil::Insn::Get(Self::RBP));
                        self.emit(Self::RBP, mil::Insn::LoadMem8(Self::RSP));
                        self.emit(Self::RSP, mil::Insn::AddK(Self::RSP, 8));
                    }
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
                        self.emit_set_flags_arith(a);
                    }
                    M::Sub => {
                        let (a, a_sz) = self.emit_read(&insn, 0);
                        let (b, b_sz) = self.emit_read(&insn, 1);
                        assert_eq!(a_sz, b_sz, "sub: operands must be the same size");
                        self.emit(a, mil::Insn::Sub(a, b));
                        self.emit_set_flags_arith(a);
                    }

                    M::Test => {
                        let (a, a_sz) = self.emit_read(&insn, 0);
                        let (b, b_sz) = self.emit_read(&insn, 1);
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
                        let (dest, dest_sz) = self.emit_read(&insn, 0);
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
                            | OpKind::MemoryESRDI => panic!(
                                "lea: segment-relative memory operands are not yet supported"
                            ),
                            _ => {
                                panic!(
                                    "lea: invalid operand: second operand must be of type memory"
                                )
                            }
                        }
                    }

                    M::Shl => {
                        let (value, sz) = self.emit_read(&insn, 0);
                        let bits_count = self.emit_read_value(&insn, 1);
                        self.emit(value, mil::Insn::Shl(value, bits_count));

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
                        for arch_reg in [Register::RDI, Register::RSI, Register::RDX, Register::RCX]
                        {
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
                    self.emit(
                        Self::V0,
                        mil::Insn::TODO(format!("todo: mov to memory").leak()),
                    );
                }
            }
        }

        fn emit_compute_address(&mut self, insn: &iced_x86::Instruction) -> mil::Reg {
            self.emit_compute_address_into(insn, Self::V0);
            Self::V0
        }
        fn emit_compute_address_into(&mut self, insn: &iced_x86::Instruction, dest: mil::Reg) {
            assert_eq!(
                insn.segment_prefix(), Register::None,
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
        const V0: mil::Reg = mil::Reg(0);
        const V1: mil::Reg = mil::Reg(1);

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

    #[cfg(test)]
    mod tests {
        use super::translate;
        use iced_x86::Instruction;

        #[test]
        fn test_push_non_reg_operand() {
            translate([Instruction::new()].into_iter()).unwrap();
        }
    }
}

/// Machine-Independent Language
mod mil {
    // TODO This currently only represents the pre-SSA version of the program.

    #[derive(Debug)]
    pub struct Program {
        // TODO Add origin info to each insn
        insns: Vec<Insn>,
        dests: Vec<Reg>,
    }

    impl Program {
        pub fn dump(&self) {
            println!("program  {} instrs", self.insns.len());
            for (insn, dest) in self.insns.iter().zip(self.dests.iter()) {
                print!(" r{:<3} <- ", dest.0);

                match insn {
                    Insn::Const1(val) => print!("{:8} {} (0x{:x})", "const1", *val as i64, val),
                    Insn::Const2(val) => print!("{:8} {} (0x{:x})", "const2", *val as i64, val),
                    Insn::Const4(val) => print!("{:8} {} (0x{:x})", "const4", *val as i64, val),
                    Insn::Const8(val) => print!("{:8} {} (0x{:x})", "const8", *val as i64, val),
                    Insn::L1(x) => print!("{:8} r{}", "l1", x.0),
                    Insn::L2(x) => print!("{:8} r{}", "l2", x.0),
                    Insn::L4(x) => print!("{:8} r{}", "l4", x.0),
                    Insn::Get(x) => print!("{:8} r{}", "get", x.0),
                    Insn::WithL1(full, part) => {
                        print!("{:8} r{}<-r{}", "with.l1", full.0, part.0)
                    }
                    Insn::WithL2(full, part) => {
                        print!("{:8} r{}<-r{}", "with.l2", full.0, part.0)
                    }
                    Insn::WithL4(full, part) => {
                        print!("{:8} r{}<-r{}", "with.l4", full.0, part.0)
                    }

                    Insn::Add(a, b) => print!("{:8} r{},r{}", "add", a.0, b.0),
                    Insn::AddK(a, b) => print!("{:8} r{},{}", "add", a.0, *b as i64),
                    Insn::Sub(a, b) => print!("{:8} r{},r{}", "sub", a.0, b.0),
                    Insn::Mul(a, b) => print!("{:8} r{},r{}", "mul", a.0, b.0),
                    Insn::MulK32(a, b) => print!("{:8} r{},0x{:x}", "mul", a.0, b),
                    Insn::Shl(value, bits_count) => {
                        print!("{:8} r{},r{}", "shl", value.0, bits_count.0)
                    }
                    Insn::BitAnd(a, b) => print!("{:8} r{},r{}", "and", a.0, b.0),
                    Insn::BitOr(a, b) => print!("{:8} r{},r{}", "or", a.0, b.0),

                    Insn::LoadMem1(addr) => print!("{:8} addr:r{}", "lmem1", addr.0),
                    Insn::LoadMem2(addr) => print!("{:8} addr:r{}", "lmem2", addr.0),
                    Insn::LoadMem4(addr) => print!("{:8} addr:r{}", "lmem4", addr.0),
                    Insn::LoadMem8(addr) => print!("{:8} addr:r{}", "lmem8", addr.0),

                    Insn::StoreMem(addr, val) => {
                        print!("{:8} *r{}<r{}", "smem", addr.0, val.0)
                    }
                    Insn::TODO(msg) => print!("{:8} {}", "TODO", msg),

                    Insn::Call { callee, arg0 } => {
                        print!("{:8} r{}(r{})", "call", callee.0, arg0.0)
                    }
                    Insn::CArgEnd => print!("cargend"),
                    Insn::CArg { value, prev } => {
                        print!("{:8} r{} after r{}", "carg", value.0, prev.0)
                    }

                    Insn::OverflowOf(x) => print!("{:8} r{}", "overflow", x.0),
                    Insn::CarryOf(x) => print!("{:8} r{}", "carry", x.0),
                    Insn::SignOf(x) => print!("{:8} r{}", "sign", x.0),
                    Insn::IsZero(x) => print!("{:8} r{}", "is0", x.0),
                    Insn::Parity(x) => print!("{:8} r{}", "parity", x.0),

                    Insn::Undefined => print!("undef"),
                }

                println!();
            }
        }
    }

    // will be mostly useful to keep origin info later
    pub struct ProgramBuilder {
        insns: Vec<Insn>,
        dests: Vec<Reg>,
    }

    impl ProgramBuilder {
        pub fn new() -> Self {
            Self {
                insns: Vec::new(),
                dests: Vec::new(),
            }
        }

        pub fn push(&mut self, dest: Reg, insn: Insn) -> Reg {
            self.dests.push(dest);
            self.insns.push(insn);
            dest
        }

        pub fn build(self) -> Program {
            assert_eq!(self.dests.len(), self.insns.len());
            Program {
                insns: self.insns,
                dests: self.dests,
            }
        }
    }

    /// Register ID
    ///
    /// The language admits as many registers as a u16 can represent (2**16). They're
    /// abstract, so we don't pay for them!
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Reg(pub u16);

    #[derive(Clone, Copy, Debug)]
    pub enum Insn {
        Const1(u8),
        Const2(u16),
        Const4(u32),
        Const8(u64),

        L1(Reg),
        L2(Reg),
        L4(Reg),
        Get(Reg),
        WithL1(Reg, Reg),
        WithL2(Reg, Reg),
        WithL4(Reg, Reg),

        Add(Reg, Reg),
        AddK(Reg, i64),
        Sub(Reg, Reg),
        Mul(Reg, Reg),
        MulK32(Reg, u32),
        Shl(Reg, Reg),
        BitAnd(Reg, Reg),
        BitOr(Reg, Reg),

        // call args are represented with a linked list:
        //  r0 <- [compute callee]
        //  r1 <- [compute arg 0]
        //  r2 <- [compute arg 1]
        //  r3 <- cargend
        //  r4 <- carg r1 then r3
        //  r5 <- carg r2 then r4
        //  r6 <- call r0(r5)
        // destination vreg is for the return value
        Call { callee: Reg, arg0: Reg },
        CArgEnd,
        CArg { value: Reg, prev: Reg },

        TODO(&'static str),

        LoadMem1(Reg),
        LoadMem2(Reg),
        LoadMem4(Reg),
        LoadMem8(Reg),
        StoreMem(Reg, Reg),

        OverflowOf(Reg),
        CarryOf(Reg),
        SignOf(Reg),
        IsZero(Reg),
        Parity(Reg),

        Undefined,
    }
}

/// Control Flow Graph of some MCode
struct CFG {}
