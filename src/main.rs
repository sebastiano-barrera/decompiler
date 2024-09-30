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

            let read_operand = |builder: &mut Builder, insn: &Instruction, op_ndx| match insn
                .op_kind(op_ndx)
            {
                OpKind::Register => {
                    let reg = insn.op_register(op_ndx);
                    let full_reg = Builder::xlat_reg(reg.full_register());
                    match reg.size() {
                        1 => {
                            builder.emit(V0, mil::Insn::L1(full_reg));
                            (V0, 1u8)
                        }
                        2 => {
                            builder.emit(V0, mil::Insn::L2(full_reg));
                            (V0, 2)
                        }
                        4 => {
                            builder.emit(V0, mil::Insn::L4(full_reg));
                            (V0, 4)
                        }
                        8 => (full_reg, 8),
                        other => panic!("invalid register size: {other}"),
                    }
                }
                OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => {
                    builder.emit(V0, mil::Insn::Const8(insn.near_branch_target()));
                    (V0, 8)
                }
                OpKind::FarBranch16 | OpKind::FarBranch32 => {
                    todo!("not supported: far branch operands")
                }

                OpKind::Immediate8 => {
                    builder.emit(V0, mil::Insn::Const1(insn.immediate8()));
                    (V0, 1)
                }
                OpKind::Immediate8_2nd => {
                    builder.emit(V0, mil::Insn::Const1(insn.immediate8_2nd()));
                    (V0, 1)
                }
                OpKind::Immediate16 => {
                    builder.emit(V0, mil::Insn::Const2(insn.immediate16()));
                    (V0, 2)
                }
                OpKind::Immediate32 => {
                    builder.emit(V0, mil::Insn::Const4(insn.immediate32()));
                    (V0, 4)
                }
                OpKind::Immediate64 => {
                    builder.emit(V0, mil::Insn::Const8(insn.immediate64()));
                    (V0, 8)
                }
                // these are sign-extended (to different sizes). the conversion to u64 keeps the same bits,
                // so I think we don't lose any info (semantic or otherwise)
                OpKind::Immediate8to16 => {
                    builder.emit(V0, mil::Insn::Const2(insn.immediate8to16() as u16));
                    (V0, 2)
                }
                OpKind::Immediate8to32 => {
                    builder.emit(V0, mil::Insn::Const4(insn.immediate8to32() as u32));
                    (V0, 4)
                }
                OpKind::Immediate8to64 => {
                    builder.emit(V0, mil::Insn::Const8(insn.immediate8to64() as u64));
                    (V0, 8)
                }
                OpKind::Immediate32to64 => {
                    builder.emit(V0, mil::Insn::Const8(insn.immediate32to64() as u64));
                    (V0, 8)
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

                    let addr = builder.emit_compute_address(insn);

                    use iced_x86::MemorySize;
                    match insn.memory_size() {
                        MemorySize::UInt8 | MemorySize::Int8 => {
                            builder.emit(V0, mil::Insn::LoadMem1(addr));
                            (V0, 1)
                        }
                        MemorySize::UInt16 | MemorySize::Int16 => {
                            builder.emit(V0, mil::Insn::LoadMem2(addr));
                            (V0, 2)
                        }
                        MemorySize::UInt32 | MemorySize::Int32 => {
                            builder.emit(V0, mil::Insn::LoadMem4(addr));
                            (V0, 4)
                        }
                        MemorySize::UInt64 | MemorySize::Int64 => {
                            builder.emit(V0, mil::Insn::LoadMem8(addr));
                            (V0, 8)
                        }
                        other => todo!("unsupported size for memory operand: {:?}", other),
                    }
                }
            };

            let mut formatter = IntelFormatter::new();

            for insn in insns {
                let mut output = String::new();
                formatter.format(&insn, &mut output);
                eprintln!("converting: {}", output);

                use iced_x86::Mnemonic as M;
                match insn.mnemonic() {
                    M::Push => {
                        assert_eq!(insn.op_count(), 1);

                        let rsp = Builder::xlat_reg(Register::RSP);
                        let (value, sz) = read_operand(&mut self, &insn, 0);
                        let sz = sz as i64;

                        self.emit(rsp, mil::Insn::AddK(rsp, -sz));
                        self.emit(V0, mil::Insn::StoreMem(rsp, value));
                    }

                    M::Mov => {
                        let (value, sz) = read_operand(&mut self, &insn, 1);
                        self.emit_write(insn, value, sz);
                    }

                    M::Shl => {
                        let (value, sz) = read_operand(&mut self, &insn, 0);
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

        fn emit_write(&mut self, insn: iced_x86::Instruction, value: mil::Reg, value_size: u8) {
            match insn.op0_kind() {
                OpKind::Register => {
                    let dest = insn.op0_register();
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
            assert_eq!(insn.segment_prefix(), Register::None);

            self.pb
                .push(Self::V0, mil::Insn::Const8(insn.memory_displacement64()));

            match insn.memory_base() {
                Register::None => {}
                base => {
                    // TODO make this recursive and use read_operand instead of xlat_reg?
                    let push = self
                        .pb
                        .push(Self::V0, mil::Insn::Add(Self::V0, Self::xlat_reg(base)));
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
                    self.pb.push(Self::V0, mil::Insn::Add(Self::V0, Self::V1));
                }
            }

            Self::V0
        }

        const V0: mil::Reg = mil::Reg(0);
        const V1: mil::Reg = mil::Reg(1);
        const TMP_REG_COUNT: u16 = 2;

        /// Translate a *full* register name
        fn xlat_reg(reg: iced_x86::Register) -> mil::Reg {
            let rel_id = match reg.full_register() {
                Register::None => panic!("invalid register: none"),
                Register::RAX => 0,
                Register::RBP => 1,
                Register::RBX => 2,
                Register::RCX => 3,
                Register::RDI => 4,
                Register::RDX => 5,
                Register::RIP => 6,
                Register::RSI => 7,
                Register::RSP => 8,
                Register::R8 => 9,
                Register::R9 => 10,
                Register::R10 => 11,
                Register::R11 => 12,
                Register::R12 => 13,
                Register::R13 => 14,
                Register::R14 => 15,
                Register::R15 => 16,
                _ => panic!(
                    "unsupported register: {:?} (full: {:?})",
                    reg,
                    reg.full_register()
                ),
            };
            mil::Reg(Self::TMP_REG_COUNT + rel_id)
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

                    Insn::LoadMem1(addr) => print!("{:8} addr:r{}", "lmem1", addr.0),
                    Insn::LoadMem2(addr) => print!("{:8} addr:r{}", "lmem2", addr.0),
                    Insn::LoadMem4(addr) => print!("{:8} addr:r{}", "lmem4", addr.0),
                    Insn::LoadMem8(addr) => print!("{:8} addr:r{}", "lmem8", addr.0),

                    Insn::StoreMem(addr, val) => {
                        print!("{:8} *r{}<r{}", "smem", addr.0, val.0)
                    }
                    Insn::TODO(msg) => print!("{:8} {}", "TODO", msg),
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

        TODO(&'static str),

        LoadMem1(Reg),
        LoadMem2(Reg),
        LoadMem4(Reg),
        LoadMem8(Reg),
        StoreMem(Reg, Reg),
    }
}

/// Control Flow Graph of some MCode
struct CFG {}
