/// Machine-Independent Language
// TODO This currently only represents the pre-SSA version of the program, but SSA conversion is
// coming
use std::collections::HashMap;

/// A MIL program.
///
/// This is logically constituted of a linear sequence of instructions, each with:
///  - a destination register (`Reg`);
///  - an operation with its operands/inputs (`Insn`);
///  - a corresponding address in the original machine code (`u64`).
///
/// By convention, the entry point of the program is always at index 0.
#[derive(Debug)]
pub struct Program {
    insns: Vec<Insn>,
    dests: Vec<Reg>,
    addrs: Vec<u64>,
    // TODO More specific types
    mil_of_input_addr: HashMap<u64, Index>,
}

/// Register ID
///
/// The language admits as many registers as a u16 can represent (2**16). They're
/// abstract, so we don't pay for them!
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reg(pub u16);

pub type Index = usize;

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
    Ret(Reg),
    Jmp(Reg),
    JmpK(u64),
    JmpIfK { cond: Reg, target: u64 },

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

impl Insn {
    pub fn is_block_ender(&self) -> bool {
        matches!(self, Insn::JmpIfK { .. } | Insn::Jmp(_) | Insn::Ret(_))
    }

    fn dump(&self) {
        match self {
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
            Insn::Ret(x) => print!("{:8} r{}", "ret", x.0),
            Insn::Jmp(x) => print!("{:8} r{}", "jmp", x.0),
            Insn::JmpIfK { cond, target } => {
                print!("{:8} r{},0x{:x}", "jmp.if", cond.0, target)
            }
            Insn::JmpK(target) => print!("{:8} 0x{:x}", "jmp", target),

            Insn::OverflowOf(x) => print!("{:8} r{}", "overflow", x.0),
            Insn::CarryOf(x) => print!("{:8} r{}", "carry", x.0),
            Insn::SignOf(x) => print!("{:8} r{}", "sign", x.0),
            Insn::IsZero(x) => print!("{:8} r{}", "is0", x.0),
            Insn::Parity(x) => print!("{:8} r{}", "parity", x.0),

            Insn::Undefined => print!("undef"),
        }
    }
}

impl Program {
    pub fn dump(&self) {
        println!("program  {} instrs", self.insns.len());
        let mut last_addr = 0;
        for (ndx, ((insn, dest), addr)) in self
            .insns
            .iter()
            .zip(self.dests.iter())
            .zip(self.addrs.iter())
            .enumerate()
        {
            if last_addr != *addr {
                println!("0x{:x}:", addr);
                last_addr = *addr;
            }
            print!("{:5} r{:<3} <- ", ndx, dest.0);
            insn.dump();

            println!();
        }
    }

    #[inline(always)]
    pub fn get(&self, ndx: Index) -> Option<InsnView> {
        let insn = self.insns.get(ndx)?;
        let dest = *self.dests.get(ndx).unwrap();
        let addr = *self.addrs.get(ndx).unwrap();
        Some(InsnView { insn, dest, addr })
    }

    pub fn index_of_addr(&self, addr: u64) -> Option<Index> {
        self.mil_of_input_addr.get(&addr).copied()
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.insns.len()
    }

    #[inline(always)]
    pub(crate) fn iter(&self) -> impl Iterator<Item = InsnView> {
        (0..self.len()).map(|ndx| self.get(ndx).unwrap())
    }
}

pub struct InsnView<'a> {
    pub insn: &'a Insn,
    pub dest: Reg,
    pub addr: u64,
}

// will be mostly useful to keep origin info later
pub struct ProgramBuilder {
    insns: Vec<Insn>,
    dests: Vec<Reg>,
    addrs: Vec<u64>,
    cur_input_addr: u64,
}

impl ProgramBuilder {
    pub fn new() -> Self {
        Self {
            insns: Vec::new(),
            dests: Vec::new(),
            addrs: Vec::new(),
            cur_input_addr: 0,
        }
    }

    pub fn push(&mut self, dest: Reg, insn: Insn) -> Reg {
        self.dests.push(dest);
        self.insns.push(insn);
        self.addrs.push(self.cur_input_addr);
        dest
    }

    /// Associate the instructions emitted via the following calls to `emit_*` to the given
    /// address.
    ///
    /// This establishes a correspondence between the output MIL code and the input machine
    /// code, which is then used to resolve jumps, etc.
    pub fn set_input_addr(&mut self, addr: u64) {
        self.cur_input_addr = addr;
    }

    pub fn build(self) -> Program {
        let Self {
            insns,
            dests,
            addrs,
            ..
        } = self;

        assert_eq!(dests.len(), insns.len());
        assert_eq!(dests.len(), addrs.len());

        // addrs is input addr of mil addr;
        // we also need mil addr of input addr to resolve jumps
        let mil_of_input_addr = {
            let mut map = HashMap::new();
            let mut addrs = addrs.iter().enumerate();
            let mut last_addr = u64::MAX;
            while let Some((ndx, &addr)) = addrs.next() {
                if addr != last_addr {
                    map.insert(addr, ndx);
                    last_addr = addr;
                }
            }
            map
        };

        Program {
            insns,
            dests,
            addrs,
            mil_of_input_addr,
        }
    }
}
