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
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Reg {
    Nor(u16),
    Phi(u16),
    /// Represents an undefined value.  Only allowed as a source, and only used in SSA.
    Und,
}

impl Reg {
    pub fn as_nor(&self) -> Option<u16> {
        match self {
            Reg::Nor(id) => Some(*id),
            _ => None,
        }
    }
    pub fn as_phi(&self) -> Option<u16> {
        match self {
            Reg::Phi(id) => Some(*id),
            _ => None,
        }
    }
}
impl std::fmt::Debug for Reg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Reg::Nor(ndx) => write!(f, "r{}", ndx),
            Reg::Phi(ndx) => write!(f, "ɸ{}", ndx),
            Reg::Und => write!(f, "<undefined>"),
        }
    }
}

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
    Ancestral(Ancestral),
}

#[derive(Clone, Copy, Debug)]
pub enum Ancestral {
    StackBot,
}

impl Insn {
    pub fn is_block_ender(&self) -> bool {
        matches!(self, Insn::JmpIfK { .. } | Insn::Jmp(_) | Insn::Ret(_))
    }

    // TODO There must be some macro magic to generate these two functions
    pub fn input_regs_mut(&mut self) -> [Option<&mut Reg>; 2] {
        match self {
            Insn::Const1(_)
            | Insn::Const2(_)
            | Insn::Const4(_)
            | Insn::Const8(_)
            | Insn::CArgEnd
            | Insn::JmpK(_)
            | Insn::TODO(_)
            | Insn::Undefined
            | Insn::Ancestral(_) => [None, None],

            Insn::L1(reg)
            | Insn::L2(reg)
            | Insn::L4(reg)
            | Insn::Get(reg)
            | Insn::AddK(reg, _)
            | Insn::MulK32(reg, _)
            | Insn::Ret(reg)
            | Insn::Jmp(reg)
            | Insn::JmpIfK {
                cond: reg,
                target: _,
            }
            | Insn::LoadMem1(reg)
            | Insn::LoadMem2(reg)
            | Insn::LoadMem4(reg)
            | Insn::LoadMem8(reg)
            | Insn::OverflowOf(reg)
            | Insn::CarryOf(reg)
            | Insn::SignOf(reg)
            | Insn::IsZero(reg)
            | Insn::Parity(reg) => [Some(reg), None],

            Insn::WithL1(a, b)
            | Insn::WithL2(a, b)
            | Insn::WithL4(a, b)
            | Insn::Add(a, b)
            | Insn::Sub(a, b)
            | Insn::Mul(a, b)
            | Insn::Shl(a, b)
            | Insn::BitAnd(a, b)
            | Insn::BitOr(a, b)
            | Insn::Call { callee: a, arg0: b }
            | Insn::CArg { value: a, prev: b }
            | Insn::StoreMem(a, b) => [Some(a), Some(b)],
        }
    }

    pub fn input_regs(&self) -> [Option<&Reg>; 2] {
        match self {
            Insn::Const1(_)
            | Insn::Const2(_)
            | Insn::Const4(_)
            | Insn::Const8(_)
            | Insn::CArgEnd
            | Insn::JmpK(_)
            | Insn::TODO(_)
            | Insn::Undefined
            | Insn::Ancestral(_) => [None, None],

            Insn::L1(reg)
            | Insn::L2(reg)
            | Insn::L4(reg)
            | Insn::Get(reg)
            | Insn::AddK(reg, _)
            | Insn::MulK32(reg, _)
            | Insn::Ret(reg)
            | Insn::Jmp(reg)
            | Insn::JmpIfK {
                cond: reg,
                target: _,
            }
            | Insn::LoadMem1(reg)
            | Insn::LoadMem2(reg)
            | Insn::LoadMem4(reg)
            | Insn::LoadMem8(reg)
            | Insn::OverflowOf(reg)
            | Insn::CarryOf(reg)
            | Insn::SignOf(reg)
            | Insn::IsZero(reg)
            | Insn::Parity(reg) => [Some(reg), None],

            Insn::WithL1(a, b)
            | Insn::WithL2(a, b)
            | Insn::WithL4(a, b)
            | Insn::Add(a, b)
            | Insn::Sub(a, b)
            | Insn::Mul(a, b)
            | Insn::Shl(a, b)
            | Insn::BitAnd(a, b)
            | Insn::BitOr(a, b)
            | Insn::Call { callee: a, arg0: b }
            | Insn::CArg { value: a, prev: b }
            | Insn::StoreMem(a, b) => [Some(a), Some(b)],
        }
    }

    pub fn dump(&self) {
        match self {
            Insn::Const1(val) => print!("{:8} {} (0x{:x})", "const1", *val as i64, val),
            Insn::Const2(val) => print!("{:8} {} (0x{:x})", "const2", *val as i64, val),
            Insn::Const4(val) => print!("{:8} {} (0x{:x})", "const4", *val as i64, val),
            Insn::Const8(val) => print!("{:8} {} (0x{:x})", "const8", *val as i64, val),
            Insn::L1(x) => print!("{:8} {:?}", "l1", x),
            Insn::L2(x) => print!("{:8} {:?}", "l2", x),
            Insn::L4(x) => print!("{:8} {:?}", "l4", x),
            Insn::Get(x) => print!("{:8} {:?}", "get", x),
            Insn::WithL1(full, part) => {
                print!("{:8} {:?} ← {:?}", "with.l1", full, part)
            }
            Insn::WithL2(full, part) => {
                print!("{:8} {:?} ← {:?}", "with.l2", full, part)
            }
            Insn::WithL4(full, part) => {
                print!("{:8} {:?} ← {:?}", "with.l4", full, part)
            }

            Insn::Add(a, b) => print!("{:8} {:?},{:?}", "add", a, b),
            Insn::AddK(a, b) => print!("{:8} {:?},{}", "add", a, *b),
            Insn::Sub(a, b) => print!("{:8} {:?},{:?}", "sub", a, b),
            Insn::Mul(a, b) => print!("{:8} {:?},{:?}", "mul", a, b),
            Insn::MulK32(a, b) => print!("{:8} {:?},0x{:x}", "mul", a, b),
            Insn::Shl(value, bits_count) => {
                print!("{:8} {:?},{:?}", "shl", value, bits_count)
            }
            Insn::BitAnd(a, b) => print!("{:8} {:?},{:?}", "and", a, b),
            Insn::BitOr(a, b) => print!("{:8} {:?},{:?}", "or", a, b),

            Insn::LoadMem1(addr) => print!("{:8} addr:{:?}", "loadm1", addr),
            Insn::LoadMem2(addr) => print!("{:8} addr:{:?}", "loadm2", addr),
            Insn::LoadMem4(addr) => print!("{:8} addr:{:?}", "loadm4", addr),
            Insn::LoadMem8(addr) => print!("{:8} addr:{:?}", "loadm8", addr),

            Insn::StoreMem(addr, val) => {
                print!("{:8} *{:?} ← {:?}", "storem", addr, val)
            }
            Insn::TODO(msg) => print!("{:8} {}", "TODO", msg),

            Insn::Call { callee, arg0 } => {
                print!("{:8} {:?}({:?})", "call", callee, arg0)
            }
            Insn::CArgEnd => print!("cargend"),
            Insn::CArg { value, prev } => {
                print!("{:8} {:?} after {:?}", "carg", value, prev)
            }
            Insn::Ret(x) => print!("{:8} {:?}", "ret", x),
            Insn::Jmp(x) => print!("{:8} {:?}", "jmp", x),
            Insn::JmpIfK { cond, target } => {
                print!("{:8} {:?},0x{:x}", "jmp.if", cond, target)
            }
            Insn::JmpK(target) => print!("{:8} 0x{:x}", "jmp", target),

            Insn::OverflowOf(x) => print!("{:8} {:?}", "overflow", x),
            Insn::CarryOf(x) => print!("{:8} {:?}", "carry", x),
            Insn::SignOf(x) => print!("{:8} {:?}", "sign", x),
            Insn::IsZero(x) => print!("{:8} {:?}", "is0", x),
            Insn::Parity(x) => print!("{:8} {:?}", "parity", x),

            Insn::Undefined => print!("undef"),
            Insn::Ancestral(anc) => match anc {
                Ancestral::StackBot => print!("#stackBottom"),
            },
        }
    }

    pub fn is_control_flow(&self) -> bool {
        match self {
            Insn::Const1(_)
            | Insn::Const2(_)
            | Insn::Const4(_)
            | Insn::Const8(_)
            | Insn::L1(_)
            | Insn::L2(_)
            | Insn::L4(_)
            | Insn::Get(_)
            | Insn::WithL1(_, _)
            | Insn::WithL2(_, _)
            | Insn::WithL4(_, _)
            | Insn::Add(_, _)
            | Insn::AddK(_, _)
            | Insn::Sub(_, _)
            | Insn::Mul(_, _)
            | Insn::MulK32(_, _)
            | Insn::Shl(_, _)
            | Insn::BitAnd(_, _)
            | Insn::BitOr(_, _)
            | Insn::OverflowOf(_)
            | Insn::CarryOf(_)
            | Insn::SignOf(_)
            | Insn::IsZero(_)
            | Insn::Parity(_)
            | Insn::Undefined
            | Insn::LoadMem1(_)
            | Insn::LoadMem2(_)
            | Insn::LoadMem4(_)
            | Insn::LoadMem8(_)
            | Insn::Ancestral(_) => false,

            Insn::Call { .. }
            | Insn::CArgEnd
            | Insn::CArg { .. }
            | Insn::Ret(_)
            | Insn::Jmp(_)
            | Insn::JmpK(_)
            | Insn::JmpIfK { .. }
            | Insn::TODO(_)
            | Insn::StoreMem(_, _) => true,
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
            print!("{:5} {:?} <- ", ndx, dest);
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

    #[inline(always)]
    pub fn get_mut(&mut self, ndx: Index) -> Option<InsnViewMut> {
        let insn = self.insns.get_mut(ndx)?;
        let dest = self.dests.get_mut(ndx).unwrap();
        let addr = *self.addrs.get_mut(ndx).unwrap();
        Some(InsnViewMut { insn, dest, addr })
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

pub struct InsnViewMut<'a> {
    pub insn: &'a mut Insn,
    pub dest: &'a mut Reg,
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
            for (ndx, &addr) in addrs {
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
