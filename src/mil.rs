/// Machine-Independent Language
// TODO This currently only represents the pre-SSA version of the program, but SSA conversion is
// coming
use std::{cell::Cell, collections::HashMap, ops::Range};

/// A MIL program.
///
/// This is logically constituted of a linear sequence of instructions, each with:
///  - a destination register (`Reg`);
///  - an operation with its operands/inputs (`Insn`);
///  - a corresponding address in the original machine code (`u64`).
///
/// By convention, the entry point of the program is always at index 0.
pub struct Program {
    insns: Vec<Cell<Insn>>,
    dests: Vec<Cell<Reg>>,
    addrs: Vec<u64>,
    // TODO More specific types
    // kept even if dead, because we will still want to trace each MIL
    // instruction back to the original machine code / assembly
    #[allow(dead_code)]
    mil_of_input_addr: HashMap<u64, Index>,
}

/// Register ID
///
/// The language admits as many registers as a u16 can represent (2**16). They're
/// abstract, so we don't pay for them!
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reg(pub u16);

impl Reg {
    pub fn reg_index(&self) -> u16 {
        self.0
    }
}
impl std::fmt::Debug for Reg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "r{}", self.0)
    }
}

pub type Index = u16;

#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
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
    #[allow(dead_code)]
    Mul(Reg, Reg),
    MulK(Reg, i64),
    Shl(Reg, Reg),
    BitAnd(Reg, Reg),
    BitOr(Reg, Reg),
    Eq(Reg, Reg),
    Not(Reg),
    LT(Reg, Reg),

    // call args are represented by a sequence of adjacent CArg instructions,
    // immediately following the "main" Call insn:
    //  r0 <- [compute callee]
    //  r1 <- [compute arg 0]
    //  r2 <- [compute arg 1]
    //  r3 <- call r0
    //  r4 <- carg r1
    //  r5 <- carg r2
    //  r6 <- carg r3
    // destination vreg r3 is for the return value. r4, r5, r6 are entirely
    // fictitious, they don't correspond to any value.
    Call(Reg),
    CArg(Reg),
    Ret(Reg),
    #[allow(dead_code)]
    JmpI(Reg),
    Jmp(Index),
    JmpExt(u64),
    JmpIf {
        cond: Reg,
        target: Index,
    },
    JmpExtIf {
        cond: Reg,
        target: u64,
    },

    #[allow(clippy::upper_case_acronyms)]
    TODO(&'static str),

    LoadMem1(Reg),
    LoadMem2(Reg),
    LoadMem4(Reg),
    LoadMem8(Reg),
    StoreMem1(Reg, Reg),
    StoreMem2(Reg, Reg),
    StoreMem4(Reg, Reg),
    StoreMem8(Reg, Reg),

    OverflowOf(Reg),
    CarryOf(Reg),
    SignOf(Reg),
    IsZero(Reg),
    Parity(Reg),

    Undefined,
    Ancestral(AncestralName),

    /// Phi node.
    ///
    /// Exists purely to give the phi node an index that the rest of the program can refer to (in
    /// SSA).
    Phi,

    // argument to a phi node
    //
    //   - args to the same phi node are all adjacent, forming a 'group'
    //      - always the same number as there are predecessors.
    //      - their order matches the order of the block's predecessors in the CFG.
    //   - all the groups in the same block are also adjacent
    // e.g.  3 phis on a block with 2 predecessors: B5, B7
    //   B3:
    //     PhiArg r2   ;; --+-- phi0 <- B5:r2  B7:r8
    //     PhiArg r8   ;; --'
    //     PhiArg r11  ;; --+-- phi1 <- B5:r11 B7:r29
    //     PhiArg r29  ;; --'
    //     PhiArg r93  ;; --+-- phi2 <- B5:r93 B7:r332
    //     PhiArg r332 ;; --'
    //     ... ;; normal insns
    PhiArg(Reg),
}

/// The "name" (identifier) of an "ancestral" value, i.e. a value in MIL code
/// that represents the pre-existing value of a machine register at the time
/// the function started execution.  Mostly useful to allow the decompilation to
/// proceed forward even when somehting is out of place.
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct AncestralName(&'static str);

impl AncestralName {
    pub const fn new(name: &'static str) -> Self {
        AncestralName(name)
    }

    pub fn name(&self) -> &'static str {
        self.0
    }
}

macro_rules! define_ancestral_name {
    ($name:ident, $value:literal) => {
        pub const $name: $crate::mil::AncestralName = $crate::mil::AncestralName::new($value);
    };
}

define_ancestral_name!(ANC_STACK_BOTTOM, "stack_bottom");

impl Insn {
    // TODO There must be some macro magic to generate these two functions
    pub fn input_regs_mut(&mut self) -> [Option<&mut Reg>; 2] {
        match self {
            Insn::Const1(_)
            | Insn::Const2(_)
            | Insn::Const4(_)
            | Insn::Const8(_)
            | Insn::JmpExt(_)
            | Insn::Jmp(_)
            | Insn::TODO(_)
            | Insn::Undefined
            | Insn::Phi
            | Insn::Ancestral(_) => [None, None],

            Insn::L1(reg)
            | Insn::L2(reg)
            | Insn::L4(reg)
            | Insn::Get(reg)
            | Insn::AddK(reg, _)
            | Insn::MulK(reg, _)
            | Insn::Not(reg)
            | Insn::Ret(reg)
            | Insn::JmpI(reg)
            | Insn::JmpExtIf {
                cond: reg,
                target: _,
            }
            | Insn::JmpIf {
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
            | Insn::Parity(reg)
            | Insn::Call(reg)
            | Insn::CArg(reg)
            | Insn::PhiArg(reg) => [Some(reg), None],

            Insn::WithL1(a, b)
            | Insn::WithL2(a, b)
            | Insn::WithL4(a, b)
            | Insn::Add(a, b)
            | Insn::Sub(a, b)
            | Insn::Mul(a, b)
            | Insn::Shl(a, b)
            | Insn::BitAnd(a, b)
            | Insn::BitOr(a, b)
            | Insn::Eq(a, b)
            | Insn::LT(a, b)
            | Insn::StoreMem1(a, b)
            | Insn::StoreMem2(a, b)
            | Insn::StoreMem4(a, b)
            | Insn::StoreMem8(a, b) => [Some(a), Some(b)],
        }
    }

    pub fn input_regs_iter<'s>(&'s self) -> impl 's + Iterator<Item = Reg> {
        self.input_regs().into_iter().flatten().copied()
    }
    pub fn input_regs(&self) -> [Option<&Reg>; 2] {
        match self {
            Insn::Const1(_)
            | Insn::Const2(_)
            | Insn::Const4(_)
            | Insn::Const8(_)
            | Insn::JmpExt(_)
            | Insn::Jmp(_)
            | Insn::TODO(_)
            | Insn::Undefined
            | Insn::Phi
            | Insn::Ancestral(_) => [None, None],

            Insn::L1(reg)
            | Insn::L2(reg)
            | Insn::L4(reg)
            | Insn::Get(reg)
            | Insn::AddK(reg, _)
            | Insn::MulK(reg, _)
            | Insn::Not(reg)
            | Insn::Ret(reg)
            | Insn::JmpI(reg)
            | Insn::JmpIf {
                cond: reg,
                target: _,
            }
            | Insn::JmpExtIf {
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
            | Insn::Parity(reg)
            | Insn::Call(reg)
            | Insn::CArg(reg)
            | Insn::PhiArg(reg) => [Some(reg), None],

            Insn::WithL1(a, b)
            | Insn::WithL2(a, b)
            | Insn::WithL4(a, b)
            | Insn::Add(a, b)
            | Insn::Sub(a, b)
            | Insn::Mul(a, b)
            | Insn::Shl(a, b)
            | Insn::BitAnd(a, b)
            | Insn::BitOr(a, b)
            | Insn::Eq(a, b)
            | Insn::LT(a, b)
            | Insn::StoreMem1(a, b)
            | Insn::StoreMem2(a, b)
            | Insn::StoreMem4(a, b)
            | Insn::StoreMem8(a, b) => [Some(a), Some(b)],
        }
    }

    pub fn has_side_effects(&self) -> bool {
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
            | Insn::MulK(_, _)
            | Insn::Shl(_, _)
            | Insn::BitAnd(_, _)
            | Insn::BitOr(_, _)
            | Insn::Eq(_, _)
            | Insn::LT(_, _)
            | Insn::Not(_)
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
            | Insn::Ancestral(_)
            | Insn::Phi
            | Insn::PhiArg { .. } => false,

            Insn::Call { .. }
            | Insn::CArg { .. }
            | Insn::Ret(_)
            | Insn::JmpI(_)
            | Insn::JmpExt(_)
            | Insn::Jmp(_)
            | Insn::JmpExtIf { .. }
            | Insn::JmpIf { .. }
            | Insn::TODO(_)
            | Insn::StoreMem1(_, _)
            | Insn::StoreMem2(_, _)
            | Insn::StoreMem4(_, _)
            | Insn::StoreMem8(_, _) => true,
        }
    }
}

impl std::fmt::Debug for Insn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Insn::Const1(val) => write!(f, "{:8} {} (0x{:x})", "const1", *val as i64, val),
            Insn::Const2(val) => write!(f, "{:8} {} (0x{:x})", "const2", *val as i64, val),
            Insn::Const4(val) => write!(f, "{:8} {} (0x{:x})", "const4", *val as i64, val),
            Insn::Const8(val) => write!(f, "{:8} {} (0x{:x})", "const8", *val as i64, val),
            Insn::L1(x) => write!(f, "{:8} {:?}", "l1", x),
            Insn::L2(x) => write!(f, "{:8} {:?}", "l2", x),
            Insn::L4(x) => write!(f, "{:8} {:?}", "l4", x),
            Insn::Get(x) => write!(f, "{:8} {:?}", "get", x),
            Insn::WithL1(full, part) => write!(f, "{:8} {:?} ← {:?}", "with.l1", full, part),
            Insn::WithL2(full, part) => write!(f, "{:8} {:?} ← {:?}", "with.l2", full, part),
            Insn::WithL4(full, part) => write!(f, "{:8} {:?} ← {:?}", "with.l4", full, part),

            Insn::Add(a, b) => write!(f, "{:8} {:?},{:?}", "add", a, b),
            Insn::AddK(a, b) => write!(f, "{:8} {:?},{}", "addk", a, *b),
            Insn::Sub(a, b) => write!(f, "{:8} {:?},{:?}", "sub", a, b),
            Insn::Mul(a, b) => write!(f, "{:8} {:?},{:?}", "mul", a, b),
            Insn::MulK(a, b) => write!(f, "{:8} {:?},0x{:x}", "mul", a, b),
            Insn::Shl(value, bits_count) => write!(f, "{:8} {:?},{:?}", "shl", value, bits_count),
            Insn::BitAnd(a, b) => write!(f, "{:8} {:?},{:?}", "and", a, b),
            Insn::BitOr(a, b) => write!(f, "{:8} {:?},{:?}", "or", a, b),
            Insn::Eq(a, b) => write!(f, "{:8} {:?},{:?}", "eq", a, b),
            Insn::LT(a, b) => write!(f, "{:8} {:?},{:?}", "<", a, b),
            Insn::Not(x) => write!(f, "{:8} {:?}", "not", x),

            Insn::LoadMem1(addr) => write!(f, "{:8} addr:{:?}", "loadm1", addr),
            Insn::LoadMem2(addr) => write!(f, "{:8} addr:{:?}", "loadm2", addr),
            Insn::LoadMem4(addr) => write!(f, "{:8} addr:{:?}", "loadm4", addr),
            Insn::LoadMem8(addr) => write!(f, "{:8} addr:{:?}", "loadm8", addr),

            Insn::StoreMem1(addr, val) => write!(f, "{:8} *{:?} ← {:?}", "storem1", addr, val),
            Insn::StoreMem2(addr, val) => write!(f, "{:8} *{:?} ← {:?}", "storem2", addr, val),
            Insn::StoreMem4(addr, val) => write!(f, "{:8} *{:?} ← {:?}", "storem4", addr, val),
            Insn::StoreMem8(addr, val) => write!(f, "{:8} *{:?} ← {:?}", "storem8", addr, val),
            Insn::TODO(msg) => write!(f, "{:8} {}", "TODO", msg),

            Insn::Call(callee) => write!(f, "{:8} {:?}", "call", callee),
            Insn::CArg(value) => write!(f, "{:8} {:?}", "carg", value),
            Insn::Ret(x) => write!(f, "{:8} {:?}", "ret", x),
            Insn::JmpI(x) => write!(f, "{:8} *{:?}", "jmp", x),
            Insn::Jmp(x) => write!(f, "{:8} {:?}", "jmp", x),
            Insn::JmpExt(target) => write!(f, "{:8} 0x{:x} extern", "jmp", target),
            Insn::JmpExtIf { cond, target } => {
                write!(f, "{:8} {:?},0x{:x} extern", "jmp.if", cond, target)
            }
            Insn::JmpIf { cond, target } => write!(f, "{:8} {:?},{}", "jmp.if", cond, target),

            Insn::OverflowOf(x) => write!(f, "{:8} {:?}", "overflow", x),
            Insn::CarryOf(x) => write!(f, "{:8} {:?}", "carry", x),
            Insn::SignOf(x) => write!(f, "{:8} {:?}", "sign", x),
            Insn::IsZero(x) => write!(f, "{:8} {:?}", "is0", x),
            Insn::Parity(x) => write!(f, "{:8} {:?}", "parity", x),

            Insn::Undefined => write!(f, "undef"),
            Insn::Ancestral(anc) => write!(f, "#pre:{}", anc.name()),

            Insn::Phi => write!(f, "phi"),
            Insn::PhiArg(reg) => {
                write!(f, "{:8} {:?}", "phiarg", reg)
            }
        }
    }
}

impl std::fmt::Debug for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "program  {} instrs", self.insns.len())?;
        let mut last_addr = 0;
        for (ndx, ((insn, dest), addr)) in self
            .insns
            .iter()
            .zip(self.dests.iter())
            .zip(self.addrs.iter())
            .enumerate()
        {
            if last_addr != *addr {
                writeln!(f, "0x{:x}:", addr)?;
                last_addr = *addr;
            }
            writeln!(f, "{:5} {:?} <- {:?}", ndx, dest.get(), insn.get())?;
        }
        Ok(())
    }
}

fn range_conv<T, U: From<T>>(range: Range<T>) -> Range<U> {
    Range {
        start: range.start.into(),
        end: range.end.into(),
    }
}

impl Program {
    #[inline(always)]
    pub fn get(&self, ndx: Index) -> Option<InsnView> {
        let ndx = ndx as usize;
        let insn = &self.insns[ndx];
        let dest = &self.dests[ndx];
        let addr = self.addrs[ndx];
        Some(InsnView { insn, dest, addr })
    }

    pub fn slice(&self, ndxr: Range<Index>) -> Option<InsnSlice> {
        let insn = self.insns.get(range_conv(ndxr.clone()))?;
        let dest = &self.dests[range_conv(ndxr)];
        Some(InsnSlice {
            insns: insn,
            dests: dest,
        })
    }

    pub fn get_call_args(&self, ndx: Index) -> impl '_ + Iterator<Item = Reg> {
        let ndx = ndx as usize;
        assert!(matches!(self.insns[ndx].get(), Insn::Call(_)));
        self.insns
            .iter()
            .skip(ndx + 1)
            .map_while(|i| match i.get() {
                Insn::CArg(arg) => Some(arg),
                _ => None,
            })
    }

    pub fn get_phi_args(&self, ndx: Index) -> impl '_ + Iterator<Item = Reg> {
        let ndx = ndx as usize;
        assert!(matches!(self.insns[ndx].get(), Insn::Phi));
        self.insns
            .iter()
            .skip(ndx + 1)
            .map_while(|i| match i.get() {
                Insn::PhiArg(arg) => Some(arg),
                _ => None,
            })
    }

    #[inline(always)]
    pub fn len(&self) -> Index {
        self.insns.len().try_into().unwrap()
    }

    #[inline(always)]
    pub(crate) fn iter(&self) -> impl ExactSizeIterator<Item = InsnView> {
        (0..self.len()).map(|ndx| self.get(ndx).unwrap())
    }

    pub fn push(&mut self, dest: Reg, insn: Insn) -> Index {
        let index = self.insns.len().try_into().unwrap();
        self.insns.push(Cell::new(insn));
        self.dests.push(Cell::new(dest));
        self.addrs.push(u64::MAX);
        index
    }
}

pub struct InsnView<'a> {
    pub insn: &'a Cell<Insn>,
    pub dest: &'a Cell<Reg>,
    pub addr: u64,
}

#[derive(Clone, Copy)]
pub struct InsnSlice<'a> {
    pub insns: &'a [Cell<Insn>],
    pub dests: &'a [Cell<Reg>],
}
impl<'a> InsnSlice<'a> {
    pub fn iter<'s>(
        &'s self,
    ) -> impl 's + DoubleEndedIterator<Item = (&'a Cell<Reg>, &'a Cell<Insn>)> {
        self.dests.iter().zip(self.insns.iter())
    }
    pub fn iter_copied<'s>(&'s self) -> impl 's + DoubleEndedIterator<Item = (Reg, Insn)> {
        self.iter().map(|(d, r)| (d.get(), r.get()))
    }
}

// will be mostly useful to keep origin info later
pub struct ProgramBuilder {
    insns: Vec<Cell<Insn>>,
    dests: Vec<Cell<Reg>>,
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
        self.dests.push(Cell::new(dest));
        self.insns.push(Cell::new(insn));
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
            let mut map: HashMap<u64, Index> = HashMap::new();
            let mut last_addr = u64::MAX;
            for (ndx, &addr) in addrs.iter().enumerate() {
                if addr != last_addr {
                    let ndx = ndx.try_into().unwrap();
                    map.insert(addr, ndx);
                    last_addr = addr;
                }
            }
            map
        };

        for insn in &insns {
            match insn.get() {
                Insn::JmpExt(addr) => {
                    if let Some(ndx) = mil_of_input_addr.get(&addr) {
                        insn.set(Insn::Jmp(*ndx));
                    }
                }
                Insn::JmpExtIf { cond, target } => {
                    if let Some(ndx) = mil_of_input_addr.get(&target) {
                        insn.set(Insn::JmpIf { cond, target: *ndx });
                    }
                }
                _ => {}
            }
        }

        Program {
            insns,
            dests,
            addrs,
            mil_of_input_addr,
        }
    }
}
