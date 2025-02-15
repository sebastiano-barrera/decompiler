use enum_assoc::Assoc;
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
#[derive(Clone)]
pub struct Program {
    insns: Vec<Cell<Insn>>,
    dests: Vec<Cell<Reg>>,
    addrs: Vec<u64>,
    reg_count: Index,

    // TODO More specific types
    // kept even if dead, because we will still want to trace each MIL
    // instruction back to the original machine code / assembly
    #[allow(dead_code)]
    mil_of_input_addr: HashMap<u64, Index>,

    anc_types: HashMap<AncestralName, RegType>,
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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RegType {
    Effect,
    Bytes(usize),
    Bool,
}
impl RegType {
    pub(crate) fn bytes_size(&self) -> Option<usize> {
        match self {
            RegType::Effect => None,
            RegType::Bool => None,
            RegType::Bytes(sz) => Some(*sz),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Assoc)]
#[func(pub fn has_side_effects(&self) -> bool { false })]
#[allow(clippy::upper_case_acronyms)]
pub enum Insn {
    True,
    False,
    Const1(i8),
    Const2(i16),
    Const4(i32),
    Const8(i64),
    Get(Reg),

    Part {
        src: Reg,
        offset: u8,
        size: u8,
    },
    Concat {
        lo: Reg,
        hi: Reg,
    },

    StructGet8 {
        struct_value: Reg,
        offset: u8,
    },
    Widen1_2(Reg),
    Widen1_4(Reg),
    Widen1_8(Reg),
    Widen2_4(Reg),
    Widen2_8(Reg),
    Widen4_8(Reg),

    Arith1(ArithOp, Reg, Reg),
    Arith2(ArithOp, Reg, Reg),
    Arith4(ArithOp, Reg, Reg),
    Arith8(ArithOp, Reg, Reg),

    ArithK1(ArithOp, Reg, i64),
    ArithK2(ArithOp, Reg, i64),
    ArithK4(ArithOp, Reg, i64),
    ArithK8(ArithOp, Reg, i64),

    Cmp(CmpOp, Reg, Reg),
    Bool(BoolOp, Reg, Reg),
    Not(Reg),

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
    #[assoc(has_side_effects = true)]
    Call(Reg),
    #[assoc(has_side_effects = true)]
    CArg(Reg),
    #[assoc(has_side_effects = true)]
    Ret(Reg),
    #[allow(dead_code)]
    #[assoc(has_side_effects = true)]
    JmpInd(Reg),
    #[assoc(has_side_effects = true)]
    Jmp(Index),
    #[assoc(has_side_effects = true)]
    JmpIf {
        cond: Reg,
        target: Index,
    },
    #[assoc(has_side_effects = true)]
    JmpExt(u64),
    #[assoc(has_side_effects = true)]
    JmpExtIf {
        cond: Reg,
        addr: u64,
    },

    #[allow(clippy::upper_case_acronyms)]
    #[assoc(has_side_effects = true)]
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
    Ancestral(AncestralName),

    /// Phi node.
    ///
    /// Exists purely to give the phi node an index that the rest of the program can refer to (in
    /// SSA).
    Phi {
        size: usize,
    },
    PhiBool,

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

/// Binary comparison operators. Inputs are integers; the output is a boolean.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CmpOp {
    EQ,
    LT,
}

/// Binary boolean operators. Inputs and outputs are booleans.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BoolOp {
    Or,
    And,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArithOp {
    Add,
    Sub,
    #[allow(dead_code)]
    Mul,
    Shl,
    BitXor,
    BitAnd,
    BitOr,
}

/// The "name" (identifier) of an "ancestral" value, i.e. a value in MIL code
/// that represents the pre-existing value of a machine register at the time
/// the function started execution.  Mostly useful to allow the decompilation to
/// proceed forward even when somehting is out of place.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct AncestralName(&'static str);

impl AncestralName {
    pub const fn new(name: &'static str) -> Self {
        AncestralName(name)
    }

    pub fn name(&self) -> &'static str {
        self.0
    }
}

#[macro_export]
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
            Insn::False
            | Insn::True
            | Insn::Const1(_)
            | Insn::Const2(_)
            | Insn::Const4(_)
            | Insn::Const8(_)
            | Insn::JmpExt(_)
            | Insn::Jmp(_)
            | Insn::TODO(_)
            | Insn::Undefined
            | Insn::Phi { size: _ }
            | Insn::PhiBool
            | Insn::Ancestral(_) => [None, None],

            Insn::Part {
                src: reg,
                offset: _,
                size: _,
            }
            | Insn::Get(reg)
            | Insn::ArithK1(_, reg, _)
            | Insn::ArithK2(_, reg, _)
            | Insn::ArithK4(_, reg, _)
            | Insn::ArithK8(_, reg, _)
            | Insn::Widen1_2(reg)
            | Insn::Widen1_4(reg)
            | Insn::Widen1_8(reg)
            | Insn::Widen2_4(reg)
            | Insn::Widen2_8(reg)
            | Insn::Widen4_8(reg)
            | Insn::Not(reg)
            | Insn::Ret(reg)
            | Insn::JmpInd(reg)
            | Insn::JmpExtIf { cond: reg, addr: _ }
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
            | Insn::PhiArg(reg)
            | Insn::StructGet8 {
                struct_value: reg,
                offset: _,
            } => [Some(reg), None],

            Insn::Concat { lo: a, hi: b }
            | Insn::Arith1(_, a, b)
            | Insn::Arith2(_, a, b)
            | Insn::Arith4(_, a, b)
            | Insn::Arith8(_, a, b)
            | Insn::Cmp(_, a, b)
            | Insn::Bool(_, a, b)
            | Insn::StoreMem(a, b) => [Some(a), Some(b)],
        }
    }

    pub fn input_regs_iter<'s>(&'s self) -> impl 's + Iterator<Item = Reg> {
        self.input_regs().into_iter().flatten().copied()
    }
    pub fn input_regs_iter_mut<'s>(&'s mut self) -> impl 's + Iterator<Item = &'s mut Reg> {
        self.input_regs_mut().into_iter().flatten()
    }
    pub fn input_regs(&self) -> [Option<&Reg>; 2] {
        match self {
            Insn::False
            | Insn::True
            | Insn::Const1(_)
            | Insn::Const2(_)
            | Insn::Const4(_)
            | Insn::Const8(_)
            | Insn::JmpExt(_)
            | Insn::Jmp(_)
            | Insn::TODO(_)
            | Insn::Undefined
            | Insn::Phi { size: _ }
            | Insn::PhiBool
            | Insn::Ancestral(_) => [None, None],

            Insn::Part {
                src: reg,
                offset: _,
                size: _,
            }
            | Insn::Get(reg)
            | Insn::ArithK1(_, reg, _)
            | Insn::ArithK2(_, reg, _)
            | Insn::ArithK4(_, reg, _)
            | Insn::ArithK8(_, reg, _)
            | Insn::Widen1_2(reg)
            | Insn::Widen1_4(reg)
            | Insn::Widen1_8(reg)
            | Insn::Widen2_4(reg)
            | Insn::Widen2_8(reg)
            | Insn::Widen4_8(reg)
            | Insn::Not(reg)
            | Insn::Ret(reg)
            | Insn::JmpInd(reg)
            | Insn::JmpExtIf { cond: reg, addr: _ }
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
            | Insn::PhiArg(reg)
            | Insn::StructGet8 {
                struct_value: reg,
                offset: _,
            } => [Some(reg), None],

            Insn::Concat { lo: b, hi: a }
            | Insn::Arith1(_, a, b)
            | Insn::Arith2(_, a, b)
            | Insn::Arith4(_, a, b)
            | Insn::Arith8(_, a, b)
            | Insn::Cmp(_, a, b)
            | Insn::Bool(_, a, b)
            | Insn::StoreMem(a, b) => [Some(a), Some(b)],
        }
    }

    #[inline(always)]
    pub fn is_phi(&self) -> bool {
        matches!(
            self,
            Insn::Phi { size: 1 }
                | Insn::Phi { size: 2 }
                | Insn::Phi { size: 4 }
                | Insn::Phi { size: 8 }
                | Insn::PhiBool
        )
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
        let phi_insn = self.insns[ndx].get();
        assert!(phi_insn.is_phi());
        self.insns
            .iter()
            .skip(ndx + 1)
            .map_while(|i| match i.get() {
                Insn::PhiArg(arg) => Some(arg),
                _ => None,
            })
    }

    pub fn map_phi_args(&self, ndx: Index, f: impl Fn(Reg) -> Reg) {
        let ndx = ndx as usize;
        let phi_insn = self.insns[ndx].get();
        assert!(phi_insn.is_phi());
        self.insns.iter().skip(ndx + 1).for_each(|i| {
            if let Insn::PhiArg(arg) = i.get() {
                i.set(Insn::PhiArg(f(arg)))
            }
        })
    }

    #[inline(always)]
    pub fn reg_count(&self) -> Index {
        self.reg_count
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

    pub fn ancestor_type(&self, anc_name: AncestralName) -> Option<RegType> {
        self.anc_types.get(&anc_name).copied()
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
    anc_types: HashMap<AncestralName, RegType>,
}

impl ProgramBuilder {
    pub fn new() -> Self {
        Self {
            insns: Vec::new(),
            dests: Vec::new(),
            addrs: Vec::new(),
            cur_input_addr: 0,
            anc_types: HashMap::new(),
        }
    }

    pub fn push(&mut self, dest: Reg, insn: Insn) -> Reg {
        assert!(!insn.is_phi());
        self.dests.push(Cell::new(dest));
        self.insns.push(Cell::new(insn));
        self.addrs.push(self.cur_input_addr);
        dest
    }

    pub fn set_ancestral_type(&mut self, anc_name: AncestralName, typ: RegType) {
        self.anc_types.insert(anc_name, typ);
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
                Insn::JmpExtIf { cond, addr: target } => {
                    if let Some(ndx) = mil_of_input_addr.get(&target) {
                        insn.set(Insn::JmpIf { cond, target: *ndx });
                    }
                }
                _ => {}
            }
        }

        let reg_count = {
            let max_dest = dests
                .iter()
                .map(|reg| reg.get().reg_index())
                .max()
                .unwrap_or(0);
            let max_input = insns
                .iter()
                .flat_map(|insn| {
                    insn.get()
                        .input_regs_iter()
                        .map(|reg| reg.reg_index())
                        .max()
                })
                .max()
                .unwrap_or(0);
            1 + max_dest.max(max_input)
        };

        Program {
            insns,
            dests,
            addrs,
            reg_count,
            mil_of_input_addr,
            anc_types: self.anc_types,
        }
    }
}
