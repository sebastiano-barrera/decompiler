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
    Bytes(u8),
    Bool,
}
impl RegType {
    pub(crate) fn bytes_size(&self) -> Option<u8> {
        match self {
            RegType::Effect => None,
            RegType::Bool => None,
            RegType::Bytes(sz) => Some(*sz),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
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
    StructGetMember {
        struct_value: Reg,
        name: &'static str,
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
    Call(Reg),
    CArg(Reg),
    Ret(Reg),
    #[allow(dead_code)]
    JmpInd(Reg),
    Jmp(Index),
    JmpIf {
        cond: Reg,
        target: Index,
    },
    JmpExt(u64),
    JmpExtIf {
        cond: Reg,
        addr: u64,
    },

    #[allow(clippy::upper_case_acronyms)]
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
        size: u8,
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
            }
            | Insn::StructGetMember {
                struct_value: reg,
                name: _,
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
            }
            | Insn::StructGetMember {
                struct_value: reg,
                name: _,
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

    pub fn has_side_effects(&self) -> bool {
        match self {
            Insn::False
            | Insn::True
            | Insn::Const1(_)
            | Insn::Const2(_)
            | Insn::Const4(_)
            | Insn::Const8(_)
            | Insn::Part { .. }
            | Insn::Get(_)
            | Insn::Concat { .. }
            | Insn::Widen1_2(_)
            | Insn::Widen1_4(_)
            | Insn::Widen1_8(_)
            | Insn::Widen2_4(_)
            | Insn::Widen2_8(_)
            | Insn::Widen4_8(_)
            | Insn::Arith1(_, _, _)
            | Insn::Arith2(_, _, _)
            | Insn::Arith4(_, _, _)
            | Insn::Arith8(_, _, _)
            | Insn::ArithK1(_, _, _)
            | Insn::ArithK2(_, _, _)
            | Insn::ArithK4(_, _, _)
            | Insn::ArithK8(_, _, _)
            | Insn::Cmp(_, _, _)
            | Insn::Bool(_, _, _)
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
            | Insn::Phi { size: _ }
            | Insn::PhiBool
            | Insn::PhiArg { .. }
            | Insn::StructGet8 { .. }
            | Insn::StructGetMember { .. } => false,

            Insn::Call { .. }
            | Insn::CArg { .. }
            | Insn::Ret(_)
            | Insn::StoreMem(_, _)
            | Insn::JmpInd(_)
            | Insn::JmpExt(_)
            | Insn::Jmp(_)
            | Insn::JmpExtIf { .. }
            | Insn::JmpIf { .. }
            | Insn::TODO(_) => true,
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

fn fmt_arith(
    f: &mut std::fmt::Formatter<'_>,
    op: ArithOp,
    sz: u8,
    a: Reg,
    b: Reg,
) -> std::fmt::Result {
    let op = match op {
        ArithOp::Add => "add",
        ArithOp::Sub => "sub",
        ArithOp::Mul => "mul",
        ArithOp::Shl => "shl",
        ArithOp::BitAnd => "and",
        ArithOp::BitOr => "or",
        ArithOp::BitXor => "xor",
    };
    write!(f, "{:8} {:?},{:?}  {}", op, a, b, size_keyword(sz))
}

fn fmt_arithk(
    f: &mut std::fmt::Formatter<'_>,
    op: ArithOp,
    sz: u8,
    a: Reg,
    k: i64,
) -> std::fmt::Result {
    let op = match op {
        ArithOp::Add => "addk",
        ArithOp::Sub => "subk",
        ArithOp::Mul => "mulk",
        ArithOp::Shl => "shlk",
        ArithOp::BitAnd => "andk",
        ArithOp::BitOr => "ork",
        ArithOp::BitXor => "xork",
    };
    write!(f, "{:8} {:?},{:?} {}", op, a, k, size_keyword(sz))
}

fn size_keyword(sz: u8) -> &'static str {
    match sz {
        1 => "byte",
        2 => "word",
        4 => "dword",
        8 => "qword",
        _ => panic!("invalid size: {sz}"),
    }
}

impl std::fmt::Debug for Insn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Insn::True => write!(f, "true"),
            Insn::False => write!(f, "false"),
            Insn::Const1(val) => write!(f, "{:8} {} (0x{:x})", "const1", *val as i64, val),
            Insn::Const2(val) => write!(f, "{:8} {} (0x{:x})", "const2", *val as i64, val),
            Insn::Const4(val) => write!(f, "{:8} {} (0x{:x})", "const4", *val as i64, val),
            Insn::Const8(val) => write!(f, "{:8} {} (0x{:x})", "const8", *val as i64, val),
            Insn::Part { src, offset, size } => {
                write!(f, "{:8} {:?}[{}..{}]", "part", src, offset, offset + size)
            }
            Insn::Get(x) => write!(f, "{:8} {:?}", "get", x),
            Insn::Concat { lo, hi } => write!(f, "{:8} {:?}⧺{:?}", "concat", hi, lo),
            Insn::Widen1_2(x) => write!(f, "{:8} 1->2 {:?}", "widen", *x),
            Insn::Widen1_4(x) => write!(f, "{:8} 1->4 {:?}", "widen", *x),
            Insn::Widen1_8(x) => write!(f, "{:8} 1->8 {:?}", "widen", *x),
            Insn::Widen2_4(x) => write!(f, "{:8} 2->4 {:?}", "widen", *x),
            Insn::Widen2_8(x) => write!(f, "{:8} 2->8 {:?}", "widen", *x),
            Insn::Widen4_8(x) => write!(f, "{:8} 4->8 {:?}", "widen", *x),

            Insn::Arith1(op, a, b) => fmt_arith(f, *op, 1, *a, *b),
            Insn::Arith2(op, a, b) => fmt_arith(f, *op, 2, *a, *b),
            Insn::Arith4(op, a, b) => fmt_arith(f, *op, 4, *a, *b),
            Insn::Arith8(op, a, b) => fmt_arith(f, *op, 8, *a, *b),
            Insn::ArithK1(op, reg, k) => fmt_arithk(f, *op, 1, *reg, *k),
            Insn::ArithK2(op, reg, k) => fmt_arithk(f, *op, 2, *reg, *k),
            Insn::ArithK4(op, reg, k) => fmt_arithk(f, *op, 4, *reg, *k),
            Insn::ArithK8(op, reg, k) => fmt_arithk(f, *op, 8, *reg, *k),
            Insn::Cmp(op, a, b) => {
                let op = match op {
                    CmpOp::EQ => "==",
                    CmpOp::LT => "<",
                };
                write!(f, "{:8} {:?},{:?}", op, a, b)
            }
            Insn::Bool(op, a, b) => {
                let op = match op {
                    BoolOp::Or => "||",
                    BoolOp::And => "&&",
                };
                write!(f, "{:8} {:?},{:?}", op, a, b)
            }
            Insn::Not(x) => write!(f, "{:8} {:?}", "not", x),

            Insn::LoadMem1(addr) => write!(f, "{:8} addr:{:?}", "loadm1", addr),
            Insn::LoadMem2(addr) => write!(f, "{:8} addr:{:?}", "loadm2", addr),
            Insn::LoadMem4(addr) => write!(f, "{:8} addr:{:?}", "loadm4", addr),
            Insn::LoadMem8(addr) => write!(f, "{:8} addr:{:?}", "loadm8", addr),

            Insn::StoreMem(addr, val) => write!(f, "{:8} *{:?} ← {:?}", "store", addr, val),
            Insn::TODO(msg) => write!(f, "{:8} {}", "TODO", msg),

            Insn::Call(callee) => write!(f, "{:8} {:?}", "call", callee),
            Insn::CArg(value) => write!(f, "{:8} {:?}", "carg", value),
            Insn::Ret(x) => write!(f, "{:8} {:?}", "ret", x),
            Insn::JmpInd(x) => write!(f, "{:8} *{:?}", "jmp", x),
            Insn::Jmp(x) => write!(f, "{:8} {:?}", "jmp", x),
            Insn::JmpExt(target) => write!(f, "{:8} 0x{:x} extern", "jmp", target),
            Insn::JmpExtIf { cond, addr: target } => {
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

            Insn::Phi { size } => write!(f, "phi{size}"),
            Insn::PhiBool => write!(f, "phibool"),
            Insn::PhiArg(reg) => {
                write!(f, "{:8} {:?}", "phiarg", reg)
            }
            Insn::StructGet8 {
                struct_value,
                offset,
            } => write!(f, "{:8} {:?},{}", "sget8", *struct_value, offset),
            Insn::StructGetMember { struct_value, name } => {
                write!(f, "{:8} {:?},\"{}\"", "memb", *struct_value, name)
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
