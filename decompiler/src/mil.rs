use enum_assoc::Assoc;
use facet_reflect::HasFields;

// TODO This currently only represents the pre-SSA version of the program, but SSA conversion is
// coming
use std::{cell::Cell, collections::HashMap};

use crate::{
    ty,
    util::{Bytes, Float32Bytes, Float64Bytes},
};

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
    /// Initial partial assignment of TypeID's to each instruction
    tyids: Vec<Option<ty::TypeID>>,

    // TODO More specific types
    // kept even if dead, because we will still want to trace each MIL
    // instruction back to the original machine code / assembly
    #[allow(dead_code)]
    mil_of_input_addr: HashMap<u64, Index>,

    /// Address that will be associated to all instructions emitted until the
    /// next call to `set_input_addr`.
    cur_input_addr: u64,

    /// Generator of temporary registers.
    ///
    /// The generated registers are always numerically >= `first_tmp`.
    ///
    /// The generator can be reset to start again from `first_tmp`, which is
    /// supposed to happen at the start of each ASM instruction (compilation of
    /// distinct assembly instruction should never communicate via temporary
    /// registers).
    reg_gen: RegGen,

    /// Number of instructions that have already been checked for correct
    /// use-after-init. See `check_use_after_init`.
    init_checked_count: usize,

    /// Machine endianness.
    ///
    /// This needs to be specified in order to keep the code really
    /// machine-independent.
    ///
    /// The default is Little Endian.
    endianness: Endianness,
}

/// Register ID
///
/// The language admits as many registers as a u16 can represent (2**16). They're
/// abstract, so we don't pay for them!
#[derive(Clone, Copy, PartialEq, Eq, Hash, facet::Facet)]
pub struct Reg(pub u16);

/// Ergonomic short-hand for the mil::Reg constructor
#[allow(non_snake_case)]
pub const fn R(ndx: u16) -> Reg {
    Reg(ndx)
}

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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, facet::Facet)]
#[repr(u8)]
pub enum RegType {
    Bytes(usize),
    Float { bytes_size: usize },
    Bool,
    Effect,
}
impl RegType {
    pub(crate) fn bytes_size(&self) -> Option<usize> {
        match self {
            RegType::Bytes(sz) => Some(*sz),
            RegType::Float { bytes_size } => Some(*bytes_size),
            RegType::Bool => None,
            RegType::Effect => None,
        }
    }
}

pub type ArgsMut<'a> = arrayvec::ArrayVec<&'a mut Reg, 3>;

fn array<T, const M: usize, const N: usize>(items: [T; M]) -> arrayvec::ArrayVec<T, N> {
    items.into_iter().collect()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, facet::Facet)]
#[repr(u8)]
pub enum Endianness {
    Little,
    Big,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug, Assoc, facet::Facet)]
#[repr(u8)]
#[func(pub fn has_side_effects(&self) -> bool { false })]
#[func(pub fn is_replaceable_with_get(&self) -> bool { ! self.has_side_effects() })]
#[func(pub fn input_regs(&mut self) -> ArgsMut<'_> { ArgsMut::new() })]
#[allow(dead_code)]
pub enum Insn {
    Void,
    True,
    False,
    Bytes(Bytes),
    Int {
        value: i64,
        size: u16,
    },
    Float32(Float32Bytes),
    Float64(Float64Bytes),

    #[assoc(input_regs = array([_0]))]
    ReinterpretFloat32(Reg),
    #[assoc(input_regs = array([_0]))]
    ReinterpretFloat64(Reg),

    #[assoc(input_regs = array([_0]))]
    Get(Reg),

    #[assoc(input_regs = array([_src]))]
    Part {
        src: Reg,
        offset: u16,
        size: u16,
    },
    #[assoc(input_regs = array([_lo, _hi]))]
    Concat {
        lo: Reg,
        hi: Reg,
    },

    #[assoc(input_regs = array([_struct_value]))]
    StructGetMember {
        /// Not necessarily a Struct
        struct_value: Reg,
        name: &'static str,
        // larger size are definitely possible
        size: u32,
    },
    #[assoc(input_regs = [_first_member.as_mut()].into_iter().flatten().collect())]
    Struct {
        // TODO figure out proper memory management for these
        type_name: &'static str,
        first_member: Option<Reg>,
        size: u32,
    },
    #[assoc(input_regs = [Some(_value), _next.as_mut()].into_iter().flatten().collect())]
    StructMember {
        // TODO figure out proper memory management for these
        name: &'static str,
        value: Reg,
        next: Option<Reg>,
    },
    #[assoc(input_regs = array([_array]))]
    ArrayGetElement {
        array: Reg,
        index: u32,
        size: u32,
    },

    #[assoc(input_regs = array([_reg]))]
    Widen {
        reg: Reg,
        target_size: u16,
        sign: bool,
    },

    #[assoc(input_regs = array([_1, _2]))]
    Arith(ArithOp, Reg, Reg),
    #[assoc(input_regs = array([_1]))]
    ArithK(ArithOp, Reg, i64),
    #[assoc(input_regs = array([_1, _2]))]
    Cmp(CmpOp, Reg, Reg),
    #[assoc(input_regs = array([_1, _2]))]
    Bool(BoolOp, Reg, Reg),
    #[assoc(input_regs = array([_0]))]
    Not(Reg),

    #[assoc(has_side_effects = true)]
    #[assoc(input_regs = [Some(_callee), _first_arg.as_mut()].into_iter().flatten().collect())]
    Call {
        callee: Reg,
        first_arg: Option<Reg>,
    },
    #[assoc(is_replaceable_with_get = false)]
    #[assoc(input_regs = [Some(_value), _next_arg.as_mut()].into_iter().flatten().collect())]
    CArg {
        value: Reg,
        next_arg: Option<Reg>,
    },

    #[assoc(has_side_effects = true)]
    #[assoc(input_regs = array([_0]))]
    SetReturnValue(Reg),

    #[assoc(has_side_effects = true)]
    #[assoc(input_regs = array([_0]))]
    SetJumpCondition(Reg),

    #[assoc(has_side_effects = true)]
    #[assoc(input_regs = array([_0]))]
    SetJumpTarget(Reg),

    Control(Control),

    #[allow(clippy::upper_case_acronyms)]
    #[assoc(has_side_effects = true)]
    NotYetImplemented(&'static str),

    #[assoc(input_regs = array([_addr]))]
    LoadMem {
        addr: Reg,
        size: u32,
    },
    #[assoc(has_side_effects = true)]
    #[assoc(input_regs = array([_addr, _value]))]
    StoreMem {
        addr: Reg,
        value: Reg,
    },

    #[assoc(input_regs = array([_0]))]
    OverflowOf(Reg),
    #[assoc(input_regs = array([_0]))]
    CarryOf(Reg),
    #[assoc(input_regs = array([_0]))]
    SignOf(Reg),
    #[assoc(input_regs = array([_0]))]
    IsZero(Reg),
    #[assoc(input_regs = array([_0]))]
    Parity(Reg),
    UndefinedBool,
    UndefinedBytes {
        size: u32,
    },

    FuncArgument {
        index: u16,
        reg_type: RegType,
    },
    Ancestral {
        anc_name: AncestralName,
        reg_type: RegType,
    },

    Phi,

    // must be marked with has_side_effects = true, in order to be associated to specific basic blocks
    #[assoc(has_side_effects = true)]
    #[assoc(input_regs = array([_value]))]
    Upsilon {
        value: Reg,
        phi_ref: Reg,
    },
}

// TODO Match/unify Control and cfg::BlockCont?
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug, facet::Facet)]
#[repr(u8)]
pub enum Control {
    /// Return to the calling function.
    ///
    /// The return value must previously be set by an `Insn::SetReturnValue(_)`.
    /// If the function has no return value, an `Insn::Void` value can be used; if
    /// it is unknown, `Insn::Undefined` can be used.
    Ret,

    /// Jump to the associated Index.
    Jmp(Index),

    /// Jump to the associated Index if the value set by the last
    /// Insn::SetJumpCondition(_) is true (must be a RegType::Bool).
    JmpIf(Index),

    /// Jump to the associated machine address, which is external to the function.
    JmpExt(u64),

    /// Jump to the associated machine address, which is external to the
    /// function, if the value set by the last Insn::SetJumpCondition(_) is true
    /// (must be a RegType::Bool).
    JmpExtIf(u64),

    /// Jump to the last address set by Insn::SetJumpTarget(_) in this block
    JmpIndirect,
}

/// Binary comparison operators. Inputs are integers; the output is a boolean.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug, facet::Facet)]
#[repr(u8)]
pub enum CmpOp {
    EQ,
    LT,
}
impl CmpOp {
    pub fn symbol(&self) -> &'static str {
        match self {
            CmpOp::EQ => "==",
            CmpOp::LT => "<",
        }
    }
}

/// Binary boolean operators. Inputs and outputs are booleans.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug, facet::Facet)]
#[repr(u8)]
pub enum BoolOp {
    Or,
    And,
}
impl BoolOp {
    pub fn symbol(&self) -> &'static str {
        match self {
            BoolOp::Or => "||",
            BoolOp::And => "&&",
        }
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug, facet::Facet)]
#[repr(u8)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Shl,
    Shr,
    BitXor,
    BitAnd,
    BitOr,
}

impl ArithOp {
    pub fn symbol(&self) -> &'static str {
        match self {
            ArithOp::Add => "+",
            ArithOp::Sub => "-",
            ArithOp::Mul => "*",
            ArithOp::Shl => "<<",
            ArithOp::Shr => ">>",
            ArithOp::BitXor => "^",
            ArithOp::BitAnd => "&",
            ArithOp::BitOr => "|",
        }
    }
}

/// The "name" (identifier) of an "ancestral" value, i.e. a value in MIL code
/// that represents the pre-existing value of a machine register at the time
/// the function started execution.  Mostly useful to allow the decompilation to
/// proceed forward even when somehting is out of place.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug, facet::Facet)]
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
    pub fn input_regs_iter(&mut self) -> impl Iterator<Item = &mut Reg> {
        self.input_regs().into_iter()
    }
}

impl std::fmt::Debug for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "program  {} instrs", self.insns.len())?;
        let mut last_addr = 0;
        let len = self.dests.len();
        for ndx in 0..len {
            let insn = self.insns[ndx].get();
            let dest = self.dests[ndx].get();
            let addr = self.addrs[ndx];

            if last_addr != addr {
                writeln!(f, "0x{:x}:", addr)?;
                last_addr = addr;
            }
            write!(f, "{:5} {:?}", ndx, dest,)?;
            writeln!(f, " <- {:?}", insn)?;
        }
        Ok(())
    }
}

impl Program {
    pub fn new(lowest_tmp: Reg) -> Self {
        Self {
            insns: Vec::new(),
            dests: Vec::new(),
            addrs: Vec::new(),
            cur_input_addr: 0,
            tyids: Vec::new(),
            reg_gen: RegGen::new(lowest_tmp),
            init_checked_count: 0,
            mil_of_input_addr: HashMap::new(),
            endianness: Endianness::Little,
        }
    }

    pub fn endianness(&self) -> Endianness {
        self.endianness
    }
    pub fn set_endianness(&mut self, endianness: Endianness) {
        self.endianness = endianness;
    }

    /// Generate a new temporary register.
    ///
    /// Call `tmp_reset()` to reset the generator after finishing compilation of
    /// a single assembly instruction.
    pub fn tmp_gen(&mut self) -> Reg {
        self.reg_gen.next()
    }
    /// Reset the temporary register generator, so that the next call to
    /// `tmp_gen()` returns the first temporary register again (parameter `lowest_tmp` of
    /// [`Self::new`]).
    pub fn tmp_reset(&mut self) {
        self.reg_gen.reset();
        self.check_use_after_init();
    }

    /// Check that all used temporary registers have been initialized (written)
    /// before use.
    ///
    /// This is a necessary condition for phi nodes to always have a valid input
    /// value for all predecessors.
    ///
    /// At every call to this function, only the instructions inserted since the
    /// last call (or since the construction of this ProgramBuilder) are
    /// checked. For all used tmeporary MIL registers, initialization must *in
    /// this range* happen before each use.
    fn check_use_after_init(&mut self) {
        let ndx_start = self.init_checked_count;
        let ndx_end = self.dests.len();

        for ndx in ndx_start..ndx_end {
            for &mut input in self.insns[ndx].get().input_regs() {
                let is_temporary = input.0 >= self.reg_gen.first.0;
                if !is_temporary {
                    continue;
                }

                // in the preceding part of the program (since the last check/reset),
                // at least one instruction writes `input`
                let is_initialized =
                    (ndx_start..ndx).any(|ndx_inner| self.dests[ndx_inner].get() == input);
                if !is_initialized {
                    panic!(
                        "Temporary register {:?} used at instruction {} without prior initialization in this block.",
                        input,
                        ndx,
                    );
                }
            }
        }

        self.init_checked_count = ndx_end;
    }

    /// Push a new instruction to the program, associating it to the given
    /// destination register.
    ///
    /// The instruction must not be a Phi node (these are only allowed to be
    /// introduced during conversion to SSA).
    pub fn push(&mut self, dest: Reg, insn: Insn) -> Index {
        self.dests.push(Cell::new(dest));
        self.insns.push(Cell::new(insn));
        self.addrs.push(self.cur_input_addr);
        self.tyids.push(None);
        self.len() - 1
    }

    /// Associate the instructions emitted via the following calls to `emit_*` to the given
    /// address.
    ///
    /// This establishes a correspondence between the output MIL code and the input machine
    /// code, which is then used to resolve jumps, etc.
    pub fn set_input_addr(&mut self, addr: u64) {
        self.cur_input_addr = addr;
    }

    /// Check that this MIL program is valid, assuming that it is not currently
    /// undergoing active edits/modifications.
    ///
    /// In particular, this will call tmp_reset(), and therefore check that
    /// all the instructions added since the last call to `tmp_reset` ahve been
    /// initialized before use. (See [`check_use_after_init`].)
    pub fn validate(&mut self) {
        self.tmp_reset();
        self.assert_invariants();
    }

    fn assert_invariants(&self) {
        let count = self.dests.len();
        assert_eq!(self.init_checked_count, count, "Not all instructions have been checked for use-after-init. Call tmp_reset() at the end of each assembly instruction.");
        assert_eq!(count, self.insns.len());
        assert_eq!(count, self.addrs.len());
        assert_eq!(count, self.tyids.len());
        assert!(!self
            .insns
            .iter()
            .any(|insn| matches!(insn.get(), Insn::Phi)));
    }

    pub fn convert_jmp_ext_to_int(&mut self) {
        // mil::Index (in this MIL code) corresponding to each machine code address
        let index_of_mcode_addr = self.map_index_of_mcode_addr();

        for insn in &self.insns {
            match insn.get() {
                Insn::Control(Control::JmpExt(addr)) => {
                    if let Some(ndx) = index_of_mcode_addr.get(&addr) {
                        insn.set(Insn::Control(Control::Jmp(*ndx)));
                    }
                }
                Insn::Control(Control::JmpExtIf(addr)) => {
                    if let Some(ndx) = index_of_mcode_addr.get(&addr) {
                        insn.set(Insn::Control(Control::JmpIf(*ndx)));
                    }
                }
                _ => {}
            }
        }
    }

    fn map_index_of_mcode_addr(&self) -> HashMap<u64, Index> {
        let mut map = HashMap::new();
        let mut last_addr = u64::MAX;
        for (ndx, &addr) in self.addrs.iter().enumerate() {
            if addr != last_addr {
                let ndx = ndx.try_into().unwrap();
                map.insert(addr, ndx);
                last_addr = addr;
            }
        }
        map
    }

    /// Return the number of distinct registers used in this program.
    ///
    /// This information is not cached. Every call to this function will iterate
    /// through the code and recompute it.
    pub fn count_distinct_regs(&self) -> Index {
        let max_dest = self
            .dests
            .iter()
            .map(|reg| reg.get().reg_index())
            .max()
            .unwrap_or(0);
        let max_input = self
            .insns
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
    }

    // pub fn build(mut self) -> Program {
    //     self.assert_invariants();
    //     self.convert_jmp_ext_to_int();
    //     self.count_distinct_regs();
    //     self.cur_input_addr = u64::MAX;
    // }

    #[inline(always)]
    pub fn get(&self, ndx: Index) -> Option<InsnView<'_>> {
        let index = ndx;
        let ndx = ndx as usize;
        // if this  slot is enabled as per the mask, then every Vec access must succeed
        let insn = &self.insns[ndx];
        let dest = &self.dests[ndx];
        let addr = self.addrs[ndx];
        Some(InsnView {
            insn,
            dest,
            index,
            addr,
        })
    }

    #[inline(always)]
    pub fn len(&self) -> Index {
        self.insns.len().try_into().unwrap()
    }

    #[inline(always)]
    pub(crate) fn iter(&self) -> impl DoubleEndedIterator<Item = InsnView<'_>> {
        (0..self.len()).map(|ndx| self.get(ndx).unwrap())
    }

    pub fn value_type(&self, index: Index) -> Option<ty::TypeID> {
        self.tyids.get(index as usize).copied().flatten()
    }

    /// Set the value type (TypeID) for the result of the instruction located at
    /// the given index.
    ///
    /// This operation must not be done twice for the same instruction (index).
    /// This function panics on violation of this rule.
    pub fn set_value_type(&mut self, index: Index, tyid: ty::TypeID) {
        let prev = std::mem::replace(&mut self.tyids[index as usize], Some(tyid));
        assert!(prev.is_none());
    }

    pub fn unwrap(self) -> ProgramCore {
        ProgramCore {
            insns: self.insns.into_iter().map(Cell::into_inner).collect(),
            dests: self.dests.into_iter().map(Cell::into_inner).collect(),
            addrs: self.addrs,
            tyids: self.tyids,
            endianness: self.endianness,
        }
    }
}

/// The innards of a [Program], without the API.
///
/// Designed to favor transformation into other forms and not to be converted
/// back into a [mil::Program].
pub struct ProgramCore {
    pub insns: Vec<Insn>,
    pub dests: Vec<Reg>,
    pub addrs: Vec<u64>,
    pub tyids: Vec<Option<ty::TypeID>>,
    pub endianness: Endianness,
}

pub struct InsnView<'a> {
    pub insn: &'a Cell<Insn>,
    pub dest: &'a Cell<Reg>,
    pub index: Index,
    pub addr: u64,
}

pub struct ExpandedInsn {
    pub variant_name: &'static str,
    pub fields: arrayvec::ArrayVec<(&'static str, ExpandedValue), 3>,
}
pub enum ExpandedValue {
    Reg(Reg),
    Generic(String),
}

pub fn to_expanded(insn: &Insn) -> ExpandedInsn {
    let peek = facet_reflect::Peek::new(insn).into_enum().unwrap();

    let variant_index = peek.variant_index().unwrap();
    let variant_name = peek.variant_name(variant_index).unwrap();

    let fields = peek
        .fields()
        .map(|(field, peek)| {
            // fields of type Reg and Option<Reg> are translated explicitly;
            // everything else is translated to Generic with the debug string

            let ev = if let Ok(reg) = peek.get::<Reg>() {
                ExpandedValue::Reg(*reg)
            } else if let Some(reg) = peek
                .into_option()
                .ok()
                .and_then(|peek| peek.value())
                .as_ref()
                .and_then(|peek| peek.get::<Reg>().ok())
            {
                ExpandedValue::Reg(*reg)
            } else {
                ExpandedValue::Generic(format!("{:?}", peek))
            };

            (field.name, ev)
        })
        .collect();

    ExpandedInsn {
        variant_name,
        fields,
    }
}

#[derive(Clone)]
struct RegGen {
    first: Reg,
    next: Reg,
}
impl RegGen {
    fn new(first: Reg) -> Self {
        RegGen { first, next: first }
    }

    fn next(&mut self) -> Reg {
        let ret = self.next;
        self.next.0 += 1;
        ret
    }

    fn reset(&mut self) {
        self.next = self.first;
    }
}
