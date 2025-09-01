use enum_assoc::Assoc;
use facet_reflect::HasFields;

// TODO This currently only represents the pre-SSA version of the program, but SSA conversion is
// coming
use std::{cell::Cell, collections::HashMap, sync::Arc};

use crate::ty;

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
    dest_tyids: Vec<Cell<ty::TypeID>>,
    reg_count: Index,

    // Not sure about the Arc here.  Very likely that it's going to have to
    // change when I do the GUI layer.
    types: Arc<ty::TypeSet>,

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
#[derive(Clone, Copy, PartialEq, Eq, Hash, facet::Facet)]
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
    Bytes(usize),
    Bool,
    MemoryEffect,
    Undefined,
    Unit,
    Control,
}
impl RegType {
    pub(crate) fn bytes_size(&self) -> Option<usize> {
        match self {
            RegType::Bytes(sz) => Some(*sz),
            RegType::Bool => None,
            RegType::MemoryEffect => None,
            RegType::Undefined => None,
            RegType::Unit => None,
            RegType::Control => None,
        }
    }
}

pub type ArgsMut<'a> = arrayvec::ArrayVec<&'a mut Reg, 3>;

fn array<T, const M: usize, const N: usize>(items: [T; M]) -> arrayvec::ArrayVec<T, N> {
    items.into_iter().collect()
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
    Const {
        value: i64,
        size: u16,
    },
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
        struct_value: Reg,
        name: &'static str,
        // larger size are definitely possible
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
    Ancestral(AncestralName),

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
            let dest_tyid = self.dest_tyids[ndx].get();
            let addr = self.addrs[ndx];

            if last_addr != addr {
                writeln!(f, "0x{:x}:", addr)?;
                last_addr = addr;
            }
            write!(f, "{:5} {:?}", ndx, dest,)?;
            write!(f, ": {:?}", dest_tyid)?;
            writeln!(f, " <- {:?}", insn)?;
        }
        Ok(())
    }
}

impl Program {
    #[inline(always)]
    pub fn get(&self, ndx: Index) -> Option<InsnView<'_>> {
        let index = ndx;
        let ndx = ndx as usize;
        // if this  slot is enabled as per the mask, then every Vec access must succeed
        let insn = &self.insns[ndx];
        let dest = &self.dests[ndx];
        let dest_tyid = &self.dest_tyids[ndx];
        let addr = self.addrs[ndx];
        Some(InsnView {
            insn,
            dest,
            tyid: dest_tyid,
            index,
            addr,
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
    pub(crate) fn iter(&self) -> impl Iterator<Item = InsnView<'_>> {
        (0..self.len()).filter_map(|ndx| self.get(ndx))
    }

    pub fn push(&mut self, dest: Reg, insn: Insn) -> Index {
        let index = self.insns.len().try_into().unwrap();
        self.insns.push(Cell::new(insn));
        self.dests.push(Cell::new(dest));
        self.dest_tyids.push(Cell::new(self.types.tyid_unknown()));
        self.addrs.push(u64::MAX);
        index
    }

    pub fn push_new(&mut self, insn: Insn) -> Index {
        let dest = Reg(self.len());
        let ndx = self.push(dest, insn);
        assert_eq!(Reg(ndx), dest);
        ndx
    }

    pub fn ancestor_type(&self, anc_name: AncestralName) -> Option<RegType> {
        self.anc_types.get(&anc_name).copied()
    }

    pub fn types(&self) -> &ty::TypeSet {
        &self.types
    }
}

pub struct InsnView<'a> {
    pub insn: &'a Cell<Insn>,
    pub tyid: &'a Cell<ty::TypeID>,
    pub dest: &'a Cell<Reg>,
    pub index: Index,
    pub addr: u64,
}

// will be mostly useful to keep origin info later
pub struct ProgramBuilder {
    insns: Vec<Cell<Insn>>,
    dests: Vec<Cell<Reg>>,
    addrs: Vec<u64>,
    dest_ty: Vec<Cell<ty::TypeID>>,
    cur_input_addr: u64,
    anc_types: HashMap<AncestralName, RegType>,
    types: Arc<ty::TypeSet>,

    reg_gen: RegGen,
    // Number of instructions that have already been checked for correct
    // use-after-init. See `check_use_after_init`.
    init_checked_count: usize,
}

impl ProgramBuilder {
    pub fn new(lowest_tmp: Reg, types: Arc<ty::TypeSet>) -> Self {
        Self {
            insns: Vec::new(),
            dests: Vec::new(),
            addrs: Vec::new(),
            dest_ty: Vec::new(),
            cur_input_addr: 0,
            anc_types: HashMap::new(),
            types,

            reg_gen: RegGen::new(lowest_tmp),
            init_checked_count: 0,
        }
    }

    pub fn types(&self) -> &Arc<ty::TypeSet> {
        &self.types
    }

    pub fn tmp_gen(&mut self) -> Reg {
        self.reg_gen.next()
    }
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

    pub fn push(&mut self, dest: Reg, insn: Insn) -> Reg {
        assert!(!matches!(insn, Insn::Phi));
        self.dests.push(Cell::new(dest));
        self.insns.push(Cell::new(insn));
        self.addrs.push(self.cur_input_addr);
        self.dest_ty.push(Cell::new(self.types.tyid_unknown()));
        dest
    }

    pub fn set_ancestral_type(&mut self, anc_name: AncestralName, typ: RegType) {
        let prev = self.anc_types.insert(anc_name, typ);
        assert!(
            prev.is_none(),
            "type assigned multiple times to same ancestral: {:?}",
            anc_name
        );
    }

    /// Associate the instructions emitted via the following calls to `emit_*` to the given
    /// address.
    ///
    /// This establishes a correspondence between the output MIL code and the input machine
    /// code, which is then used to resolve jumps, etc.
    pub fn set_input_addr(&mut self, addr: u64) {
        self.cur_input_addr = addr;
    }

    /// Associate the given type ID to the given register.
    ///
    /// The type ID is associated to the last instruction that assigned the
    /// register. Panics if this instruction can't be located (it's a user bug)
    pub fn set_type(&mut self, reg: Reg, tyid: ty::TypeID) {
        let (ndx, _) = self
            .dests
            .iter()
            .enumerate()
            .rev()
            .find(|(_, dest)| dest.get() == reg)
            .expect("no instruction writes to the given register");

        self.dest_ty[ndx].set(tyid);
    }

    pub fn build(mut self) -> Program {
        self.check_use_after_init();
        assert_eq!(self.init_checked_count, self.dests.len());

        let Self {
            insns,
            dests,
            addrs,
            dest_ty,
            types,
            ..
        } = self;

        assert_eq!(dests.len(), insns.len());
        assert_eq!(dests.len(), addrs.len());
        assert_eq!(dests.len(), dest_ty.len());

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
                Insn::Control(Control::JmpExt(addr)) => {
                    if let Some(ndx) = mil_of_input_addr.get(&addr) {
                        insn.set(Insn::Control(Control::Jmp(*ndx)));
                    }
                }
                Insn::Control(Control::JmpExtIf(addr)) => {
                    if let Some(ndx) = mil_of_input_addr.get(&addr) {
                        insn.set(Insn::Control(Control::JmpIf(*ndx)));
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
            dest_tyids: dest_ty,
            types,
            reg_count,
            mil_of_input_addr,
            anc_types: self.anc_types,
        }
    }
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
