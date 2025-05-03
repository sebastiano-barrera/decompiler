use enum_assoc::Assoc;

// TODO This currently only represents the pre-SSA version of the program, but SSA conversion is
// coming
use std::{cell::Cell, collections::HashMap, ops::Range, sync::Arc};

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
    dest_tyids: Vec<Cell<Option<ty::TypeID>>>,
    // aligned to insns. when false, the corresponding instruction is entirely
    // disabled and inaccessible. for all intents and purposes, it's deleted.
    // its index never yielded by iterators, and accesses to it result in
    // a panic.
    is_enabled: Vec<bool>,
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
    Bytes(usize),
    Bool,
    MemoryEffect,
    Undefined,
    Unit,
}
impl RegType {
    pub(crate) fn bytes_size(&self) -> Option<usize> {
        match self {
            RegType::Bytes(sz) => Some(*sz),
            RegType::Bool => None,
            RegType::MemoryEffect => None,
            RegType::Undefined => None,
            RegType::Unit => None,
        }
    }
}

pub type ArgsMut<'a> = arrayvec::ArrayVec<&'a mut Reg, 3>;

fn array<T, const M: usize, const N: usize>(items: [T; M]) -> arrayvec::ArrayVec<T, N> {
    items.into_iter().collect()
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug, Assoc)]
#[func(pub fn has_side_effects(&self) -> bool { false })]
#[func(pub fn is_replaceable_with_get(&self) -> bool { ! self.has_side_effects() })]
#[func(pub fn input_regs(&mut self) -> ArgsMut { ArgsMut::new() })]
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
    Ret(Reg),
    #[assoc(has_side_effects = true)]
    #[assoc(input_regs = array([_0]))]
    JmpInd(Reg),
    #[assoc(has_side_effects = true)]
    Jmp(Index),
    #[assoc(has_side_effects = true)]
    #[assoc(input_regs = array([_cond]))]
    JmpIf {
        cond: Reg,
        target: Index,
    },
    #[assoc(has_side_effects = true)]
    JmpExt(u64),
    #[assoc(has_side_effects = true)]
    #[assoc(input_regs = array([_cond]))]
    JmpExtIf {
        cond: Reg,
        addr: u64,
    },

    #[allow(clippy::upper_case_acronyms)]
    #[assoc(has_side_effects = true)]
    NotYetImplemented(&'static str),

    #[assoc(input_regs = array([_mem, _addr]))]
    LoadMem {
        mem: Reg,
        addr: Reg,
        size: u32,
    },
    #[assoc(has_side_effects = true)]
    #[assoc(input_regs = array([_mem, _addr, _value]))]
    StoreMem {
        mem: Reg,
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

    Undefined,
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

/// Binary comparison operators. Inputs are integers; the output is a boolean.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
#[allow(dead_code)]
pub enum CmpOp {
    EQ,
    LT,
}

/// Binary boolean operators. Inputs and outputs are booleans.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub enum BoolOp {
    Or,
    And,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
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
            let is_enabled = self.is_enabled[ndx];
            let insn = self.insns[ndx].get();
            let dest = self.dests[ndx].get();
            let dest_tyid = self.dest_tyids[ndx].get();
            let addr = self.addrs[ndx];

            if last_addr != addr {
                writeln!(f, "0x{:x}:", addr)?;
                last_addr = addr;
            }
            write!(f, "   {:10}", if is_enabled { "" } else { "|DISABLED|" })?;
            write!(f, "{:5} {:?}", ndx, dest,)?;
            if let Some(dest_tyid) = dest_tyid {
                write!(f, ": {:?}", dest_tyid)?;
            }
            writeln!(f, " <- {:?}", insn)?;
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
        if self.is_enabled(ndx) {
            let ndx = ndx as usize;
            // if this  slot is enabled as per the mask, then every Vec access must succeed
            let insn = &self.insns[ndx];
            let dest = &self.dests[ndx];
            let dest_tyid = &self.dest_tyids[ndx];
            // will be re-enabled one day
            // let addr = self.addrs[ndx];
            Some(InsnView {
                insn,
                dest,
                tyid: dest_tyid,
            })
        } else {
            None
        }
    }

    pub fn slice(&self, ndxr: Range<Index>) -> Option<InsnSlice> {
        let insn = self.insns.get(range_conv(ndxr.clone()))?;
        let dest = &self.dests[range_conv(ndxr)];
        Some(InsnSlice {
            insns: insn,
            dests: dest,
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
    pub(crate) fn iter(&self) -> impl Iterator<Item = InsnView> {
        (0..self.len()).filter_map(|ndx| self.get(ndx))
    }

    pub fn push(&mut self, dest: Reg, insn: Insn) -> Index {
        let index = self.insns.len().try_into().unwrap();
        self.insns.push(Cell::new(insn));
        self.dests.push(Cell::new(dest));
        self.dest_tyids.push(Cell::new(None));
        self.addrs.push(u64::MAX);
        self.is_enabled.push(true);
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

    /// Replace the "enabled" mask.
    ///
    /// `mask` MUST be a Vec with length equal to `self.len()`. `mask[i]`
    /// is true iff the i-th instruction is enabled. Otherwise, index `i` is
    /// disabled (accesses to it via .get() result in a panic) and it is never
    /// yielded from iterators.
    pub fn set_enabled_mask(&mut self, mask: Vec<bool>) {
        assert_eq!(mask.len(), self.len() as usize);
        self.is_enabled = mask;
    }

    /// Return true iff the instruction at the given Index is enabled, as per the mask (see [`set_enabled_mask`]).
    ///
    /// Panics if `ndx` is invalid (out of the Program's range, i.e. `>= self.len()`)
    pub(crate) fn is_enabled(&self, ndx: Index) -> bool {
        self.is_enabled[ndx as usize]
    }
}

pub struct InsnView<'a> {
    pub insn: &'a Cell<Insn>,
    pub tyid: &'a Cell<Option<ty::TypeID>>,
    pub dest: &'a Cell<Reg>,
    // no use for this right now
    // pub addr: u64,
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
    pub fn iter_copied(&self) -> impl '_ + DoubleEndedIterator<Item = (Reg, Insn)> {
        self.iter().map(|(d, r)| (d.get(), r.get()))
    }
}

// will be mostly useful to keep origin info later
pub struct ProgramBuilder {
    insns: Vec<Cell<Insn>>,
    dests: Vec<Cell<Reg>>,
    addrs: Vec<u64>,
    dest_ty: Vec<Cell<Option<ty::TypeID>>>,
    cur_input_addr: u64,
    anc_types: HashMap<AncestralName, RegType>,
    types: Arc<ty::TypeSet>,
}

impl ProgramBuilder {
    pub fn new(types: Arc<ty::TypeSet>) -> Self {
        Self {
            insns: Vec::new(),
            dests: Vec::new(),
            addrs: Vec::new(),
            dest_ty: Vec::new(),
            cur_input_addr: 0,
            anc_types: HashMap::new(),
            types,
        }
    }

    pub fn types(&self) -> &Arc<ty::TypeSet> {
        &self.types
    }

    pub fn push(&mut self, dest: Reg, insn: Insn) -> Reg {
        assert!(!matches!(insn, Insn::Phi));
        self.dests.push(Cell::new(dest));
        self.insns.push(Cell::new(insn));
        self.addrs.push(self.cur_input_addr);
        self.dest_ty.push(Cell::new(None));
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

    /// Associate the given type ID to the given register.
    ///
    /// The type ID is associated to the last instruction that assigned the
    /// register. Panics if this instruction can't be located (it's a user bug)
    pub fn set_type(&mut self, reg: Reg, tyid: Option<ty::TypeID>) {
        let (ndx, _) = self
            .dests
            .iter()
            .enumerate()
            .rev()
            .find(|(_, dest)| dest.get() == reg)
            .expect("no instruction writes to the given register");

        self.dest_ty[ndx].set(tyid);
    }

    pub fn build(self) -> Program {
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

        let is_enabled = vec![true; insns.len()];

        Program {
            insns,
            dests,
            addrs,
            dest_tyids: dest_ty,
            is_enabled,
            types,
            reg_count,
            mil_of_input_addr,
            anc_types: self.anc_types,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask() {
        let mut bld = ProgramBuilder::new(Arc::new(ty::TypeSet::new()));
        bld.push(Reg(10), Insn::True);
        bld.push(
            Reg(20),
            Insn::Const {
                value: 874,
                size: 4,
            },
        );
        bld.push(Reg(15), Insn::Arith(ArithOp::Add, Reg(10), Reg(10)));
        let mut prog = bld.build();
        let mask = vec![true, false, true];
        prog.set_enabled_mask(mask.clone());

        // the register count stays the same (nicer to let every downstream user
        // reserve space for insn slots that may be made active later)
        assert_eq!(prog.reg_count(), 21);

        let dests: Vec<_> = prog.iter().map(|iv| iv.dest.get()).collect();
        assert_eq!(&dests, &[Reg(10), Reg(15)]);

        // the point is these won't panic
        let mask_check: Vec<_> = (0..3).map(|ndx| prog.get(ndx).is_some()).collect();
        assert_eq!(mask_check, mask);
    }
}
