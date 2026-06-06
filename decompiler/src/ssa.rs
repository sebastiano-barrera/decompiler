//! Static Single-Assignment representation of a program (and conversion from direct multiple
//! assignment).
//!
//! The algorithms in this module are mostly derived from the descriptions in:
//! > Cooper, Keith & Harvey, Timothy & Kennedy, Ken. (2006).
//! > A Simple, Fast Dominance Algorithm.
//! > Rice University, CS Technical Report 06-33870.

use thiserror::Error;
use tracing::{event, Level};

use crate::{
    cfg,
    mil::{self, Endianness},
    pp, ty,
    util::Bytes,
    LLType,
};
use std::io::Write;

mod cons;

#[derive(Clone)]
pub struct Program {
    // # Design notes
    //
    // - SSA registers are numerically equal to the index of the defining
    //   instruction in `inner`. mil::Index values are just as good as mil::Reg
    //   for identifying both insns and values. This allows for fast lookups.
    //
    // - Struct-of-Arrays. Each of the Vecs below is the same length, and the
    //   element at the same index in each represents a different facet of the
    //   same instruction. This is checked by `assert_invariants`, among other
    //   things.
    //
    // - the order of instructions in the arrays below (insns, dest_tyids, etc.)
    //   is meaningless. after conversion from a mil::Program, extra instructions
    //   can be appended for various uses.  the actual program order is determined
    //   entirely by `schedule`.  instructions included in `scheduled` are said to
    //   be "scheduled".
    //
    // - it's a bug if an instruction has an input register whose defining
    //   instruction is not scheduled.
    insns: Vec<mil::Insn>,
    addrs: Vec<u64>,

    /// Cached LLType for each instruction.
    ll_types: Vec<mil::LLType>,

    /// Type ID for the instruction's result.
    ///
    /// Managed externally (via getter `value_type` and setter `set_value_type`).
    ///
    /// The default for all instruction after initialization or insertion after
    /// initialization is None, which represents lack of type information.
    ///
    /// This is intended to be manipulated by specific passes in `xform`.
    tyids: Vec<Option<ty::TypeID>>,

    /// The TypeID of the function this SSA program represents.
    func_tyid: Option<ty::TypeID>,

    schedule: cfg::Schedule,
    cfg: cfg::Graph,

    endianness: Endianness,

    faults: Vec<Fault>,
}

/// Correctness relies on a few invariants.
///
/// In order to make it feasible to assert them, every API in this impl only gives
/// read-only access. Every modification (1) requires a &mut Program; (2) requires
/// that the &mut Program is wrapped in a temporary [OpenProgram], which is the
/// only type provding editing APIs.
impl Program {
    pub fn empty() -> Self {
        // I know this looks weird, but it's the easiest way to get all the cfg
        // structures properly initialized without having to think too much
        // about it.
        let mil_empty = mil::Program::new(mil::Reg(0), None);
        Self::from_mil(mil_empty)
    }

    pub fn from_mil(mil: mil::Program) -> Self {
        cons::mil_to_ssa(mil)
    }

    pub fn cfg(&self) -> &cfg::Graph {
        &self.cfg
    }

    pub fn endianness(&self) -> Endianness {
        self.endianness
    }

    pub fn function_type_id(&self) -> Option<ty::TypeID> {
        self.func_tyid
    }

    /// Get the defining instruction for the given register by shared reference.
    ///
    /// This is borrowed access only: callers that merely inspect instructions
    /// should avoid cloning, while callers that need to rewrite a temporary
    /// instruction value can clone explicitly at the call site.
    ///
    /// (Note that it's not allowed to fetch instructions by position.)
    pub fn get(&self, reg: mil::Reg) -> Option<&mil::Insn> {
        // In SSA, Reg(ndx) happens to be located at index ndx.
        // if this  slot is enabled as per the mask, then every Vec access must succeed
        self.insns.get(reg.reg_index() as usize)
    }

    /// Get the machine-code address for the given register's defining instruction.
    pub fn machine_addr(&self, reg: mil::Reg) -> Option<u64> {
        self.addrs.get(reg.reg_index() as usize).copied()
    }

    /// Return the bytes represented by an `Insn::Int {value, size}`.
    ///
    /// Depends on the source machine's endianness.
    pub fn int_bytes(&self, value: i64, size: u16) -> Bytes {
        int_bytes(value, size, self.endianness)
    }

    /// Total number of registers/instructions stored in this Program.
    ///
    /// This may well include dead values. Use [`insns_rpo`] to iterate through
    /// the program only "through dependency edges" and only get live registers.
    pub fn reg_count(&self) -> mil::Index {
        self.insns.len().try_into().unwrap()
    }

    pub fn registers(&self) -> impl Iterator<Item = mil::Reg> {
        (0..self.reg_count()).map(mil::Reg)
    }

    pub fn insns_rpo(&self) -> impl '_ + DoubleEndedIterator<Item = (cfg::BlockID, mil::Reg)> {
        self.cfg
            .block_ids_rpo()
            .flat_map(|bid| self.block_regs(bid).map(move |reg| (bid, reg)))
    }

    pub fn block_regs(&self, bid: cfg::BlockID) -> impl '_ + DoubleEndedIterator<Item = mil::Reg> {
        self.schedule.of_block(bid).iter().map(|&ndx| mil::Reg(ndx))
    }

    pub fn find_last_matching<P, R>(&self, bid: cfg::BlockID, pred: P) -> Option<R>
    where
        P: Fn(&mil::Insn) -> Option<R>,
    {
        self.block_regs(bid)
            .rev()
            .map(|reg| self.get(reg).unwrap())
            .find_map(pred)
    }

    pub fn upsilons_of_phi(&self, phi_reg: mil::Reg) -> impl '_ + Iterator<Item = UpsilonDesc> {
        assert!(matches!(self.get(phi_reg).unwrap(), mil::Insn::Phi));

        self.registers()
            .filter_map(move |reg| match self.get(reg).unwrap() {
                mil::Insn::Upsilon { value, phi_ref } if *phi_ref == phi_reg => Some(UpsilonDesc {
                    upsilon_reg: reg,
                    input_reg: *value,
                }),
                _ => None,
            })
    }

    /// Return the TypeID of the given register (or, equivalently, of the given
    /// register's defining instruction's result).
    ///
    /// Returns None if `reg` is invalid.
    pub fn value_type(&self, reg: mil::Reg) -> Option<ty::TypeID> {
        self.tyids.get(reg.0 as usize).copied().flatten()
    }

    /// Return the LLType assigned to the given register.
    ///
    /// This amounts to a simple lookup in the internal.
    pub fn ll_type(&self, reg: mil::Reg) -> mil::LLType {
        self.ll_types[reg.0 as usize]
    }

    pub fn block_len(&self, bid: cfg::BlockID) -> usize {
        self.schedule.of_block(bid).len()
    }

    pub fn find_input_chain(&self, root: mil::Reg) -> Vec<mil::Reg> {
        let mut is_visited = RegMap::for_program(self, false);
        let mut chain = Vec::new();

        let mut queue = vec![root];
        while let Some(reg) = queue.pop() {
            if is_visited[reg] {
                continue;
            }

            is_visited[reg] = true;
            chain.push(reg);

            let insn = self.get(reg).unwrap();
            for input in insn.input_regs_iter() {
                queue.push(input);
            }
        }

        chain
    }

    /// Check that the SSA program is valid and refresh cached derived data
    /// (such as each value's LLType).
    ///
    /// Any violations are recorded in `self.faults`.
    ///
    /// LLType's are also recomputed from scratch.
    pub fn check_invariants(&mut self) {
        // NOTE that LLTypes are NOT being reset here.
        // - they are assumed to have been reset by calling `reset_ll_types` right after construction;
        // - edits only allow changing instructions with ones with the same LLType.

        self.faults.clear();

        self.check_insns_valid();
        self.check_consistent_arrays_len();
        self.check_no_circular_refs();
        self.check_inputs_visible_scheduled();
        self.check_consistent_phis();

        #[cfg(test)]
        {
            if !self.faults.is_empty() {
                panic!("SSA invariants violated: {:#?}", self.faults);
            }
            if self.ll_types.contains(&mil::LLType::Error) {
                panic!("SSA contains type errors");
            }
        }
    }

    pub fn mutate<R>(&mut self, block: impl FnOnce(OpenProgram) -> R) -> R {
        let ret = block(OpenProgram { program: self });

        eliminate_dead_code(self);
        self.check_invariants();

        ret
    }

    pub fn faults(&self) -> impl '_ + ExactSizeIterator<Item = &Fault> {
        self.faults.iter()
    }
}

fn int_bytes(value: i64, size: u16, endianness: Endianness) -> Bytes {
    let size = size as usize;
    assert!(size <= 8, "very large int! {size} bytes");

    // although `value` can store integer values of up to 8 bytes, we should
    // treat it as an integer of exactly `size` bytes.
    //
    // to fit the two things together, we (1) mask the value
    let value = if size == 8 {
        value
    } else {
        value & ((1 << (8 * size)) - 1)
    };

    // (2) convert to 8 bytes (of the right endianness)
    // (3) cut `size` bytes (on the right side of the 8 byte sequence), and return it
    match endianness {
        Endianness::Little => {
            let bytes = value.to_le_bytes();
            Bytes::from_slice(&bytes[0..size]).unwrap()
        }
        Endianness::Big => {
            let bytes = value.to_be_bytes();
            Bytes::from_slice(&bytes[8 - size..]).unwrap()
        }
    }
}

#[cfg(test)]
#[test]
fn test_int_bytes() {
    assert_eq!(
        int_bytes(0xaabb, 2, Endianness::Little).as_slice(),
        &[0xbb, 0xaa]
    );
}

/// Internal API
impl Program {
    fn reset_ll_types(&mut self) {
        rt_infer::reset(self);
    }

    /// Check that each single instruction is (independently) valid.
    fn check_insns_valid(&mut self) {
        for reg in self.registers() {
            let insn = self.get(reg).unwrap();
            if !insn.is_valid() {
                self.faults.push(Fault::InvalidInsn(reg));
            }
        }
    }

    fn check_consistent_arrays_len(&mut self) {
        if self.insns.len() != self.addrs.len()
            || self.insns.len() != self.tyids.len()
            || self.insns.len() != self.ll_types.len()
        {
            self.faults.push(Fault::InconsistentArrayLengths);
        }
    }

    fn check_inputs_visible_scheduled(&mut self) {
        enum Cmd {
            Start(cfg::BlockID),
            End(cfg::BlockID),
        }
        let entry_bid = self.cfg().entry_block_id();
        let mut queue = vec![Cmd::End(entry_bid), Cmd::Start(entry_bid)];

        let dom_tree = self.cfg().dom_tree();
        let mut faults = Vec::new();

        // walk the dom tree depth-first

        // def_block[reg] = Some(bid) iff
        //
        // register <reg> is defined in block <bid>, which is a dominator of the
        // currently-visited block.
        // (as the depth-first search leaves a block, the corresponding
        // registers are cleared from this map)
        let mut def_block = RegMap::for_program(self, None);
        let mut block_visited = cfg::BlockMap::new(self.cfg(), false);

        while let Some(cmd) = queue.pop() {
            match cmd {
                Cmd::Start(bid) => {
                    assert!(!block_visited[bid]);
                    block_visited[bid] = true;

                    for reg in self.block_regs(bid) {
                        let insn = self.get(reg).unwrap();
                        for input in insn.input_regs_iter() {
                            if let Some(def_block_input) = def_block[input] {
                                if !(def_block_input == bid
                                    || dom_tree.imm_doms(bid).any(|b| b == def_block_input))
                                {
                                    faults.push(Fault::InputNotDominated {
                                        reg,
                                        input,
                                        def_block_input,
                                    });
                                }
                            } else {
                                faults.push(Fault::InputNotDefined { reg, input });
                            };
                        }

                        // otherwise it's a bug in this function:
                        assert!(def_block[reg].is_none());
                        def_block[reg] = Some(bid);
                    }

                    for &dominated in dom_tree.children_of(bid) {
                        queue.push(Cmd::End(dominated));
                        queue.push(Cmd::Start(dominated));
                    }
                }
                Cmd::End(bid) => {
                    for reg in self.block_regs(bid) {
                        def_block[reg] = None;
                    }
                }
            }
        }

        self.faults.extend(faults);
    }

    fn check_consistent_phis(&mut self) {
        // all Upsilons linked to the same Phi have the same regtype
        let mut phi_type = RegMap::for_program(self, None);

        for reg in self.registers() {
            let mil::Insn::Upsilon { value, phi_ref } = self.get(reg).unwrap() else {
                continue;
            };

            let ll_type = self.ll_type(*value);
            // Phi's are allowed to have one or more of their inputs be LLType::Error:
            // - they don't count for consistency;
            // - any Error already causes a panic in tests, so this is caught;
            // - they allow the type deduction algorithm to recover the problem
            //   "later" (with multiple xform cycles)
            if ll_type == mil::LLType::Error {
                continue;
            }

            match &mut phi_type[*phi_ref] {
                slot @ None => {
                    *slot = Some(ll_type);
                }
                Some(prev) => {
                    if *prev != ll_type {
                        self.faults.push(Fault::InconsistentPhiTypes {
                            phi_reg: *phi_ref,
                            value: *value,
                            expected: *prev,
                            found: ll_type,
                        });
                    }
                }
            }
        }
    }

    fn check_no_circular_refs(&mut self) {
        // kahn's algorithm for topological sorting, but we don't actually store
        // the topological order

        let mut rdr_count = count_readers_with_dead(self);
        let mut queue: Vec<_> = rdr_count
            .items()
            .filter(|(_, count)| **count == 0)
            .map(|(reg, _)| reg)
            .collect();

        let mut topo_order_len = 0;
        while let Some(reg) = queue.pop() {
            topo_order_len += 1;
            for input in self.get(reg).unwrap().input_regs_iter() {
                rdr_count[input] -= 1;
                if rdr_count[input] == 0 {
                    queue.push(input);
                }
            }
        }

        if topo_order_len < self.reg_count() as usize {
            self.faults.push(Fault::CircularRef);
        }
    }

    pub fn dump<S: std::fmt::Write>(
        &self,
        mut f: S,
        types: Option<&ty::TypeSet>,
    ) -> Result<(), std::fmt::Error> {
        let rdr_count = count_readers(self);

        let mut type_s = Vec::with_capacity(64);
        writeln!(f, "ssa program  {} instrs", self.reg_count())?;

        let mut cur_bid = None;
        for (bid, reg) in self.insns_rpo() {
            if cur_bid != Some(bid) {
                write!(f, ".B{}:    ;;", bid.as_usize())?;
                let preds = self.cfg.block_preds(bid);
                if !preds.is_empty() {
                    write!(f, " preds:")?;
                    for (ndx, pred) in preds.iter().enumerate() {
                        if ndx > 0 {
                            write!(f, ",")?;
                        }
                        write!(f, "B{}", pred.as_number())?;
                    }
                }
                write!(f, "  → {:?}", self.cfg.block_cont(bid))?;
                writeln!(f, ".")?;

                cur_bid = Some(bid);
            }

            let insn = self.get(reg).unwrap();
            if *insn == mil::Insn::Void {
                continue;
            }

            let rdr_count = rdr_count[reg];
            if rdr_count > 1 {
                write!(f, "  ({:3})  ", rdr_count)?;
            } else {
                write!(f, "         ")?;
            }

            if let Some(types) = types {
                if let Some(tyid) = self.value_type(reg) {
                    type_s.clear();
                    let mut pp = pp::PrettyPrinter::start(&mut type_s);
                    write!(pp, ": ").unwrap();

                    match types.read_tx() {
                        Ok(rtx) => {
                            rtx.read().dump_type_ref(&mut pp, tyid).unwrap();
                        }
                        Err(err) => {
                            write!(pp, "[error: {}]", err).unwrap();
                        }
                    }
                }
            }

            let type_s = std::str::from_utf8(&type_s).unwrap();
            writeln!(
                f,
                "{:?}: {:?}{} <- {:?}",
                reg,
                self.ll_type(reg),
                type_s,
                insn
            )?;
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct UpsilonDesc {
    upsilon_reg: mil::Reg,
    #[allow(dead_code)]
    input_reg: mil::Reg,
}

mod rt_infer {
    // Recompute low-level SSA register types.
    //
    // The non-obvious bit here is how Phi/Upsilon are handled:
    // - every `Upsilon { value, phi_ref }` means `value` and `phi_ref`
    //   must have the same `LLType`
    // - we therefore build union-find equivalence classes over registers
    // - non-Phi instructions contribute a type to their class
    // - Phi instructions do not infer locally; they read the type that was
    //   discovered for their whole class
    //
    // This is mainly here to make full-program `reset()` robust in the
    // presence of Phi chains/cycles, where purely local inference is too
    // order-sensitive.
    use super::*;

    use crate::util::disjoint_set::DisjointSet;
    use mil::{Insn, LLType};

    #[derive(Debug)]
    struct TypeInferenceState {
        // Union-find over registers that are constrained to have the same
        // low-level type by Phi/Upsilon flow.
        disjoint_set: DisjointSet,
        equiv_class_types: Vec<LLType>,
        reg_to_equiv_class: Vec<usize>,
    }

    impl TypeInferenceState {
        fn new(reg_count: usize) -> Self {
            // `equiv_class_types` starts at `Error` for every class.
            // In this module that means "no concrete class-wide type has been
            // established yet"; if conflicting concrete types are later seen,
            // we also collapse back to `Error`.
            Self {
                disjoint_set: DisjointSet::new(reg_count),
                equiv_class_types: vec![LLType::Error; reg_count],
                reg_to_equiv_class: vec![0; reg_count],
            }
        }

        fn build_equivalence_classes(&mut self, prog: &Program) {
            // `Upsilon` is the edge that feeds a value into a Phi. Type-wise,
            // that means `value` and `phi_ref` must be equal, so we union them.
            //
            // Doing this globally up front lets us type Phi cycles without
            // chasing them recursively or depending on a lucky visitation order.
            for (_block_id, reg) in prog.insns_rpo() {
                if let Insn::Upsilon { value, phi_ref } = prog.get(reg).unwrap() {
                    // Unite the value register and phi register in the same equivalence class
                    self.disjoint_set
                        .union(value.0 as usize, phi_ref.0 as usize);
                }
            }

            // Build the mapping from register to equivalence class representative
            for reg_idx in 0..self.reg_to_equiv_class.len() {
                let rep = self.disjoint_set.find(reg_idx);
                self.reg_to_equiv_class[reg_idx] = rep;
            }
        }

        fn get_equiv_class_type(&self, reg: mil::Reg) -> LLType {
            // All regs in the same class share one class-wide type slot.
            self.equiv_class_types[self.reg_to_equiv_class[reg.0 as usize]]
        }

        fn set_equiv_class_type(&mut self, reg: mil::Reg, ll_type: LLType) {
            // Merge this register's deduced type into its class-wide type.
            //
            // Rules:
            // - first concrete type seen for the class wins
            // - same type again is fine
            // - a different type poisons the whole class to `Error`
            let equiv_rep = self.reg_to_equiv_class[reg.0 as usize];
            if self.equiv_class_types[equiv_rep] == LLType::Error {
                // First assignment to this equivalence class
                self.equiv_class_types[equiv_rep] = ll_type;
            } else {
                // Check for type conflicts within the equivalence class
                if self.equiv_class_types[equiv_rep] != ll_type {
                    self.equiv_class_types[equiv_rep] = LLType::Error;
                }
            }
        }
    }

    #[tracing::instrument(skip_all)]
    pub(super) fn reset(prog: &mut Program) {
        // Full recomputation path.
        //
        // We first discover all Phi/Upsilon type-equivalence classes, then walk
        // the program and let non-Phi instructions populate those classes.
        // Phi instructions themselves just read back the class type.
        let mut state = TypeInferenceState::new(prog.reg_count().into());
        state.build_equivalence_classes(prog);

        // reverse post order, this way every instruction can 'see' the LLType
        // of its inputs
        let order: Vec<_> = prog.insns_rpo().map(|(_, reg)| reg).collect();
        for reg in order {
            update_reg(reg, prog, &mut state);
        }
    }

    fn update_reg(reg: mil::Reg, prog: &mut Program, state: &mut TypeInferenceState) -> LLType {
        // Update one register during full `reset()`.
        //
        // Non-Phi regs are typed from their opcode/operands. Phi regs are typed
        // indirectly via the precomputed equivalence class they belong to.
        let llt = match prog.get(reg).unwrap() {
            Insn::Phi => {
                // Phi does not infer locally. Instead it inherits the type of
                // the whole connected Phi/Upsilon component.
                let class_type = state.get_equiv_class_type(reg);
                if class_type == LLType::Error {
                    // Usually this means a "pure" Phi cycle: the class never saw
                    // a concrete non-Phi producer that could anchor its type.
                    // (It can also mean the class was type-inconsistent.)
                    prog.faults.push(Fault::PhiCycle { reg });
                }

                prog.ll_types[reg.0 as usize] = class_type;
                class_type
            }

            _ => deduce_one_reg(reg, prog),
        };

        event!(Level::TRACE, ?reg, ?llt, "updated low level type");

        // Update equivalence class type for this register
        state.set_equiv_class_type(reg, llt);
        llt
    }

    pub(super) fn deduce_one_reg(reg: mil::Reg, prog: &mut Program) -> LLType {
        // Local per-instruction type deduction.
        //
        // This is used both by full `reset()` for non-Phi instructions and by
        // incremental single-reg updates (`set`/`add_insn`). The incremental
        // path does *not* rebuild Phi/Upsilon equivalence classes, so Phi typing
        // remains a responsibility of the full reset pass.
        let ltt = match prog.get(reg).unwrap() {
            Insn::Void => LLType::Bytes(0),
            Insn::True => LLType::Bool,
            Insn::False => LLType::Bool,
            Insn::Int { size, .. } => LLType::Bytes(*size as usize),
            Insn::Bytes(bytes) => LLType::Bytes(bytes.len()),
            // TODO not machine independent and doesn't cover various cases,
            // but good enough for now
            Insn::Global(_) => LLType::Bytes(8),
            Insn::Part { size, .. } => LLType::Bytes(*size as usize),
            Insn::Get(arg) => prog.ll_type(*arg),
            Insn::Concat { lo, hi } => {
                let lo_type = prog.ll_type(*lo);
                let hi_type = prog.ll_type(*hi);

                match (lo_type.bytes_size(), hi_type.bytes_size()) {
                    (Some(lo_size), Some(hi_size)) => LLType::Bytes(lo_size + hi_size),
                    _ => {
                        prog.faults.push(Fault::ConcatNonBytesType {
                            reg,
                            found: lo_type,
                        });
                        LLType::Error
                    }
                }
            }
            Insn::Widen {
                reg: _,
                target_size,
                sign: _,
            } => LLType::Bytes(*target_size as usize),
            Insn::Arith(_, a, b) => {
                let at = prog.ll_type(*a);
                let bt = prog.ll_type(*b);

                if at == bt {
                    if let LLType::Bytes(sz) = at {
                        LLType::Bytes(sz)
                    } else {
                        prog.faults
                            .push(Fault::ArithNonBytesType { reg, found: at });
                        LLType::Error
                    }
                } else {
                    prog.faults.push(Fault::ArithDifferentTypes {
                        reg,
                        left: at,
                        right: bt,
                    });
                    LLType::Error
                }
            }
            Insn::ArithK(_, a, _) => {
                let at = prog.ll_type(*a);
                match at {
                    LLType::Bytes(_) => at,
                    _ => {
                        prog.faults
                            .push(Fault::ArithKNonBytesType { reg, found: at });
                        LLType::Error
                    }
                }
            }

            Insn::Cmp(_, _, _) => LLType::Bool,
            Insn::Bool(_, _, _) => LLType::Bool,
            Insn::Not(_) => LLType::Bool,
            // TODO This might have to change based on the use of calling
            // convention and function type info
            Insn::Call { ret_ll_type, .. } => *ret_ll_type,

            Insn::SetReturnValue(_)
            | Insn::SetJumpTarget(_)
            | Insn::SetJumpCondition(_)
            | Insn::Control(_)
            | Insn::NotYetImplemented(_)
            | Insn::StoreMem { .. }
            | Insn::Upsilon { .. } => LLType::Effect,

            Insn::LoadMem { size, .. } => LLType::Bytes(*size as usize),
            Insn::OverflowOf(_) => LLType::Bool,
            Insn::CarryOf(_) => LLType::Bool,
            Insn::SignOf(_) => LLType::Bool,
            Insn::IsZero(_) => LLType::Bool,
            Insn::Parity(_) => LLType::Bool,
            Insn::UndefinedBool => LLType::Bool,
            Insn::UndefinedBytes { size } => LLType::Bytes(*size as usize),
            Insn::FuncArgument { ll_type, .. } => *ll_type,
            Insn::Ancestral { ll_type, .. } => *ll_type,
            Insn::StructGetMember { size, .. } => LLType::Bytes(*size as usize),
            Insn::ArrayGetElement { size, .. } => LLType::Bytes(*size as usize),
            Insn::Struct { size, .. } => LLType::Bytes(*size as usize),

            // phis cannot be introduced after initial construction (and that's when phi's types are computed, during reset())
            Insn::Phi => unreachable!(),
        };

        prog.ll_types[reg.0 as usize] = ltt;
        ltt
    }
}

/// A wrapper of a Program, that allows editing/mutation.
///
/// # Notes on RegTypes
///
/// In order to maintain consistency, it is *forbidden* to change an instruction
/// to one with a different LLType, or in such a way that unresolvable cyclic
/// dependencies between RegTypes are introduced (via phi nodes that don't
/// have an independent LLType).
///
/// No problem for added instructions  (e.g. [Self::append_existing],
/// [Self::append_new]). (Implementation note: Some effort is required in terms
/// of bookkeeping to make this work without externally-visible hassle.)
///
/// # Panics
///
/// This type protects from internal bugs: on Drop, it will reassert internal
/// invariants and panic if any is violated.
pub struct OpenProgram<'a> {
    program: &'a mut Program,
}
pub type MutationResult<T> = std::result::Result<T, MutationError>;
#[derive(Error, Debug)]
pub enum MutationError {
    #[error("invalid register {reg:?}")]
    InvalidRegister { reg: mil::Reg },
    #[error("expected type {expected:?} but found {found:?} for register {reg:?}")]
    TypeError {
        reg: mil::Reg,
        expected: mil::LLType,
        found: mil::LLType,
    },
}
impl<'a> OpenProgram<'a> {
    /// Set the defining instruction for the given register.
    ///
    /// Returns `true` if the operation is completed successfully. Returns `false` on
    /// failure (the register is invalid for this SSA program.)
    ///
    /// Panics if the new instruction would change the register's LLType.
    pub fn set(&mut self, reg: mil::Reg, insn: mil::Insn) {
        use std::mem::discriminant;

        let Some(orig_insn) = self.insns.get(reg.reg_index() as usize).cloned() else {
            return;
        };
        if !orig_insn.is_replaceable() && discriminant(&orig_insn) != discriminant(&insn) {
            panic!(
                "instructions can't be replaced: {:?} -> {:?}",
                orig_insn, insn
            );
        }
        if orig_insn == insn {
            return;
        }

        self.program.insns[reg.reg_index() as usize] = insn.clone();
        event!(Level::TRACE, ?orig_insn, ?insn, "insn set");

        let old_rt = self.ll_type(reg);
        let new_rt = rt_infer::deduce_one_reg(reg, &mut self.program);
        assert_eq!(
            old_rt, new_rt,
            "{reg:?} changed type (left: {orig_insn:?} -> right {insn:?})"
        );
    }

    /// Add a new instruction to the set of values
    ///
    /// # Important note
    ///
    /// After calling this function, the returned Reg is valid (i.e. assigned
    /// to an instruction), but it is not scheduled yet.
    ///
    /// Schedule the instruction by calling `append_existing`. Failing to
    /// do so before dropping this [OpenProgram] will result in a panic in
    /// [Program::assert_invariants].
    fn add_insn(&mut self, insn: mil::Insn) -> mil::Reg {
        let ndx = self.insns.len().try_into().unwrap();
        let reg = mil::Reg(ndx);

        self.program.insns.push(insn.clone());
        self.program.addrs.push(u64::MAX);
        self.program.tyids.push(None);
        self.program.ll_types.push(mil::LLType::Effect);
        // adds faults if type inference encounters errors
        rt_infer::deduce_one_reg(reg, &mut self.program);
        let ltt = self.program.ll_type(reg);
        event!(Level::TRACE, reg = ?reg, insn = ?&self.program.insns[reg.reg_index() as usize], ?ltt, "add insn");

        reg
    }

    pub fn clear_block_schedule(&mut self, bid: cfg::BlockID) -> Vec<mil::Reg> {
        self.program
            .schedule
            .clear_block(bid)
            .into_iter()
            .map(|ndx| mil::Reg(ndx))
            .collect()
    }

    /// Add a new instruction to the data flow graph, and schedule it at the end
    /// of the given basic block.
    ///
    /// Returns the register corresponding to the new value.
    pub fn append_new(&mut self, bid: cfg::BlockID, insn: mil::Insn) -> mil::Reg {
        let reg = self.add_insn(insn);
        self.program.schedule.append(reg.reg_index(), bid);
        reg
    }
    /// Append an existing instruction (identified by the given [Reg]) at the end
    /// of the given basic block.
    ///
    /// # Note
    ///
    /// It's the caller's responsibility to make sure that the given register is:
    /// 1. valid, and
    /// 2. scheduled at exactly one position in the program.
    ///
    /// The above invariants are checked when the [OpenProgram] is dropped.
    /// Failing to uphold them will result in a panic at that time.
    pub fn append_existing(&mut self, bid: cfg::BlockID, reg: mil::Reg) {
        self.program.schedule.append(reg.reg_index(), bid);
    }

    pub fn swap(&mut self, a: mil::Reg, b: mil::Reg) {
        self.program
            .insns
            .swap(a.reg_index() as usize, b.reg_index() as usize);
    }

    pub fn set_value_type(&mut self, reg: mil::Reg, tyid: Option<ty::TypeID>) {
        self.program.tyids[reg.0 as usize] = tyid;
    }

    pub fn append_before(&mut self, reg: mil::Reg, insn: mil::Insn) -> mil::Reg {
        let new_reg = self.add_insn(insn);

        // TODO There is supposed to be a data structure that allows us to do this very efficiently
        self.program
            .schedule
            .insert_before(new_reg.reg_index(), reg.reg_index());

        new_reg
    }

    pub fn invert_bool(&mut self, cond: mil::Reg) -> MutationResult<mil::Reg> {
        let ll_type = self.program.ll_type(cond);
        if ll_type != LLType::Bool {
            return Err(MutationError::TypeError {
                reg: cond,
                expected: LLType::Bool,
                found: ll_type,
            });
        }

        let new_cond = match self.program.get(cond).unwrap() {
            mil::Insn::Not(inner) => *inner,
            _ => self.append_before(cond, mil::Insn::Not(cond)),
        };

        Ok(new_cond)
    }
}
impl std::ops::Deref for OpenProgram<'_> {
    type Target = Program;
    fn deref(&self) -> &Self::Target {
        self.program
    }
}

pub fn eliminate_dead_code(prog: &mut Program) {
    let mut is_read = RegMap::for_program(prog, false);
    let mut work = Vec::new();

    // initialize worklist with effectful instructions, purposefully skipping
    // Upsilons (only processed as a consequence of processing the corresponding
    // phi)
    for reg in prog.registers() {
        let insn = prog.get(reg).unwrap();
        if insn.has_side_effects() && !matches!(insn, mil::Insn::Upsilon { .. }) {
            work.push(reg);
        }
    }

    while let Some(reg) = work.pop() {
        if is_read[reg] {
            continue;
        }

        is_read[reg] = true;

        let insn = prog.get(reg).unwrap();
        for input in insn.input_regs_iter() {
            work.push(input);
        }
        if matches!(insn, mil::Insn::Phi) {
            for UpsilonDesc { upsilon_reg, .. } in prog.upsilons_of_phi(reg) {
                work.push(upsilon_reg);
            }
        }
    }

    prog.schedule.retain(|_, ndx| is_read[mil::Reg(ndx)]);
}

#[cfg(test)]
#[test]
#[should_panic]
fn test_assert_no_circular_refs() {
    use mil::{ArithOp, Insn, Reg};

    let mut prog = mil::Program::new(Reg(0), None);
    prog.set_input_addr(0xf0);
    prog.push(
        Reg(0),
        Insn::Int {
            value: 123,
            size: 8,
        },
    );
    prog.push(Reg(1), Insn::Arith(ArithOp::Add, Reg(0), Reg(2)));
    prog.push(Reg(2), Insn::Arith(ArithOp::Add, Reg(0), Reg(1)));

    let mut prog = Program::from_mil(prog);
    prog.check_no_circular_refs();
}

impl std::fmt::Debug for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.dump(f, None)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegMap<T>(Vec<T>);

impl<T: Clone> RegMap<T> {
    pub fn for_program(prog: &Program, init: T) -> Self {
        let inner = vec![init; prog.reg_count() as usize];
        RegMap(inner)
    }

    pub fn empty() -> Self {
        RegMap(Vec::new())
    }

    pub fn reg_count(&self) -> u16 {
        self.0.len().try_into().unwrap()
    }

    pub fn fill(&mut self, value: T) {
        self.0.fill(value);
    }

    pub fn items(&self) -> impl '_ + ExactSizeIterator<Item = (mil::Reg, &'_ T)> {
        self.0
            .iter()
            .enumerate()
            .map(|(ndx, item)| (mil::Reg(ndx.try_into().unwrap()), item))
    }

    pub fn map<F, R>(&self, mut f: F) -> RegMap<R>
    where
        F: FnMut(mil::Reg, &T) -> R,
        R: Clone,
    {
        let elements: Vec<_> = self.items().map(|(reg, value)| f(reg, value)).collect();
        assert_eq!(elements.len(), self.0.len());
        RegMap(elements)
    }
}
// unit tests for empty RegMap:
#[cfg(test)]
mod regmap_tests {
    use super::*;

    #[test]
    fn test_empty_regmap() {
        let regmap: RegMap<usize> = RegMap::empty();
        assert_eq!(regmap.reg_count(), 0);
        let mapped = regmap.map(|reg, val| (reg, *val));
        assert_eq!(mapped.reg_count(), 0);
    }

    #[test]
    fn test_empty_map_fill_and_items() {
        let mut regmap: RegMap<i32> = RegMap::empty();
        // fill should be a no-op and must not panic on empty map
        regmap.fill(42);
        assert_eq!(regmap.reg_count(), 0);
        assert_eq!(regmap.items().count(), 0);
        assert!(regmap.items().next().is_none());
    }

    #[test]
    fn test_empty_map_map_and_clone() {
        let regmap: RegMap<usize> = RegMap::empty();
        let mapped: RegMap<(mil::Reg, usize)> = regmap.map(|reg, val| (reg, *val));
        assert_eq!(mapped.reg_count(), 0);

        // cloning an empty RegMap preserves emptiness
        let cloned = regmap.clone();
        assert_eq!(cloned.reg_count(), 0);
    }
}

impl<T> std::ops::Index<mil::Reg> for RegMap<T> {
    type Output = T;

    fn index(&self, index: mil::Reg) -> &Self::Output {
        &self.0[index.0 as usize]
    }
}

impl<T> std::ops::IndexMut<mil::Reg> for RegMap<T> {
    fn index_mut(&mut self, index: mil::Reg) -> &mut Self::Output {
        &mut self.0[index.0 as usize]
    }
}

/// Count the number of "readers" (dependent) instructions for each instruction
/// in the Program, with no regard for whether these readers are dead or alive.
///
/// This function always returns.
pub fn count_readers_with_dead(prog: &Program) -> RegMap<usize> {
    let mut count = RegMap::for_program(prog, 0);

    // This function must not use Program::insns_rpo, as this is used to check
    // for circular graphs, and circular graphs make insns_rpo hang in an
    // infinite loop

    for reg in prog.registers() {
        for input in prog.get(reg).unwrap().input_regs_iter() {
            count[input] += 1;
        }
    }

    count
}

/// Count the number of live "readers" (dependent) instructions for each
/// instruction in the Program.
///
/// Dead instructions have no direct or indirect live dependent, and are
/// therefore counted as 0.
///
/// **Note:** If `prog` has circular dependencies, this function hangs forever.
/// This situation is invalid and universally considered as a bug. It should be
/// ruled out by calling [Program::assert_invariants] (and this API makes an
/// effort to make sure invariants are asserted after every change to the
/// [Program]).
pub fn count_readers(prog: &Program) -> RegMap<usize> {
    let mut count = vec![0; prog.reg_count() as usize];

    for (_, reg) in prog.insns_rpo() {
        for reg in prog.get(reg).unwrap().input_regs() {
            count[reg.reg_index() as usize] += 1;
        }
    }

    RegMap(count)
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    #[error("inconsistent array lengths in Program")]
    InconsistentArrayLengths,

    #[error("input {input:?} of {reg:?} is not defined")]
    InputNotDefined { reg: mil::Reg, input: mil::Reg },

    #[error("input {input:?} of {reg:?} is not dominated by its definition in block {def_block_input:?}")]
    InputNotDominated {
        reg: mil::Reg,
        input: mil::Reg,
        def_block_input: cfg::BlockID,
    },

    #[error("inconsistent types for phi node {phi_reg:?}: expected {expected:?}, found {found:?}")]
    InconsistentPhiTypes {
        phi_reg: mil::Reg,
        expected: mil::LLType,
        found: mil::LLType,
        value: mil::Reg,
    },

    #[error("circular reference detected in data flow graph")]
    CircularRef,

    #[error("instruction is ill-formed")]
    InvalidInsn(mil::Reg),

    #[error("{reg:?}: Insn::Concat: expected LLType::Bytes, found {found:?}")]
    ConcatNonBytesType { reg: mil::Reg, found: mil::LLType },

    #[error("{reg:?}: Insn::Arith: operands have different types ({left:?}, {right:?})")]
    ArithDifferentTypes {
        reg: mil::Reg,
        left: mil::LLType,
        right: mil::LLType,
    },

    #[error("{reg:?}: Insn::Arith: operand must be LLType::Bytes, found {found:?}")]
    ArithNonBytesType { reg: mil::Reg, found: mil::LLType },

    #[error("{reg:?}: Insn::ArithK: operand must be LLType::Bytes, found {found:?}")]
    ArithKNonBytesType { reg: mil::Reg, found: mil::LLType },

    #[error("pathologic data flow graph: phi {reg:?} can't be resolved, it's an all-phi cycle")]
    PhiCycle { reg: mil::Reg },
}

#[cfg(test)]
mod input_chain_tests {
    use super::*;
    use crate::mil::R;

    fn mk_simple_program() -> Program {
        let mut prog = mil::Program::new(R(20), None);
        prog.push(R(0), mil::Insn::Int { value: 8, size: 8 });
        prog.push(R(1), mil::Insn::Int { value: 9, size: 8 });
        prog.push(R(0), mil::Insn::Arith(mil::ArithOp::Mul, R(0), R(0)));
        prog.push(
            R(0),
            mil::Insn::LoadMem {
                addr: R(0),
                size: 8,
            },
        );
        Program::from_mil(prog)
    }

    #[test]
    fn zero() {
        let prog = mk_simple_program();
        let chain = prog.find_input_chain(R(0));
        assert_eq!(&chain, &[R(0)]);

        let chain = prog.find_input_chain(R(1));
        assert_eq!(&chain, &[R(1)]);
    }

    #[test]
    fn simple() {
        let prog = mk_simple_program();
        let chain = prog.find_input_chain(R(3));
        assert_eq!(&chain, &[R(3), R(2), R(0)]);
    }
}

#[cfg(test)]
mod tests {

    use crate::{mil, LLType};
    use mil::{ArithOp, BoolOp, Control, Insn, Reg};

    #[test]
    fn test_phi_read() {
        let mut prog = mil::Program::new(Reg(0), None);

        prog.set_input_addr(0xf0);
        prog.push(
            Reg(0),
            Insn::Int {
                value: 123,
                size: 8,
            },
        );
        prog.push(Reg(1), Insn::SetJumpCondition(Reg(0)));
        prog.push(Reg(1), Insn::Control(Control::JmpExtIf(0xf2)));

        prog.set_input_addr(0xf1);
        prog.push(Reg(2), Insn::Int { value: 4, size: 1 });
        prog.push(Reg(1), Insn::Control(Control::JmpExt(0xf3)));

        prog.set_input_addr(0xf2);
        prog.push(Reg(2), Insn::Int { value: 8, size: 1 });

        prog.set_input_addr(0xf3);
        prog.push(Reg(4), Insn::ArithK(ArithOp::Add, Reg(2), 456));
        prog.push(Reg(5), Insn::SetReturnValue(Reg(4)));
        prog.push(Reg(5), Insn::Control(Control::Ret));

        let prog = super::Program::from_mil(prog);

        assert_eq!(prog.get(Reg(9)), Some(&Insn::Phi));

        let mut upss: Vec<_> = prog.upsilons_of_phi(Reg(9)).collect();
        upss.sort_by_key(|ud| ud.input_reg.reg_index());
        assert_eq!(upss[0].input_reg, Reg(3));
        assert_eq!(upss[1].input_reg, Reg(5));
        assert_eq!(upss.len(), 2);
    }

    #[test]
    fn circular_graph_detected_neg() {
        let mut prog = make_prog_no_cycles();
        prog.check_no_circular_refs();
        assert!(prog.faults.is_empty());
    }

    #[test]
    #[should_panic]
    fn circular_graph_detected_pos() {
        let mut prog = make_prog_no_cycles();
        prog.mutate(|mut prog| {
            // introduce cycle:
            prog.set(Reg(0), Insn::Arith(mil::ArithOp::Add, Reg(1), Reg(2)));
        });
    }

    fn make_prog_no_cycles() -> super::Program {
        let mut prog = mil::Program::new(Reg(0), None);
        prog.push(Reg(0), Insn::Int { value: 5, size: 8 });
        prog.push(Reg(1), Insn::Int { value: 5, size: 8 });
        prog.push(Reg(0), Insn::Arith(mil::ArithOp::Add, Reg(1), Reg(0)));
        prog.push(Reg(0), Insn::SetReturnValue(Reg(0)));
        prog.push(Reg(0), Insn::Control(Control::Ret));
        // SSA conversion would fail if there was a cycle
        super::Program::from_mil(prog)
    }

    #[test]
    fn test_machine_addr() {
        let mut prog = mil::Program::new(Reg(0), None);

        prog.set_input_addr(0x1000);
        prog.push(Reg(0), Insn::Int { value: 42, size: 8 });

        prog.set_input_addr(0x1008);
        prog.push(Reg(1), Insn::Int { value: 24, size: 8 });

        prog.set_input_addr(0x1010);
        prog.push(Reg(2), Insn::Arith(mil::ArithOp::Add, Reg(0), Reg(1)));

        let ssa_prog = super::Program::from_mil(prog);

        assert_eq!(ssa_prog.machine_addr(Reg(0)), Some(0x1000));
        assert_eq!(ssa_prog.machine_addr(Reg(1)), Some(0x1008));
        assert_eq!(ssa_prog.machine_addr(Reg(2)), Some(0x1010));
        assert_eq!(ssa_prog.machine_addr(Reg(3)), None); // Invalid reg
    }

    #[test]
    #[should_panic]
    fn ll_type_invalidation() {
        let mut prog = {
            let mut prog = mil::Program::new(Reg(0), None);
            prog.push(Reg(0), Insn::Int { value: 42, size: 8 });
            prog.push(Reg(1), Insn::Get(mil::Reg(0)));
            prog.push(Reg(2), Insn::Get(mil::Reg(1)));
            prog.push(Reg(3), Insn::SetReturnValue(mil::Reg(2)));
            super::Program::from_mil(prog)
        };

        assert_eq!(prog.ll_type(mil::Reg(0)), LLType::Bytes(8));
        assert_eq!(prog.ll_type(mil::Reg(1)), LLType::Bytes(8));
        assert_eq!(prog.ll_type(mil::Reg(2)), LLType::Bytes(8));

        prog.mutate(|mut prog| {
            // with this, the type of r0 changes, and so should the other two
            // registers' types.
            prog.set(mil::Reg(0), Insn::Int { value: 42, size: 4 });
        });
    }

    #[test]
    #[should_panic]
    fn test_arith_different_types() {
        let mut prog = mil::Program::new(Reg(0), None);
        prog.push(Reg(0), Insn::Int { value: 42, size: 8 }); // LLType::Bytes(8)
        prog.push(Reg(1), Insn::Bool(BoolOp::And, Reg(0), Reg(0))); // LLType::Bool
        prog.push(Reg(2), Insn::Arith(ArithOp::Add, Reg(0), Reg(1))); // Should cause fault
        super::Program::from_mil(prog);
    }

    #[test]
    #[should_panic]
    fn test_arithk_non_bytes() {
        let mut prog = mil::Program::new(Reg(0), None);
        prog.push(Reg(0), Insn::Bool(BoolOp::And, Reg(0), Reg(0))); // LLType::Bool
        prog.push(Reg(1), Insn::ArithK(ArithOp::Add, Reg(0), 1)); // Should cause fault
        super::Program::from_mil(prog);
    }
}
