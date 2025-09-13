//! Static Single-Assignment representation of a program (and conversion from direct multiple
//! assignment).
//!
//! The algorithms in this module are mostly derived from the descriptions in:
//! > Cooper, Keith & Harvey, Timothy & Kennedy, Ken. (2006).
//! > A Simple, Fast Dominance Algorithm.
//! > Rice University, CS Technical Report 06-33870.

use crate::{cfg, mil, pp, ty};
use std::{cell::Cell, io::Write};

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
    insns: Vec<Cell<mil::Insn>>,
    addrs: Vec<u64>,

    /// Type ID for the instruction's result.
    ///
    /// Managed externally (via getter `value_type` and setter `set_value_type`).
    ///
    /// The default for all instruction after initialization or insertion after
    /// initialization is None, which represents lack of type information.
    ///
    /// This is intended to be manipulated by specific passes in `xform`.
    tyids: Vec<Option<ty::TypeID>>,

    schedule: cfg::Schedule,
    cfg: cfg::Graph,
}

/// Correctness relies on a few invariants.
///
/// In order to make it feasible to assert them, every API in this impl only gives
/// read-only access. Every modification (1) requires a &mut Program; (2) requires
/// that the &mut Program is wrapped in a temporary [OpenProgram], which is the
/// only type provding editing APIs.
impl Program {
    pub fn from_mil(mil: mil::Program) -> Self {
        cons::mil_to_ssa(mil)
    }

    pub fn cfg(&self) -> &cfg::Graph {
        &self.cfg
    }

    /// Get the defining instruction for the given register.
    ///
    /// (Note that it's not allowed to fetch instructions by position.)
    pub fn get(&self, reg: mil::Reg) -> Option<mil::Insn> {
        // In SSA, Reg(ndx) happens to be located at index ndx.
        // if this  slot is enabled as per the mask, then every Vec access must succeed
        self.insns
            .get(reg.reg_index() as usize)
            .map(|cell| cell.get())
    }

    /// Set the defining instruction for the given register.
    ///
    /// Returns `true` if the operation is completed successfully. Returns `false` on
    /// failure (the register is invalid for this SSA program.)
    pub fn set(&self, reg: mil::Reg, insn: mil::Insn) -> bool {
        let Some(cell) = self.insns.get(reg.reg_index() as usize) else {
            return false;
        };
        cell.set(insn);
        true
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

    pub fn get_call_args(&self, arg: mil::Reg) -> impl '_ + Iterator<Item = mil::Reg> {
        let mut arg = Some(arg);
        std::iter::repeat_with(move || {
            let insn = self.get(arg?).unwrap();
            let mil::Insn::CArg { value, next_arg } = insn else {
                panic!("CArg must be chained to other CArgs only")
            };
            arg = next_arg;
            Some(value)
        })
        .map_while(|x| x)
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
        P: Fn(mil::Insn) -> Option<R>,
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
                mil::Insn::Upsilon { value, phi_ref } if phi_ref == phi_reg => Some(UpsilonDesc {
                    upsilon_reg: reg,
                    input_reg: value,
                }),
                _ => None,
            })
    }

    pub fn reg_type(&self, reg: mil::Reg) -> mil::RegType {
        use mil::{Insn, RegType};
        match self.get(reg).unwrap() {
            Insn::Void => RegType::Bytes(0), // TODO better choice here?
            Insn::True => RegType::Bool,
            Insn::False => RegType::Bool,
            Insn::Const { size, .. } => RegType::Bytes(size as usize),
            Insn::Part { size, .. } => RegType::Bytes(size as usize),
            Insn::Get(arg) => self.reg_type(arg),
            Insn::Concat { lo, hi } => {
                let lo_size = self.reg_type(lo).bytes_size().unwrap();
                let hi_size = self.reg_type(hi).bytes_size().unwrap();
                RegType::Bytes(lo_size + hi_size)
            }
            Insn::Widen {
                reg: _,
                target_size,
                sign: _,
            } => RegType::Bytes(target_size as usize),
            Insn::Arith(_, a, b) => {
                let at = self.reg_type(a);
                let bt = self.reg_type(b);
                assert_eq!(at, bt); // TODO check this some better way
                at
            }
            Insn::ArithK(_, a, _) => self.reg_type(a),
            Insn::Cmp(_, _, _) => RegType::Bool,
            Insn::Bool(_, _, _) => RegType::Bool,
            Insn::Not(_) => RegType::Bool,
            // TODO This might have to change based on the use of calling
            // convention and function type info
            Insn::Call { .. } => RegType::Bytes(8),
            Insn::CArg { value, next_arg: _ } => self.reg_type(value),

            Insn::SetReturnValue(_)
            | Insn::SetJumpTarget(_)
            | Insn::SetJumpCondition(_)
            | Insn::Control(_)
            | Insn::NotYetImplemented(_)
            | Insn::StoreMem { .. }
            | Insn::Upsilon { .. } => RegType::Effect,

            Insn::LoadMem { size, .. } => RegType::Bytes(size as usize),
            Insn::OverflowOf(_) => RegType::Bool,
            Insn::CarryOf(_) => RegType::Bool,
            Insn::SignOf(_) => RegType::Bool,
            Insn::IsZero(_) => RegType::Bool,
            Insn::Parity(_) => RegType::Bool,
            Insn::UndefinedBool => RegType::Bool,
            Insn::UndefinedBytes { size } => RegType::Bytes(size as usize),
            Insn::Phi => {
                let mut ys = self.upsilons_of_phi(reg);
                let Some(y) = ys.next() else {
                    panic!("no upsilons for this phi? {:?}", reg)
                };
                // assuming that all types are the same, as per assert_phis_consistent
                self.reg_type(y.input_reg)
            }
            Insn::FuncArgument { reg_type, .. } => reg_type,
            Insn::Ancestral { reg_type, .. } => reg_type,
            Insn::StructGetMember { size, .. } => RegType::Bytes(size as usize),
            Insn::ArrayGetElement { size, .. } => RegType::Bytes(size as usize),
        }
    }

    /// Return the TypeID of the given register (or, equivalently, of the given
    /// register's defining instruction's result).
    ///
    /// Returns None if `reg` is invalid.
    pub fn value_type(&self, reg: mil::Reg) -> Option<ty::TypeID> {
        self.tyids.get(reg.0 as usize).copied().flatten()
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

            let mut insn = self.get(reg).unwrap();
            for &mut input in insn.input_regs_iter() {
                queue.push(input);
            }
        }

        chain
    }

    pub fn assert_invariants(&self) {
        self.assert_consistent_arrays_len();
        self.assert_no_circular_refs();
        self.assert_inputs_visible_scheduled();
        self.assert_consistent_phis();
        self.assert_carg_chain();
    }
}

/// Internal API
impl Program {
    fn assert_consistent_arrays_len(&self) {
        assert_eq!(self.insns.len(), self.addrs.len());
        assert_eq!(self.insns.len(), self.tyids.len());
    }

    fn assert_inputs_visible_scheduled(&self) {
        enum Cmd {
            Start(cfg::BlockID),
            End(cfg::BlockID),
        }
        let entry_bid = self.cfg().entry_block_id();
        let mut queue = vec![Cmd::End(entry_bid), Cmd::Start(entry_bid)];

        let dom_tree = self.cfg().dom_tree();

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

                    for &ndx in self.schedule.of_block(bid) {
                        let reg = mil::Reg(ndx);

                        let mut insn = self.get(reg).unwrap();
                        for &mut input in insn.input_regs_iter() {
                            let Some(def_block_input) = def_block[input] else {
                                panic!("input {input:?} of {reg:?} is not defined");
                            };
                            assert!(
                                def_block_input == bid
                                    || dom_tree.imm_doms(bid).any(|b| b == def_block_input)
                            );
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
                    for &ndx in self.schedule.of_block(bid) {
                        let reg = mil::Reg(ndx);
                        def_block[reg] = None;
                    }
                }
            }
        }
    }

    fn assert_consistent_phis(&self) {
        // all Upsilons linked to the same Phi have the same regtype
        let mut phi_type = RegMap::for_program(self, None);
        let rdr_count = count_readers(self);

        for reg in self.registers() {
            if rdr_count[reg] == 0 {
                continue;
            }
            if let mil::Insn::Upsilon { value, phi_ref } = self.get(reg).unwrap() {
                let reg_type = self.reg_type(value);
                match &mut phi_type[phi_ref] {
                    slot @ None => {
                        *slot = Some(reg_type);
                    }
                    Some(prev) => {
                        assert_eq!(*prev, reg_type);
                    }
                }
            }
        }
    }

    fn assert_no_circular_refs(&self) {
        // kahn's algorithm for topological sorting, but we don't actually store the topological order

        let mut rdr_count = count_readers_with_dead(self);
        let mut queue: Vec<_> = rdr_count
            .items()
            .filter(|(_, count)| **count == 0)
            .map(|(reg, _)| reg)
            .collect();

        while let Some(reg) = queue.pop() {
            assert_eq!(rdr_count[reg], 0);

            for &mut input in self.get(reg).unwrap().input_regs_iter() {
                rdr_count[input] -= 1;
                if rdr_count[input] == 0 {
                    queue.push(input);
                }
            }
        }

        assert!(rdr_count.items().all(|(_, count)| *count == 0));
    }

    fn assert_carg_chain(&self) {
        for (_, reg) in self.insns_rpo() {
            let insn = self.get(reg).unwrap();
            let arg = match insn {
                mil::Insn::Call {
                    callee: _,
                    first_arg: Some(arg),
                }
                | mil::Insn::CArg {
                    value: _,
                    next_arg: Some(arg),
                } => arg,
                _ => continue,
            };

            let arg_def = self.get(arg).unwrap();
            assert!(
                matches!(arg_def, mil::Insn::CArg { .. }),
                "reg {:?} does not point to a CArg, but to {:?}",
                reg,
                arg_def
            );
        }
    }

    fn dump<S: std::fmt::Write>(
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
            if insn == mil::Insn::Void {
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
                    types.dump_type_ref(&mut pp, tyid).unwrap();
                }
            }

            let type_s = std::str::from_utf8(&type_s).unwrap();
            writeln!(f, "{:?}{} <- {:?}", reg, type_s, insn)?;
        }

        Ok(())
    }
}

pub struct UpsilonDesc {
    upsilon_reg: mil::Reg,
    input_reg: mil::Reg,
}

/// A wrapper for a Program that allows editing/mutation.
///
/// # Panics
///
/// This type protects from internal bugs: on Drop, it will reassert internal
/// invariants and panic if any is violated.
pub struct OpenProgram<'a> {
    program: &'a mut Program,
}
impl<'a> OpenProgram<'a> {
    pub fn wrap(program: &'a mut Program) -> Self {
        OpenProgram { program }
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
        self.program.insns.push(Cell::new(insn));
        self.program.addrs.push(u64::MAX);
        self.program.tyids.push(None);
        mil::Reg(ndx)
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

    pub fn set_value_type(&mut self, reg: mil::Reg, tyid: Option<ty::TypeID>) {
        self.program.tyids[reg.0 as usize] = tyid;
    }
}
impl Drop for OpenProgram<'_> {
    fn drop(&mut self) {
        eliminate_dead_code(&mut self.program);
        self.program.assert_invariants();
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

        let mut insn = prog.get(reg).unwrap();
        for &mut input in insn.input_regs() {
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

    let mut prog = mil::Program::new(Reg(0));
    prog.set_input_addr(0xf0);
    prog.push(
        Reg(0),
        Insn::Const {
            value: 123,
            size: 8,
        },
    );
    prog.push(Reg(1), Insn::Arith(ArithOp::Add, Reg(0), Reg(2)));
    prog.push(Reg(2), Insn::Arith(ArithOp::Add, Reg(0), Reg(1)));

    let prog = Program::from_mil(prog);
    prog.assert_no_circular_refs();
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
        for &mut input in prog.get(reg).unwrap().input_regs_iter() {
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

#[cfg(test)]
mod input_chain_tests {
    use super::*;
    use crate::mil::R;

    fn mk_simple_program() -> Program {
        let mut prog = mil::Program::new(R(20));
        prog.push(R(0), mil::Insn::Const { value: 8, size: 8 });
        prog.push(R(1), mil::Insn::Const { value: 9, size: 8 });
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

    use crate::mil;
    use mil::{ArithOp, Control, Insn, Reg};

    #[test]
    fn test_phi_read() {
        let mut prog = mil::Program::new(Reg(0));

        prog.set_input_addr(0xf0);
        prog.push(
            Reg(0),
            Insn::Const {
                value: 123,
                size: 8,
            },
        );
        prog.push(Reg(1), Insn::SetJumpCondition(Reg(0)));
        prog.push(Reg(1), Insn::Control(Control::JmpExtIf(0xf2)));

        prog.set_input_addr(0xf1);
        prog.push(Reg(2), Insn::Const { value: 4, size: 1 });
        prog.push(Reg(1), Insn::Control(Control::JmpExt(0xf3)));

        prog.set_input_addr(0xf2);
        prog.push(Reg(2), Insn::Const { value: 8, size: 1 });

        prog.set_input_addr(0xf3);
        prog.push(Reg(4), Insn::ArithK(ArithOp::Add, Reg(2), 456));
        prog.push(Reg(5), Insn::SetReturnValue(Reg(4)));
        prog.push(Reg(5), Insn::Control(Control::Ret));

        let prog = super::Program::from_mil(prog);

        assert_eq!(prog.get(Reg(9)), Some(Insn::Phi));

        let mut upss: Vec<_> = prog.upsilons_of_phi(Reg(9)).collect();
        upss.sort_by_key(|ud| ud.input_reg.reg_index());
        assert_eq!(upss[0].input_reg, Reg(3));
        assert_eq!(upss[1].input_reg, Reg(5));
        assert_eq!(upss.len(), 2);
    }

    #[test]
    fn circular_graph_detected_neg() {
        let prog = make_prog_no_cycles();
        prog.assert_no_circular_refs();
    }

    #[test]
    #[should_panic]
    fn circular_graph_detected_pos() {
        let prog = make_prog_no_cycles();
        // introduce cycle:
        prog.set(Reg(0), Insn::Arith(mil::ArithOp::Add, Reg(1), Reg(2)));
        prog.assert_no_circular_refs();
    }

    fn make_prog_no_cycles() -> super::Program {
        let mut prog = mil::Program::new(Reg(0));
        prog.push(Reg(0), Insn::Const { value: 5, size: 8 });
        prog.push(Reg(1), Insn::Const { value: 5, size: 8 });
        prog.push(Reg(0), Insn::Arith(mil::ArithOp::Add, Reg(1), Reg(0)));
        prog.push(Reg(0), Insn::SetReturnValue(Reg(0)));
        prog.push(Reg(0), Insn::Control(Control::Ret));

        // SSA conversion would fail if there was a cycle
        super::Program::from_mil(prog)
    }
}
