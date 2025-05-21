use std::{cell::Cell, io::Write, ops::Range};

/// Static Single-Assignment representation of a program (and conversion from direct multiple
/// assignment).
///
/// The algorithms in this module are mostly derived from the descriptions in:
/// > Cooper, Keith & Harvey, Timothy & Kennedy, Ken. (2006).
/// > A Simple, Fast Dominance Algorithm.
/// > Rice University, CS Technical Report 06-33870.
use crate::{cfg, mil, pp, ty};

#[derive(Clone)]
pub struct Program {
    // an ssa::Program contains a mil::Program at its core, but never exposes it directly:
    //
    // - extra instructions are appended for various uses, although they do not belong
    //   (directly) to the program sequence (they're only referred-to by other in-sequence
    //   instruction)
    //
    // - SSA registers are numerically equal to the index of the defining
    //   instruction in `inner`. mil::Index values are just as good as mil::Reg
    //   for identifying both insns and values. This allows for fast lookups.
    //
    // - the logical order of the program (i.e. the order in which insns would
    //   be executed if this was a real vm) is entirely independent of the
    //   indices in `inner`. the logical order is stored here, with covering
    //   spans associated to each block.
    //
    inner: mil::Program,

    schedule: cfg::Schedule,
    cfg: cfg::Graph,
}

slotmap::new_key_type! { pub struct TypeID; }

impl Program {
    pub fn cfg(&self) -> &cfg::Graph {
        &self.cfg
    }

    /// Get the defining instruction for the given register.
    ///
    /// (Note that it's not allowed to fetch instructions by position.)
    pub fn get(&self, reg: mil::Reg) -> Option<mil::InsnView> {
        // In SSA, Reg(ndx) happens to be located at index ndx.
        // But it's a detail we try to hide, as it's likely we're going to have
        // to transition to a more complex structure in the future.
        let iv = self.inner.get(reg.0)?;
        debug_assert_eq!(iv.dest.get(), reg);
        Some(iv)
    }

    /// Total number of registers/instructions stored in this Program.
    ///
    /// This may well include dead values. Use [`insns_rpo`] to iterate through
    /// the program only "through dependency edges" and only get live registers.
    pub fn reg_count(&self) -> mil::Index {
        self.inner.len()
    }

    pub fn registers(&self) -> impl '_ + Iterator<Item = mil::Reg> {
        (0..self.reg_count())
            .filter(|&reg_ndx| self.inner.is_enabled(reg_ndx))
            .map(mil::Reg)
    }

    pub fn get_call_args(&self, mut arg: Option<mil::Reg>) -> impl '_ + Iterator<Item = mil::Reg> {
        std::iter::repeat_with(move || {
            let insn = self.get(arg?).unwrap().insn.get();
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
        self.schedule
            .of_block(bid)
            .into_iter()
            .map(|&ndx| mil::Reg(ndx))
    }

    pub fn find_last_matching<P, R>(&self, bid: cfg::BlockID, pred: P) -> Option<R>
    where
        P: Fn(mil::Insn) -> Option<R>,
    {
        self.block_regs(bid)
            .rev()
            .map(|reg| self[reg].get())
            .find_map(pred)
    }

    pub fn push_pure(&mut self, insn: mil::Insn) -> mil::Reg {
        // side-effecting instructions need to be added to a specific position
        // in the basic block.
        // pure instructions, instead, aren't really attached to anywhere in
        // particular.
        assert!(!insn.has_side_effects());
        mil::Reg(self.inner.push_new(insn))
    }

    pub fn upsilons_of_phi(&self, phi_reg: mil::Reg) -> impl '_ + Iterator<Item = mil::Reg> {
        assert!(matches!(self[phi_reg].get(), mil::Insn::Phi));

        self.inner.iter().filter_map(move |iv| match iv.insn.get() {
            mil::Insn::Upsilon { value, phi_ref } if phi_ref == phi_reg => Some(value),
            _ => None,
        })
    }

    pub fn reg_type(&self, reg: mil::Reg) -> mil::RegType {
        use mil::{Insn, RegType};
        match self.inner.get(reg.0).unwrap().insn.get() {
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
            | Insn::Control(_) => RegType::Control,

            Insn::NotYetImplemented(_) => RegType::Unit,
            Insn::LoadMem { size, .. } => RegType::Bytes(size as usize),
            Insn::StoreMem { .. } => RegType::MemoryEffect,
            Insn::OverflowOf(_) => RegType::Bool,
            Insn::CarryOf(_) => RegType::Bool,
            Insn::SignOf(_) => RegType::Bool,
            Insn::IsZero(_) => RegType::Bool,
            Insn::Parity(_) => RegType::Bool,
            Insn::Undefined => RegType::Undefined,
            Insn::Phi => {
                let mut ys = self.upsilons_of_phi(reg);
                let Some(y) = ys.next() else {
                    panic!("no upsilons for this phi? {:?}", reg)
                };
                // assuming that all types are the same, as per assert_phis_consistent
                self.reg_type(y)
            }
            Insn::Ancestral(anc_name) => self
                .inner
                .ancestor_type(anc_name)
                .expect("ancestor has no defined type"),
            Insn::StructGetMember { size, .. } => RegType::Bytes(size as usize),
            Insn::Upsilon { .. } => RegType::Unit,
        }
    }

    pub fn types(&self) -> &ty::TypeSet {
        self.inner.types()
    }

    pub fn assert_invariants(&self) {
        eprintln!("---- checking");
        eprintln!("{:?}", self);
        self.assert_dest_reg_is_index();
        self.assert_no_circular_refs();
        self.assert_inputs_visible();
        self.assert_consistent_phis();
        self.assert_carg_chain();
    }

    fn assert_dest_reg_is_index(&self) {
        // only scheduled instructions are subject to the "law" of SSA
        for bid in self.cfg().block_ids_rpo() {
            for &ndx in self.schedule.of_block(bid) {
                let iv = self.inner.get(ndx).unwrap();
                assert_eq!(ndx, iv.dest.get().reg_index());
            }
        }
    }

    fn assert_inputs_visible(&self) {
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
                    eprintln!("start {:?}", bid);
                    assert!(!block_visited[bid]);
                    block_visited[bid] = true;

                    for &ndx in self.schedule.of_block(bid) {
                        let reg = mil::Reg(ndx);

                        let mut insn = self.get(reg).unwrap().insn.get();
                        eprintln!("   {:?} {:?}", reg, insn);
                        for &mut input in insn.input_regs_iter() {
                            eprintln!("     input {:?}", input);
                            let Some(def_block_input) = def_block[input] else {
                                panic!("input {input:?} of {reg:?} is not defined");
                            };
                            assert!(
                                def_block_input == bid
                                    || dom_tree
                                        .imm_doms(bid)
                                        .find(|&b| b == def_block_input)
                                        .is_some()
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
                    eprintln!("end {:?}", bid);
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

        for iv in self.inner.iter() {
            if rdr_count[iv.dest.get()] == 0 {
                continue;
            }
            if let mil::Insn::Upsilon { value, phi_ref } = iv.insn.get() {
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

            if !self.inner.is_enabled(reg.reg_index()) {
                continue;
            }

            for &mut input in self[reg].get().input_regs_iter() {
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
            let insn = self[reg].get();
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

            let arg_def = self[arg].get();
            assert!(
                matches!(arg_def, mil::Insn::CArg { .. }),
                "reg {:?} does not point to a CArg, but to {:?}",
                reg,
                arg_def
            );
        }
    }
}

impl std::ops::Index<mil::Reg> for Program {
    type Output = Cell<mil::Insn>;
    fn index(&self, reg: mil::Reg) -> &Cell<mil::Insn> {
        self.get(reg).unwrap().insn
    }
}

#[cfg(test)]
#[test]
#[should_panic]
fn test_assert_no_circular_refs() {
    use std::sync::Arc;

    use mil::{ArithOp, Insn, Reg};

    use crate::ty;

    let prog = {
        let mut pb = mil::ProgramBuilder::new(Arc::new(ty::TypeSet::new()));
        pb.set_input_addr(0xf0);
        pb.push(
            Reg(0),
            Insn::Const {
                value: 123,
                size: 8,
            },
        );
        pb.push(Reg(1), Insn::Arith(ArithOp::Add, Reg(0), Reg(2)));
        pb.push(Reg(2), Insn::Arith(ArithOp::Add, Reg(0), Reg(1)));
        pb.build()
    };
    let prog = mil_to_ssa(ConversionParams::new(prog));
    prog.assert_no_circular_refs();
}

impl std::fmt::Debug for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

            let rdr_count = rdr_count[reg];
            if rdr_count > 1 {
                write!(f, "  ({:3})  ", rdr_count)?;
            } else {
                write!(f, "         ")?;
            }

            let iv = self.get(reg).unwrap();

            type_s.clear();
            if let Some(tyid) = iv.tyid.get() {
                let mut pp = pp::PrettyPrinter::start(&mut type_s);
                write!(pp, ": ").unwrap();
                self.inner.types().dump_type_ref(&mut pp, tyid).unwrap();
            }

            let type_s = std::str::from_utf8(&type_s).unwrap();
            writeln!(f, "{:?}{} <- {:?}", reg, type_s, iv.insn.get())?;
        }

        Ok(())
    }
}

// TODO Remove this. No longer needed in the current design. Already cut down to nothing.
pub struct ConversionParams {
    pub program: mil::Program,
}

impl ConversionParams {
    pub fn new(program: mil::Program) -> Self {
        ConversionParams { program }
    }
}

pub fn mil_to_ssa(input: ConversionParams) -> Program {
    let ConversionParams { mut program, .. } = input;

    let var_count = program.reg_count();
    let vars = move || (0..var_count).map(mil::Reg);

    let (cfg, mut schedule) = cfg::analyze_mil(&program);
    assert_eq!(cfg.block_count(), schedule.block_count());

    let dom_tree = cfg.dom_tree();
    let is_phi_needed = compute_phis_set(&program, &cfg, &schedule, dom_tree);

    program.set_enabled_mask(vec![true; program.len() as usize]);

    // create all required phi nodes, and link them to basic blocks and
    // variables, to be "wired" later to the data flow graph
    let phis = {
        let mut phis = RegMat::for_program(&program, &cfg, None);
        for bid in cfg.block_ids() {
            for var in vars() {
                if *is_phi_needed.get(bid, var) {
                    // just set `var` as the dest; the regular renaming process
                    // will process this insn just like the others
                    let ndx = program.push(var, mil::Insn::Phi);
                    // in `phis`, we need a stable ID for this insn (which
                    // survives the renaming)
                    phis.set(bid, var, Some(ndx));
                    schedule.insert(ndx, bid, 0);
                }
            }
        }
        phis
    };

    /*
    stack[v] is a stack of variable names (for every variable v)

    def rename(block):
      for instr in block:
        replace each argument to instr with stack[old name]

        replace instr's destination with a new name
        push that new name onto stack[old name]

      for s in block's successors:
        for p in s's ϕ-nodes:
          Assuming p is for a variable v, make it read from stack[v].

      for b in blocks immediately dominated by block:
        # That is, children in the dominance tree.
        rename(b)

      pop all the names we just pushed onto the stacks

    rename(entry)
    */

    // TODO rework this with a more efficient data structure
    struct VarMap {
        count: mil::Index,
        stack: Vec<Box<[Option<mil::Reg>]>>,
    }

    impl VarMap {
        fn new(count: mil::Index) -> Self {
            VarMap {
                count,
                stack: Vec::new(),
            }
        }

        fn push(&mut self) {
            if self.stack.is_empty() {
                self.stack
                    .push(vec![None; self.count.into()].into_boxed_slice());
            } else {
                let new_elem = self.stack.last().unwrap().clone();
                self.stack.push(new_elem);
            }

            self.check_invariants()
        }

        fn check_invariants(&self) {
            for elem in &self.stack {
                assert_eq!(elem.len(), self.count.into());
            }
        }

        fn pop(&mut self) {
            self.stack.pop();
            self.check_invariants();
        }

        fn current(&self) -> &[Option<mil::Reg>] {
            &self.stack.last().expect("no mappings!")[..]
        }
        fn current_mut(&mut self) -> &mut [Option<mil::Reg>] {
            &mut self.stack.last_mut().expect("no mappings!")[..]
        }
        fn get(&self, reg: mil::Reg) -> Option<mil::Reg> {
            let reg_num = reg.reg_index() as usize;
            self.current()[reg_num]
        }

        fn set(&mut self, old: mil::Reg, new: mil::Reg) {
            let reg_num = old.reg_index() as usize;
            self.current_mut()[reg_num] = Some(new);
        }
    }

    let mut var_map = VarMap::new(var_count);

    enum Cmd {
        Finish,
        Start(cfg::BlockID),
    }
    let mut queue = vec![Cmd::Finish, Cmd::Start(cfg.entry_block_id())];

    while let Some(cmd) = queue.pop() {
        match cmd {
            Cmd::Start(bid) => {
                var_map.push();

                // -- patch current block
                //
                // insns are treated as if already renamed so that their dest
                // becomes mil::Reg(index) (this replacement will physically
                // happen later on).
                for &insn_ndx in schedule.of_block(bid) {
                    let iv = program.get(insn_ndx).unwrap();

                    let mut insn = iv.insn.get();
                    for reg in insn.input_regs() {
                        *reg = var_map.get(*reg).expect("value not initialized in pre-ssa");
                    }
                    iv.insn.set(insn);

                    // in the output SSA, each destination register corresponds to the instruction's
                    // index. this way, the numeric value of a register can also be used as
                    // instruction ID, to locate a register/variable's defining instruction.
                    let new_dest = mil::Reg(insn_ndx);
                    // iv.dest is rewritten later (always mil::Reg(index))
                    // we only update `var_map`
                    let old_name = iv.dest.get();

                    let new_name = if let mil::Insn::Get(input_reg) = iv.insn.get() {
                        // exception: for Get(_) instructions, we just reuse the input reg for the
                        // output
                        input_reg
                    } else {
                        new_dest
                    };
                    var_map.set(old_name, new_name);
                }

                // -- patch successor's phi nodes
                // The algorithm is the following:
                // >  for s in block's successors:
                // >    for p in s's ϕ-nodes:
                // >      Assuming p is for a variable v, make it read from stack[v].
                //
                // Thanks to the Upsilon formulation of phi nodes, it's
                // sufficient to add one such insn to obtain the final correct
                // representation. Algorithms that want to know the operands of
                // the phi node will iterate in reverse through the
                // predecessor's instructions.

                for succ in cfg.block_cont(bid).block_dests() {
                    for var in vars() {
                        if let &Some(phi_ndx) = phis.get(succ, var) {
                            let value = var_map
                                .get(var)
                                .expect("value not initialized in pre-ssa (phi)");
                            let ups = program.push_new(mil::Insn::Upsilon {
                                value,
                                phi_ref: mil::Reg(phi_ndx),
                            });
                            schedule.append(ups, bid);
                        }
                    }
                }

                // TODO quadratic, but cfgs are generally pretty small
                let imm_dominated = dom_tree
                    .items()
                    .filter(|(_, parent)| **parent == Some(bid))
                    .map(|(bid, _)| bid);
                for child in imm_dominated {
                    queue.push(Cmd::Finish);
                    queue.push(Cmd::Start(child));
                }
            }

            Cmd::Finish => {
                var_map.pop();
            }
        }
    }

    // replace all dest with mil::Reg(index), regardless of whether or not the
    // insn is scheduled. not strictly required, but simplifies a bunch of other
    // algorithms later on
    for (ndx, iv) in program.iter().enumerate() {
        iv.dest.set(mil::Reg(ndx.try_into().unwrap()));
    }

    let ssa = Program {
        inner: program,
        schedule,
        cfg,
    };
    ssa.assert_invariants();
    ssa
}

fn compute_phis_set(
    program: &mil::Program,
    cfg: &cfg::Graph,
    block_spans: &cfg::Schedule,
    dom_tree: &cfg::DomTree,
) -> RegMat<bool> {
    // matrix [B * B] where B = #blocks
    // matrix[i, j] = true iff block j is in block i's dominance frontier
    let is_dom_front = compute_dominance_frontier(cfg, dom_tree);
    let block_uses_var = find_received_vars(program, block_spans, cfg);
    let mut phis_set = RegMat::for_program(program, cfg, false);

    // the rule is:
    //   if variable v is written in block b,
    //   then we have to add `v <- phi v, v, ...` on each block in b's dominance frontier
    // (phis_set is to avoid having multiple phis for the same var)
    for bid in cfg.block_ids() {
        for &ndx in block_spans.of_block(bid) {
            let dest = program.get(ndx).unwrap().dest.get();
            for target_bid in cfg.block_ids() {
                if cfg.block_preds(target_bid).len() < 2 {
                    continue;
                }

                let is_blk_in_df = *is_dom_front.item(bid.as_usize(), target_bid.as_usize());
                let is_used = *block_uses_var.get(target_bid, dest);

                if is_blk_in_df && is_used {
                    phis_set.set(target_bid, dest, true);
                }
            }
        }
    }

    phis_set
}

/// Find the set of "received variables" for each block.
///
/// This is the set of variables which are read before any write in the block.
/// In other words, for these are the variables, the block observes the values
/// left there by other blocks.
fn find_received_vars(
    prog: &mil::Program,
    block_spans: &cfg::Schedule,
    graph: &cfg::Graph,
) -> RegMat<bool> {
    let mut is_received = RegMat::for_program(prog, graph, false);
    for bid in graph.block_ids_postorder() {
        for &ndx in block_spans.of_block(bid).into_iter().rev() {
            let iv = prog.get(ndx).unwrap();
            let dest = iv.dest.get();
            let mut insn = iv.insn.get();

            is_received.set(bid, dest, false);
            for &mut input in insn.input_regs_iter() {
                is_received.set(bid, input, true);
            }
        }
    }
    is_received
}

#[derive(Debug)]
struct RegMat<T>(Mat<T>);

impl<T: Clone> RegMat<T> {
    fn for_program(program: &mil::Program, graph: &cfg::Graph, value: T) -> Self {
        let var_count = program.reg_count() as usize;
        RegMat(Mat::new(value, graph.block_count() as usize, var_count))
    }

    fn get(&self, bid: cfg::BlockID, reg: mil::Reg) -> &T {
        self.0
            .item(bid.as_number() as usize, reg.reg_index() as usize)
    }

    fn set(&mut self, bid: cfg::BlockID, reg: mil::Reg, value: T) {
        *self
            .0
            .item_mut(bid.as_number() as usize, reg.reg_index() as usize) = value;
    }
}

struct Mat<T> {
    items: Box<[T]>,
    rows: usize,
    cols: usize,
}
impl<T> Mat<T> {
    fn ndx(&self, i: usize, j: usize) -> usize {
        assert!(i < self.rows);
        assert!(j < self.cols);
        self.cols * i + j
    }

    fn item(&self, i: usize, j: usize) -> &T {
        &self.items[self.ndx(i, j)]
    }
    fn item_mut(&mut self, i: usize, j: usize) -> &mut T {
        &mut self.items[self.ndx(i, j)]
    }
}
impl<T: Clone> Mat<T> {
    fn new(init: T, rows: usize, cols: usize) -> Self {
        let items = vec![init; rows * cols].into_boxed_slice();
        Mat { items, rows, cols }
    }
}
impl<T: std::fmt::Debug> std::fmt::Debug for Mat<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut item_buf = Vec::new();

        writeln!(f, "mat. {}×{}", self.rows, self.cols)?;
        for i in 0..self.rows {
            for j in 0..self.cols {
                let value = self.item(i, j);
                item_buf.clear();
                write!(item_buf, "{:?}", value).unwrap();
                write!(f, "{:10} ", std::str::from_utf8(&item_buf).unwrap())?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

fn compute_dominance_frontier(graph: &cfg::Graph, dom_tree: &cfg::DomTree) -> Mat<bool> {
    let count = graph.block_count() as usize;
    let mut mat = Mat::new(false, count, count);

    for bid in graph.block_ids() {
        let preds = graph.block_preds(bid);
        if preds.len() < 2 {
            continue;
        }

        let runner_stop = dom_tree[bid].unwrap();
        for &pred in preds {
            // bid is in the dominance frontier of pred, and of all its dominators
            let mut runner = pred;
            while runner != runner_stop {
                *mat.item_mut(runner.as_usize(), bid.as_usize()) = true;
                runner = dom_tree[runner].unwrap();
            }
        }
    }

    mat
}

#[derive(Clone, Debug)]
pub struct RegMap<T>(Vec<T>);

impl<T: Clone> RegMap<T> {
    pub fn for_program(prog: &Program, init: T) -> Self {
        let inner = vec![init; prog.reg_count() as usize];
        RegMap(inner)
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
        for &mut input in prog[reg].get().input_regs_iter() {
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
/// ruled out by calling [Program::assert_invariants].
pub fn count_readers(prog: &Program) -> RegMap<usize> {
    let mut count = vec![0; prog.reg_count() as usize];

    for (_, reg) in prog.insns_rpo() {
        for reg in prog[reg].get().input_regs() {
            count[reg.reg_index() as usize] += 1;
        }
    }

    RegMap(count)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{mil, ty};
    use mil::{ArithOp, Control, Insn, Reg};

    #[test]
    fn test_phi_read() {
        let prog = {
            let mut pb = mil::ProgramBuilder::new(Arc::new(ty::TypeSet::new()));

            pb.set_input_addr(0xf0);
            pb.push(
                Reg(0),
                Insn::Const {
                    value: 123,
                    size: 8,
                },
            );
            pb.push(Reg(1), Insn::SetJumpCondition(Reg(0)));
            pb.push(Reg(1), Insn::Control(Control::JmpExtIf(0xf2)));

            pb.set_input_addr(0xf1);
            pb.push(Reg(2), Insn::Const { value: 4, size: 1 });
            pb.push(Reg(1), Insn::Control(Control::JmpExt(0xf3)));

            pb.set_input_addr(0xf2);
            pb.push(Reg(2), Insn::Const { value: 8, size: 1 });

            pb.set_input_addr(0xf3);
            pb.push(Reg(4), Insn::ArithK(ArithOp::Add, Reg(2), 456));
            pb.push(Reg(5), Insn::SetReturnValue(Reg(4)));
            pb.push(Reg(5), Insn::Control(Control::Ret));

            pb.build()
        };

        eprintln!("-- mil:");
        eprintln!("{:?}", prog);

        eprintln!("-- ssa:");
        let prog = super::mil_to_ssa(super::ConversionParams::new(prog));
        insta::assert_debug_snapshot!(prog);
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
        prog.get(Reg(0))
            .unwrap()
            .insn
            .set(Insn::Arith(mil::ArithOp::Add, Reg(1), Reg(2)));
        prog.assert_no_circular_refs();
    }

    fn make_prog_no_cycles() -> super::Program {
        let prog = {
            let mut b = mil::ProgramBuilder::new(Arc::new(ty::TypeSet::new()));
            b.push(Reg(0), Insn::Const { value: 5, size: 8 });
            b.push(Reg(1), Insn::Const { value: 5, size: 8 });
            b.push(Reg(0), Insn::Arith(mil::ArithOp::Add, Reg(1), Reg(0)));
            b.push(Reg(0), Insn::SetReturnValue(Reg(0)));
            b.push(Reg(0), Insn::Control(Control::Ret));
            b.build()
        };
        let prog = super::mil_to_ssa(super::ConversionParams::new(prog));
        // cycle absent here: SSA conversion would have failed by now
        prog
    }
}
