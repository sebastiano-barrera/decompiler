use std::collections::HashSet;

/// Static Single-Assignment representation of a program (and conversion from direct multiple
/// assignment).
///
/// The algorithms in this module are mostly derived from the descriptions in:
/// > Cooper, Keith & Harvey, Timothy & Kennedy, Ken. (2006).
/// > A Simple, Fast Dominance Algorithm.
/// > Rice University, CS Technical Report 06-33870.
use crate::{cfg, mil};

#[derive(Clone)]
pub struct Program {
    // an ssa::Program contains a mil::Program at its core, but never exposes it directly:
    //
    // - extra instructions are appended for various uses, although they do not belong
    //   (directly) to the program sequence (they're only referred-to by other in-sequence
    //   instruction)
    //
    // - registers are numerically equal to the index of the defining instruction.
    //   mil::Index values are just as good as mil::Reg for identifying both insns and
    //   values.
    inner: mil::Program,
    cfg: cfg::Graph,
    bbs: cfg::BlockMap<BasicBlock>,
}

slotmap::new_key_type! { pub struct TypeID; }

#[derive(Clone, Default)]
pub struct BasicBlock {
    effects: Vec<mil::Reg>,
}

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

    pub fn reg_count(&self) -> mil::Index {
        self.inner.len()
    }

    // TODO: this can yield dead instruction in some situations
    pub fn live_regs(&self) -> impl '_ + Iterator<Item = mil::Reg> {
        (0..self.reg_count()).map(mil::Reg)
    }

    /// Iterate through the instructions in the program, in no particular order
    // TODO: this can yield dead instruction in some situations
    pub fn insns_unordered(&self) -> impl Iterator<Item = mil::InsnView> {
        self.inner.iter()
    }

    pub fn get_call_args(&self, reg: mil::Reg) -> impl '_ + Iterator<Item = mil::Reg> {
        self.inner.get_call_args(reg.0)
    }

    pub fn insns_rpo(&self) -> InsnRPOIter {
        InsnRPOIter::new(self)
    }

    pub fn block_effects(&self, bid: cfg::BlockID) -> impl '_ + Iterator<Item = mil::Reg> {
        self.bbs[bid].effects.iter().copied()
    }

    pub fn value_type(&self, reg: mil::Reg) -> mil::RegType {
        use mil::{Insn, RegType};
        match self.inner.get(reg.0).unwrap().insn.get() {
            Insn::True => RegType::Bool,
            Insn::False => RegType::Bool,
            Insn::Const { size, .. } => RegType::Bytes(size as usize),
            Insn::Part { size, .. } => RegType::Bytes(size as usize),
            Insn::Get(arg) => self.value_type(arg),
            Insn::Concat { lo, hi } => {
                let lo_size = self.value_type(lo).bytes_size().unwrap();
                let hi_size = self.value_type(hi).bytes_size().unwrap();
                RegType::Bytes(lo_size + hi_size)
            }
            Insn::Widen {
                reg: _,
                target_size,
            } => RegType::Bytes(target_size as usize),
            Insn::Arith(_, a, b) => {
                let at = self.value_type(a);
                let bt = self.value_type(b);
                assert_eq!(at, bt); // TODO check this some better way
                at
            }
            Insn::ArithK(_, a, _) => self.value_type(a),
            Insn::Cmp(_, _, _) => RegType::Bool,
            Insn::Bool(_, _, _) => RegType::Bool,
            Insn::Not(_) => RegType::Bool,
            // TODO This might have to change based on the use of calling
            // convention and function type info
            Insn::Call(_) => RegType::Bytes(8),
            Insn::CArg(_) => RegType::Effect,
            Insn::Ret(_) => RegType::Effect,
            Insn::JmpInd(_) => RegType::Effect,
            Insn::Jmp(_) => RegType::Effect,
            Insn::JmpExt(_) => RegType::Effect,
            Insn::JmpIf { .. } => RegType::Effect,
            Insn::JmpExtIf { .. } => RegType::Effect,
            Insn::TODO(_) => RegType::Effect,
            Insn::LoadMem { size, .. } => RegType::Bytes(size as usize),
            Insn::StoreMem(_, _) => RegType::Effect,
            Insn::OverflowOf(_) => RegType::Effect,
            Insn::CarryOf(_) => RegType::Effect,
            Insn::SignOf(_) => RegType::Effect,
            Insn::IsZero(_) => RegType::Effect,
            Insn::Parity(_) => RegType::Effect,
            Insn::Undefined => RegType::Effect,
            Insn::Phi => {
                // TODO find all corresponding Upsilons, check that their size
                // all match, then return the size
                todo!("size of phi")
            }
            Insn::Ancestral(anc_name) => self
                .inner
                .ancestor_type(anc_name)
                .expect("ancestor has no defined type"),
            Insn::StructGet8 { .. } => RegType::Bytes(8),
            Insn::Upsilon { .. } => RegType::Effect,
        }
    }

    pub fn assert_invariants(&self) {
        self.assert_no_circular_refs();
        // self.assert_no_empty_blocks();
    }

    fn assert_no_circular_refs(&self) {
        let mut defined = HashSet::new();
        for (_, reg) in self.insns_rpo() {
            let iv = self.get(reg).unwrap();
            let insn = iv.insn.get();
            let dest = iv.dest.get();
            insn.input_regs_iter().for_each(|reg| {
                assert!(
                    defined.contains(&reg),
                    "{:?}: undefined reg: {:?}",
                    dest,
                    reg
                )
            });

            let is_first_def = defined.insert(dest);
            assert!(is_first_def);
        }
    }
}

/// An iterator that walks through the instructions of a whole program in
/// reverse post order.
///
/// Reverse postorder means that an instruction is always returned after all of
/// its dependencies (both control and data).
///
/// The item is (BlockID, mil::Reg), so the block ID is explicitly given.
pub struct InsnRPOIter<'a> {
    queue: Vec<IterCmd>,
    prog: &'a Program,
    was_yielded: Vec<bool>,
}
enum IterCmd {
    Block(cfg::BlockID),
    StartInsn((cfg::BlockID, mil::Reg)),
    EndInsn((cfg::BlockID, mil::Reg)),
}
impl<'a> InsnRPOIter<'a> {
    fn new(prog: &'a Program) -> Self {
        let queue = prog.cfg.block_ids_rpo().rev().map(IterCmd::Block).collect();
        InsnRPOIter {
            queue,
            prog,
            was_yielded: vec![false; prog.reg_count() as usize],
        }
    }
}
impl<'a> Iterator for InsnRPOIter<'a> {
    type Item = (cfg::BlockID, mil::Reg);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let Some(cmd) = self.queue.pop() else {
                return None;
            };

            match cmd {
                IterCmd::Block(bid) => {
                    self.queue.extend(
                        self.prog.bbs[bid]
                            .effects
                            .iter()
                            .rev()
                            .copied()
                            .map(|reg| IterCmd::StartInsn((bid, reg))),
                    );
                }
                IterCmd::StartInsn((bid, reg)) => {
                    self.queue.push(IterCmd::EndInsn((bid, reg)));
                    self.queue.extend(
                        self.prog
                            .get(reg)
                            .unwrap()
                            .insn
                            .get()
                            .input_regs_iter()
                            .map(|input| IterCmd::StartInsn((bid, input))),
                    );
                }
                IterCmd::EndInsn((bid, reg)) => {
                    let was_yielded =
                        std::mem::replace(&mut self.was_yielded[reg.reg_index() as usize], true);
                    if !was_yielded {
                        return Some((bid, reg));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[test]
#[should_panic]
fn test_assert_no_circular_refs() {
    use mil::{ArithOp, Insn, Reg};

    let prog = {
        let mut pb = mil::ProgramBuilder::new();
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

        writeln!(f, "ssa program  {} instrs", self.reg_count())?;

        let mut cur_bid = None;

        for (bid, reg) in self.insns_rpo() {
            if cur_bid != Some(bid) {
                write!(f, ".B{}:    ;; ", bid.as_usize())?;
                let preds = self.cfg.block_preds(bid);
                if preds.len() > 0 {
                    write!(f, "preds:")?;
                    for (ndx, pred) in preds.iter().enumerate() {
                        if ndx > 0 {
                            write!(f, ",")?;
                        }
                        write!(f, "B{}", pred.as_number())?;
                    }
                }
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
            writeln!(f, "{:?} <- {:?}", reg, iv.insn.get())?;
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

    let cfg = cfg::analyze_mil(&program);
    let dom_tree = cfg.dom_tree();
    let is_phi_needed = compute_phis_set(&program, &cfg, dom_tree);

    // create all required phi nodes, and link them to basic blocks and
    // variables, to be "wired" later to the data flow graph
    let phis = {
        let mut phis = RegMat::for_program(&program, &cfg, None);
        for bid in cfg.block_ids() {
            for var in vars() {
                if *is_phi_needed.get(bid, var) {
                    let ndx = program.push_new(mil::Insn::Phi);
                    let phi = mil::Reg(ndx);
                    phis.set(bid, var, Some(phi));
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

    let mut bbs = cfg::BlockMap::new(BasicBlock::default(), cfg.block_count());

    while let Some(cmd) = queue.pop() {
        match cmd {
            Cmd::Start(bid) => {
                var_map.push();

                let bb = &mut bbs[bid];

                // -- patch current block

                // phi nodes have already been added; we only "take in" the ones
                // belonging to this block (associate them to the mil variable
                // they represent)
                //
                // we're doing the "pizlo special" [^1], so arguments are going
                // to be "set" via Upsilon instructions
                //
                // [^1] https://gist.github.com/pizlonator/cf1e72b8600b1437dda8153ea3fdb963
                for var in vars() {
                    // var is still pre-SSA/renaming
                    if let Some(phi_reg) = phis.get(bid, var) {
                        var_map.set(var, *phi_reg);
                    }
                }

                for insn_ndx in cfg.insns_ndx_range(bid) {
                    let iv = program.get(insn_ndx).unwrap();

                    // TODO Re-establish this invariant
                    //  this will only make sense after I add abstract values representing initial
                    //  conditions to each function... (probably coming from the calling
                    //  convention)
                    //
                    // > if bid == cfg::ENTRY_BID && insn_ndx == 0 {
                    // >     assert_eq!(inputs, [None, None]);
                    // > }

                    let mut insn = iv.insn.get();
                    for reg in insn.input_regs_mut().into_iter().flatten() {
                        *reg = var_map.get(*reg).expect("value not initialized in pre-ssa");
                    }
                    iv.insn.set(insn);

                    // in the output SSA, each destination register corresponds to the instruction's
                    // index. this way, the numeric value of a register can also be used as
                    // instruction ID, to locate a register/variable's defining instruction.
                    let new_dest = mil::Reg(insn_ndx);
                    let old_name = iv.dest.replace(new_dest);

                    let new_name = if let mil::Insn::Get(input_reg) = iv.insn.get() {
                        // exception: for Get(_) instructions, we just reuse the input reg for the
                        // output
                        input_reg
                    } else {
                        new_dest
                    };
                    var_map.set(old_name, new_name);

                    if insn.has_side_effects() {
                        bb.effects.push(new_name);
                    }
                }

                // -- patch successor's phi nodes
                // The algorithm is the following:
                // >  for s in block's successors:
                // >    for p in s's ϕ-nodes:
                // >      Assuming p is for a variable v, make it read from stack[v].
                //
                // ... in order to patch the correct operand in each phi instruction, we have
                // to use its "predecessor position", i.e. whether the current block is the
                // successor's 1st, 2nd, 3rd predecessor.

                for (_my_pred_ndx, succ) in cfg.block_cont(bid).as_array().into_iter().flatten() {
                    for var in vars() {
                        if let Some(phi_reg) = phis.get(succ, var) {
                            let value = var_map
                                .get(var)
                                .expect("value not initialized in pre-ssa (phi)");
                            let ups = program.push_new(mil::Insn::Upsilon {
                                value,
                                phi_ref: *phi_reg,
                            });
                            bb.effects.push(mil::Reg(ups));
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

    // establish SSA invariants
    // the returned Program will no longer change (just some instructions are going to be marked as
    // "dead" and ignored)
    for (ndx, insn) in program.iter().enumerate() {
        let ndx = ndx.try_into().unwrap();
        assert_eq!(
            insn.dest.get(),
            mil::Reg(ndx),
            "insn unvisited: {:?} <- {:?}",
            insn.dest,
            insn.insn
        );
    }

    let mut ssa = Program {
        inner: program,
        cfg,
        bbs,
    };
    eliminate_dead_code(&mut ssa);
    ssa.assert_invariants();
    ssa
}

fn compute_phis_set(
    program: &mil::Program,
    cfg: &cfg::Graph,
    dom_tree: &cfg::DomTree,
) -> RegMat<bool> {
    // matrix [B * B] where B = #blocks
    // matrix[i, j] = true iff block j is in block i's dominance frontier
    let is_dom_front = compute_dominance_frontier(cfg, dom_tree);
    let block_uses_var = find_received_vars(program, cfg);
    let mut phis_set = RegMat::for_program(program, cfg, false);

    // the rule is:
    //   if variable v is written in block b,
    //   then we have to add `v <- phi v, v, ...` on each block in b's dominance frontier
    // (phis_set is to avoid having multiple phis for the same var)
    for bid in cfg.block_ids() {
        let slice = program.slice(cfg.insns_ndx_range(bid)).unwrap();
        for dest in slice.dests.iter() {
            let dest = dest.get();
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
fn find_received_vars(prog: &mil::Program, graph: &cfg::Graph) -> RegMat<bool> {
    let mut is_received = RegMat::for_program(prog, graph, false);
    for bid in graph.block_ids_postorder() {
        for (dest, insn) in prog
            .slice(graph.insns_ndx_range(bid))
            .unwrap()
            .iter_copied()
            .rev()
        {
            is_received.set(bid, dest, false);
            for input in insn.input_regs_iter() {
                is_received.set(bid, input, true);
            }
        }
    }
    is_received
}

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

pub fn eliminate_dead_code(prog: &mut Program) {
    // TODO just delete this function
}

#[derive(Clone, Debug)]
pub struct RegMap<T>(Vec<T>);

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

pub fn count_readers(prog: &Program) -> RegMap<usize> {
    let mut count = vec![0; prog.reg_count() as usize];

    for (_, reg) in prog.insns_rpo() {
        for reg in prog.get(reg).unwrap().insn.get().input_regs_iter() {
            count[reg.0 as usize] += 1;
        }
    }

    RegMap(count)
}

#[cfg(test)]
mod tests {
    use crate::mil;

    #[test]
    fn test_phi_read() {
        use mil::{ArithOp, Insn, Reg};

        let prog = {
            let mut pb = mil::ProgramBuilder::new();

            pb.set_input_addr(0xf0);
            pb.push(
                Reg(0),
                Insn::Const {
                    value: 123,
                    size: 8,
                },
            );
            pb.push(
                Reg(1),
                Insn::JmpExtIf {
                    cond: Reg(0),
                    addr: 0xf2,
                },
            );

            pb.set_input_addr(0xf1);
            pb.push(Reg(2), Insn::Const { value: 4, size: 1 });
            pb.push(Reg(3), Insn::JmpExt(0xf3));

            pb.set_input_addr(0xf2);
            pb.push(Reg(2), Insn::Const { value: 8, size: 1 });

            pb.set_input_addr(0xf3);
            pb.push(Reg(4), Insn::ArithK(ArithOp::Add, Reg(2), 456));
            pb.push(Reg(5), Insn::Ret(Reg(4)));

            pb.build()
        };

        eprintln!("-- mil:");
        eprintln!("{:?}", prog);

        eprintln!("-- ssa:");
        let prog = super::mil_to_ssa(super::ConversionParams::new(prog));
        insta::assert_debug_snapshot!(prog);
    }
}
