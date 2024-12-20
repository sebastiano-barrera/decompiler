use std::ops::Range;

/// Static Single-Assignment representation of a program (and conversion from direct multiple
/// assignment).
///
/// The algorithms in this module are mostly derived from the descriptions in:
/// > Cooper, Keith & Harvey, Timothy & Kennedy, Ken. (2006).
/// > A Simple, Fast Dominance Algorithm.
/// > Rice University, CS Technical Report 06-33870.
use crate::{
    cfg::{self, BlockID},
    mil,
};

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
    phis: cfg::BlockMap<PhiInfo>,
    cfg: cfg::Graph,

    is_alive: Vec<bool>,
    rdr_count: ReaderCount,
}

impl Program {
    pub fn cfg(&self) -> &cfg::Graph {
        &self.cfg
    }

    pub fn get(&self, ndx: mil::Index) -> Option<mil::InsnView> {
        self.inner.get(ndx)
    }
    pub fn len(&self) -> mil::Index {
        self.inner.len()
    }

    pub fn readers_count(&self, reg: mil::Reg) -> usize {
        self.rdr_count.get(reg)
    }

    pub fn block_phi(&self, bid: cfg::BlockID) -> &PhiInfo {
        &self.phis[bid]
    }

    pub fn is_alive(&self, ndx: mil::Index) -> bool {
        self.is_alive[ndx as usize]
    }
}

// private utility functions
impl Program {
    fn check_invariants(&self) {
        let len = self.inner.len() as usize;
        assert_eq!(len, self.is_alive.len());
        assert_eq!(len, self.rdr_count.0.len());
        // TODO more?
    }

    pub fn edit(&mut self) -> EditableProgram {
        EditableProgram(self)
    }

    /// Push a new instruction to the program.
    ///
    /// The instruction will be placed in the "trail" area of the program,
    /// outside of the executable sequence.  In this area, an instruction is
    /// only useful for being referred-to by other instructions or an
    /// equivalence set.
    fn push(&mut self, insn: mil::Insn) -> mil::Index {
        let dest = self.inner.len();
        self.inner.push(mil::Reg(dest), insn);
        self.is_alive.push(true);
        self.rdr_count.0.push(0);
        self.check_invariants();
        dest
    }
}

#[derive(Clone)]
pub struct PhiInfo {
    ndxs: Range<mil::Index>,
    // TODO smaller sizes?; these rarely go above, like 4-8
    phi_count: mil::Index,
    pred_count: mil::Index,
}

impl PhiInfo {
    fn empty() -> Self {
        PhiInfo {
            ndxs: 0..0,
            phi_count: 0,
            pred_count: 0,
        }
    }

    pub fn phi_count(&self) -> mil::Index {
        self.phi_count
    }

    pub fn pred_count(&self) -> mil::Index {
        self.pred_count
    }

    pub fn node_ndx(&self, phi_ndx: mil::Index) -> mil::Index {
        assert!(phi_ndx < self.phi_count);
        assert_eq!(
            self.ndxs.len(),
            (self.phi_count * (1 + self.pred_count)).into()
        );
        self.ndxs.start + phi_ndx * (1 + self.pred_count)
    }

    pub fn arg<'p>(
        &self,
        ssa: &'p Program,
        phi_ndx: mil::Index,
        pred_ndx: mil::Index,
    ) -> &'p mil::Reg {
        let item = ssa.get(self.arg_ndx(phi_ndx, pred_ndx)).unwrap();
        match item.insn {
            mil::Insn::PhiArg(reg) => reg,
            other => panic!("expected phiarg, got {:?}", other),
        }
    }

    fn arg_ndx(&self, phi_ndx: mil::Index, pred_ndx: mil::Index) -> mil::Index {
        assert!(pred_ndx < self.pred_count);
        self.node_ndx(phi_ndx) + 1 + pred_ndx
    }
}

impl std::fmt::Debug for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count: usize = self.is_alive.iter().map(|x| *x as usize).sum();
        writeln!(f, "ssa program  {} instrs", count)?;

        for bid in self.cfg.block_ids() {
            let phis = &self.phis[bid];
            let phi_ndxs = phis.ndxs.clone();
            let nor_ndxs = self.cfg.insns_ndx_range(bid);
            let block_addr = self.inner.get(nor_ndxs.start).unwrap().addr;

            write!(f, ".B{}:  in[", bid.as_usize())?;
            for pred in self.cfg.block_preds(bid) {
                write!(f, ".B{} ", pred.as_number())?;
            }
            write!(f, "]  ")?;
            writeln!(
                f,
                "   ;; 0x{:x}  {} insn {} phis",
                block_addr,
                nor_ndxs.len(),
                phis.phi_count,
            )?;

            for ndx in phi_ndxs.chain(nor_ndxs) {
                let is_alive = self.is_alive[ndx as usize];
                if !is_alive {
                    continue;
                }

                let item = self.inner.get(ndx).unwrap();
                let rdr_count = self.rdr_count.get(item.dest);
                if rdr_count > 1 {
                    write!(f, "  ({:3})  ", rdr_count)?;
                } else {
                    write!(f, "         ")?;
                }
                writeln!(f, "{:?} <- {:?}", item.dest, item.insn)?;
            }
        }

        Ok(())
    }
}

pub fn mil_to_ssa(mut program: mil::Program) -> Program {
    let cfg = cfg::analyze_mil(&program);
    let dom_tree = cfg.dom_tree();
    eprintln!("//  --- dom tree ---");
    cfg.dump_graphviz(Some(&dom_tree));
    eprintln!("//  --- END ---");

    let mut phis = place_phi_nodes(&mut program, &cfg, &dom_tree);

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

    let var_count = count_variables(&program);
    let mut var_map = VarMap::new(var_count);

    enum Cmd {
        Finish,
        Start(BlockID),
    }
    let mut queue = vec![Cmd::Finish, Cmd::Start(cfg::ENTRY_BID)];

    while let Some(cmd) = queue.pop() {
        match cmd {
            Cmd::Start(bid) => {
                var_map.push();

                // -- patch current block

                // phi nodes are *used* here for their destination; their arguments are fixed up
                // while processing the predecessors
                let block_phis = &phis[bid];
                for ndx in block_phis.ndxs.clone() {
                    let item = program.get_mut(ndx).unwrap();
                    match item.insn {
                        mil::Insn::Phi { .. } => {
                            var_map.set(*item.dest, mil::Reg(ndx));
                            *item.dest = mil::Reg(ndx);
                        }
                        mil::Insn::PhiArg { .. } => {}
                        _ => panic!("non-phi node in block phi ndx range"),
                    }
                }

                for insn_ndx in cfg.insns_ndx_range(bid) {
                    let insn = program.get_mut(insn_ndx).unwrap();
                    let inputs = insn.insn.input_regs_mut();

                    // TODO Re-establish this invariant
                    //  this will only make sense after I add abstract values representing initial
                    //  conditions to each function... (probably coming from the calling
                    //  convention)
                    //
                    // > if bid == cfg::ENTRY_BID && insn_ndx == 0 {
                    // >     assert_eq!(inputs, [None, None]);
                    // > }

                    for reg in inputs.into_iter().flatten() {
                        *reg = var_map.get(*reg).expect("value not initialized in pre-ssa");
                    }

                    // in the output SSA, each destination register corrsponds to the instruction's
                    // index. this way, the numeric value of a register can also be used as
                    // instruction ID, to locate a register/variable's defining instruction.
                    let new_dest = mil::Reg(insn_ndx);
                    let old_name = std::mem::replace(insn.dest, new_dest);
                    let new_name = if let mil::Insn::Get(input_reg) = insn.insn {
                        // exception: for Get(_) instructions, we just reuse the input reg for the
                        // output
                        *input_reg
                    } else {
                        new_dest
                    };

                    var_map.set(old_name, new_name);
                }
                for insn_ndx in cfg.insns_ndx_range(bid) {
                    let item = program.get(insn_ndx).unwrap();
                    assert_eq!(item.dest, mil::Reg(insn_ndx));
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

                for (my_pred_ndx, succ) in cfg.block_cont(bid).as_array().into_iter().flatten() {
                    let succ_phis = &mut phis[succ];

                    for phi_ndx in 0..succ_phis.phi_count {
                        let ndx = succ_phis.arg_ndx(phi_ndx, my_pred_ndx.into());
                        let arg = match program.get_mut(ndx).unwrap().insn {
                            mil::Insn::Phi => continue,
                            mil::Insn::PhiArg(value) => value,
                            _ => panic!("non-phi node in block phi ndx range"),
                        };

                        // NOTE: the substitution of the *successor's* phi node's argument is
                        // done in the context of *this* node (its predecessor)
                        *arg = var_map.get(*arg).unwrap_or_else(|| {
                            panic!(
                                "value {:?} not initialized in pre-ssa (phi {:?}--[{}]-->{:?})",
                                *arg, bid, my_pred_ndx, succ
                            )
                        });
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
    let var_count = program.len();
    let is_alive = vec![true; var_count as usize];
    let rdr_count = ReaderCount::new(var_count);

    // establish SSA invariants
    // the returned Program will no longer change (just some instructions are going to be marked as
    // "dead" and ignored)
    for (ndx, insn) in program.iter().enumerate() {
        let ndx = ndx.try_into().unwrap();
        assert_eq!(
            insn.dest,
            mil::Reg(ndx),
            "insn unvisited: {:?} <- {:?}",
            insn.dest,
            insn.insn
        );
    }

    Program {
        inner: program,
        cfg,
        is_alive,
        phis,
        rdr_count,
    }
}

fn place_phi_nodes(
    program: &mut mil::Program,
    cfg: &cfg::Graph,
    dom_tree: &cfg::DomTree,
) -> cfg::BlockMap<PhiInfo> {
    let block_count = cfg.block_count();

    if program.len() == 0 {
        return cfg::BlockMap::new(PhiInfo::empty(), block_count);
    }

    // matrix [B * B] where B = #blocks
    // matrix[i, j] = true iff block j is in block i's dominance frontier
    let is_dom_front = compute_dominance_frontier(cfg, dom_tree);

    // the rule is:
    //   if variable v is written in block b,
    //   then we have to add `v <- phi v, v, ...` on each block in b's dominance frontier
    // (phis_set is to avoid having multiple phis for the same var)
    let var_count = count_variables(program);
    let mut phis_set = Mat::new(false, block_count.into(), var_count.into());
    for bid in cfg.block_ids() {
        let bid_ndx = bid.as_usize();

        for insn_ndx in cfg.insns_ndx_range(bid) {
            let insn = program.get(insn_ndx).unwrap();
            let mil::Reg(dest_ndx) = insn.dest;
            for target_ndx in 0..cfg.block_count() {
                if *is_dom_front.item(bid_ndx, target_ndx) {
                    *phis_set.item_mut(target_ndx, dest_ndx as usize) = true;
                }
            }
        }
    }

    // translate `phis` into a representation such that inputs can be replaced with specific input
    // variables assigned in predecessors.
    // they are appended as Phi/PhiArg nodes to the program
    let mut phis = cfg::BlockMap::new(PhiInfo::empty(), block_count);

    for bid in cfg.block_ids() {
        let pred_count: u16 = cfg.block_preds(bid).len().try_into().unwrap();
        let start_ndx = program.len();
        let mut phi_count = 0;

        for var_ndx in 0..var_count {
            if *phis_set.item(bid.as_usize(), var_ndx.into()) {
                let var_ndx = var_ndx.try_into().unwrap();
                let reg = mil::Reg(var_ndx);

                program.push(reg, mil::Insn::Phi);
                phi_count += 1;

                for _pred_ndx in 0..pred_count {
                    // implicitly corresponds to _pred_ndx
                    program.push(mil::Reg(program.len()), mil::Insn::PhiArg(reg));
                }
            }
        }

        let end_ndx = program.len();
        assert_eq!(end_ndx - start_ndx, phi_count * (1 + pred_count));

        let phi_info = PhiInfo {
            ndxs: start_ndx..end_ndx,
            phi_count,
            pred_count,
        };
        {
            for phi_ndx in 0..phi_info.phi_count {
                for pred_ndx in 0..phi_info.pred_count {
                    // check that it doesn't throw
                    let ndx = phi_info.arg_ndx(phi_ndx, pred_ndx);
                    let item = program.get(ndx).unwrap();
                    assert!(matches!(&item.insn, mil::Insn::PhiArg(_)));
                }
            }
        }

        phis[bid] = phi_info;
    }

    phis
}

// TODO cache this info somewhere. it's so weird to recompute it twice!
fn count_variables(program: &mil::Program) -> mil::Index {
    use std::iter::once;

    let max_reg_ndx = program
        .iter()
        .flat_map(|insn| {
            let inputs = insn.insn.input_regs().into_iter().flatten().copied();
            once(insn.dest).chain(inputs)
        })
        .map(|reg| reg.reg_index())
        .max()
        .unwrap();

    1 + max_reg_ndx
}

struct Mat<T> {
    items: Box<[T]>,
    rows: usize,
    cols: usize,
}
impl<T: Clone> Mat<T> {
    fn new(init: T, rows: usize, cols: usize) -> Self {
        let items = vec![init; rows * cols].into_boxed_slice();
        Mat { items, rows, cols }
    }

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

fn compute_dominance_frontier(graph: &cfg::Graph, dom_tree: &cfg::DomTree) -> Mat<bool> {
    let count = graph.block_count();
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
    if prog.inner.len() == 0 {
        return;
    }

    // in this ordering, each node is always processed  before any of its parents.  it starts with
    // exit nodes.
    let postorder = cfg::traverse_postorder(&prog.cfg);

    prog.is_alive.fill(false);
    prog.rdr_count.reset();

    for &bid in postorder.order() {
        for ndx in prog.cfg.insns_ndx_range(bid).rev() {
            let item = prog.inner.get(ndx).unwrap();
            let dest = item.dest;

            let is_alive = prog.is_alive.get_mut(dest.reg_index() as usize).unwrap();
            *is_alive = item.insn.has_side_effects() || prog.rdr_count.get(dest) > 0;
            if !*is_alive {
                // this insn's reads don't count
                continue;
            }

            for &input in item.insn.input_regs().into_iter().flatten() {
                prog.rdr_count.inc(input);
            }
        }

        // at this point, rdr_count has been update for Insn::Phi  nodes; we extend is_alive to the
        // corresponding Insn::PhiArg-s

        // order of processing of phi nodes should not matter
        let mut flag = false;
        for ndx in prog.phis[bid].ndxs.clone() {
            let reg = mil::Reg(ndx);
            let is_alive = prog.is_alive.get_mut(reg.reg_index() as usize).unwrap();
            *is_alive = prog.rdr_count.get(reg) > 0;
            match prog.inner.get_mut(ndx).unwrap().insn {
                mil::Insn::Phi { .. } => {
                    flag = *is_alive;
                }
                mil::Insn::PhiArg(value) => {
                    *is_alive = flag;
                    if *is_alive {
                        prog.rdr_count.inc(*value);
                    }
                }
                _ => panic!("not phi!"),
            }
        }
    }
}

#[derive(Debug)]
struct ReaderCount(Vec<usize>);
impl ReaderCount {
    fn new(var_count: mil::Index) -> Self {
        ReaderCount(vec![0; var_count as usize])
    }
    fn reset(&mut self) {
        self.0.fill(0);
    }
    fn get(&self, reg: mil::Reg) -> usize {
        self.0[reg.0 as usize]
    }
    fn inc(&mut self, reg: mil::Reg) {
        if let Some(elm) = self.0.get_mut(reg.reg_index() as usize) {
            *elm += 1;
        }
    }
}

pub struct EditableProgram<'a>(&'a mut Program);

impl<'a> EditableProgram<'a> {
    pub fn get_mut(&mut self, ndx: mil::Index) -> Option<mil::InsnViewMut> {
        self.0.inner.get_mut(ndx)
    }
}
impl<'a> std::ops::Deref for EditableProgram<'a> {
    // only afford access to the MIL program; SSA invariants are re-checked upon
    // releasing the EditableProgram
    type Target = mil::Program;

    fn deref(&self) -> &Self::Target {
        &self.0.inner
    }
}
impl<'a> std::ops::Drop for EditableProgram<'a> {
    fn drop(&mut self) {
        self.0.check_invariants();
    }
}

#[cfg(test)]
mod tests {
    use crate::mil;

    #[test]
    fn test_phi_read() {
        use mil::{Insn, Reg};
        let prog = {
            let mut pb = mil::ProgramBuilder::new();

            pb.set_input_addr(0xf0);
            pb.push(Reg(0), Insn::Const8(123));
            pb.push(
                Reg(1),
                Insn::JmpExtIf {
                    cond: Reg(0),
                    target: 0xf2,
                },
            );

            pb.set_input_addr(0xf1);
            pb.push(Reg(2), Insn::Const1(4));
            pb.push(Reg(3), Insn::JmpExt(0xf3));

            pb.set_input_addr(0xf2);
            pb.push(Reg(2), Insn::Const1(8));

            pb.set_input_addr(0xf3);
            pb.push(Reg(4), Insn::AddK(Reg(2), 456));
            pb.push(Reg(5), Insn::Ret(Reg(4)));

            pb.build()
        };

        eprintln!("-- mil:");
        eprintln!("{:?}", prog);

        eprintln!("-- ssa:");
        let prog = super::mil_to_ssa(prog);
        insta::assert_debug_snapshot!(prog);
    }
}
