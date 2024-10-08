use std::ops::Range;

/// Static Single-Assignment representation of a program (and conversion from direct multiple
/// assignment).
///
/// The algorithms in this module are mostly derived from the descriptions in:
/// > Cooper, Keith & Harvey, Timothy & Kennedy, Ken. (2006).
/// > A Simple, Fast Dominance Algorithm.
/// > Rice University, CS Technical Report 06-33870.
use crate::{
    cfg::{self, BasicBlockID},
    mil,
};

pub struct Program {
    inner: mil::Program,
    cfg: cfg::Graph,
}

impl Program {
    pub fn dump(&self) {
        self.inner.dump()
    }
}

pub fn convert_to_ssa(mut program: mil::Program) -> Program {
    let cfg = cfg::analyze_mil(&program);
    cfg.dump_graphviz(&program);

    let dom_tree = compute_dom_tree(&cfg);
    let mut phis = place_phi_nodes(&program, &cfg, &dom_tree);

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

    struct VarMap {
        count: usize,
        mappings: Vec<mil::Reg>,
    }

    impl VarMap {
        fn new(vars_count: usize) -> Self {
            VarMap {
                count: vars_count,
                mappings: Vec::new(),
            }
        }

        fn push(&mut self) {
            self.mappings.reserve(self.count);

            if self.mappings.is_empty() {
                for _ in 0..self.count {
                    self.mappings.push(mil::Reg::Und);
                }
            } else {
                // copy the current frame's map into the new one
                for i in 0..self.count {
                    let val = self.mappings[self.mappings.len() - self.count];
                    self.mappings.push(val);
                }
            }

            assert_eq!(0, self.mappings.len() % self.count);
        }

        fn pop(&mut self) {
            let old_len = self.mappings.len();
            self.mappings.truncate(old_len - self.count);
            assert_eq!(0, self.mappings.len() % self.count);
        }

        fn current(&self) -> &[mil::Reg] {
            assert!(self.mappings.len() > 0, "no mappings!");
            let len = self.mappings.len();
            &self.mappings[len - self.count..]
        }
        fn current_mut(&mut self) -> &mut [mil::Reg] {
            let len = self.mappings.len();
            &mut self.mappings[len - self.count..]
        }

        fn get(&self, reg: mil::Reg) -> mil::Reg {
            let reg_num = reg.as_nor().expect("only Reg::Nor can be mapped") as usize;
            self.current()[reg_num]
        }

        fn set(&mut self, src: mil::Reg, dst: mil::Reg) {
            let reg_num = src.as_nor().expect("only Reg::Nor can be mapped") as usize;
            self.current_mut()[reg_num] = dst;
        }
    }

    let var_count = count_variables(&program);
    let mut var_map = VarMap::new(var_count);

    enum Cmd {
        Finish(BasicBlockID),
        Start(BasicBlockID),
    }
    let mut queue = vec![Cmd::Finish(cfg::ENTRY_BID), Cmd::Start(cfg::ENTRY_BID)];

    while let Some(cmd) = queue.pop() {
        match cmd {
            Cmd::Start(bid) => {
                var_map.push();

                // patch current block
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
                        *reg = var_map.get(*reg);
                    }

                    // in the output SSA, each destination register corrsponds to the instruction's
                    // index. this way, the numeric value of a register can also be used as
                    // instruction ID, to locate a register/variable's defining instruction.
                    let new_name = mil::Reg::Nor(insn_ndx.try_into().unwrap());
                    let old_name = std::mem::replace(insn.dest, new_name);
                    var_map.set(old_name, new_name);
                }

                // The algorithm is the following:
                // >  for s in block's successors:
                // >    for p in s's ϕ-nodes:
                // >      Assuming p is for a variable v, make it read from stack[v].
                // ... but in order to patch the correct operand in each phi instruction, we have
                // to figure out a "predecessor position", i.e. whether the current block is the
                // successor's 1st, 2nd, 3rd predecessor.

                for succ in cfg.successors(bid).into_iter().flatten() {
                    // maybe there is a better way, but I'm betting that the max nubmer of
                    // predecessor is very small (< 10), so this is plenty fast
                    // index of bid in succ's predecessor list
                    let pred_ndx = cfg
                        .predecessors(succ)
                        .iter()
                        .position(|&pred| pred == bid)
                        .unwrap();
                    for ndx in 0..phis.nodes_count(succ) {
                        let phi = phis.node_mut(succ, ndx);
                        // NOTE: the substitution of the *successor's* phi node's argument is
                        // done in the context of *this* node (its predecessor)
                        phi.args[pred_ndx] = var_map.get(phi.args[pred_ndx]);
                    }
                }

                // TODO quadratic, but cfgs are generally pretty small
                let imm_dominated = dom_tree
                    .items()
                    .filter(|(_, parent)| **parent == Some(bid))
                    .map(|(bid, _)| bid);
                for child in imm_dominated {
                    queue.push(Cmd::Finish(child));
                    queue.push(Cmd::Start(child));
                }
            }

            Cmd::Finish(bid) => {
                var_map.pop();
            }
        }
    }

    Program {
        inner: program,
        cfg,
    }
}

const ERR_NON_NOR: &str = "input program must not mention any non-Nor Reg";

fn place_phi_nodes(program: &mil::Program, cfg: &cfg::Graph, dom_tree: &DomTree) -> Phis {
    let block_count = cfg.block_count();

    if program.len() == 0 {
        return Phis::empty();
    }

    let var_count = count_variables(program);

    let mut var_written = vec![false; var_count];

    // phis is a matrix of [V * B] booleans, where V = variables count; B = blocks count.
    // it represents a set of phi nodes of a specific form:
    //   phis[v, b] == true  <==>  block b has an instr: `v <- phi v, v, ... (for all predecessors)`
    let mut phis_set = vec![false; var_count * block_count].into_boxed_slice();

    // order does not matter
    for bid in cfg.block_ids() {
        var_written.iter_mut().for_each(|it| *it = false);

        let ndxs = cfg.insns_ndx_range(bid);
        let block_start_pos = ndxs.start;
        let preds_count = cfg.predecessors(bid).len();

        for insn_ndx in ndxs {
            let insn = program.get(insn_ndx).unwrap();
            let dest = insn.dest.as_nor().expect(ERR_NON_NOR) as usize;
            var_written[dest] = true;
        }

        find_dominance_frontier(cfg, &dom_tree, bid, |dom_fr_bid| {
            // TODO use bitvec and bitwise or?
            let block_phis = {
                let ofs = bid.as_usize() * var_count;
                &mut phis_set[ofs..ofs + var_count]
            };
            assert_eq!(block_phis.len(), var_count);
            assert_eq!(var_written.len(), var_count);
            for i in 0..var_count {
                block_phis[i] |= var_written[i];
            }
        });
    }

    // translate `phis` into a representation such that inputs can be replaced with specific input
    // variables assigned in predecessors. details are in struct Phis
    let mut phis_builder = {
        let mut preds_count = cfg::BlockMap::new(usize::MAX, block_count);
        for bid in cfg.block_ids() {
            preds_count[bid] = cfg.predecessors(bid).len();
        }
        PhisBuilder::new(preds_count)
    };

    for bid in cfg.block_ids() {
        phis_builder.set_start(bid);

        let phis_ofs = bid.as_usize() * var_count;
        phis_set[phis_ofs..phis_ofs + var_count]
            .iter()
            .enumerate()
            .filter(|(_, is_there)| **is_there)
            .for_each(|(var_ndx, _)| {
                let reg_id = var_ndx.try_into().unwrap();
                let var = mil::Reg::Nor(reg_id);
                phis_builder.push_init(var);
            });

        phis_builder.set_end(bid);
    }

    phis_builder.build()
}

/// Collection of phi nodes.
///
/// This representation of a collection of phi nodes is intended to augment a pre-existing
/// mil::Program, to support register renaming as part of SSA conversion. It allows us to add phi
/// nodes without adding any instruction in the mil::Program, therefore without invalidating any
/// outstanding mil::Index-es.
///
struct Phis {
    preds_count: cfg::BlockMap<usize>,
    phi_ndx_offset: cfg::BlockMap<Range<usize>>,
    dest: Box<[mil::Reg]>,
    args: Box<[mil::Reg]>,
    max_preds_count: usize,
}
struct PhiViewMut<'a> {
    dest: &'a mut mil::Reg,
    args: &'a mut [mil::Reg],
}

impl Phis {
    fn empty() -> Self {
        Phis {
            preds_count: cfg::BlockMap::new(0, 0),
            phi_ndx_offset: cfg::BlockMap::new(0..0, 0),
            dest: Box::from([]),
            args: Box::from([]),
            max_preds_count: 0,
        }
    }

    fn nodes_count(&mut self, bid: cfg::BasicBlockID) -> usize {
        self.phi_ndx_offset[bid].len()
    }

    fn node_mut(&mut self, bid: cfg::BasicBlockID, ndx: usize) -> PhiViewMut {
        let range_ndx = &self.phi_ndx_offset[bid];
        let abs_ndx = range_ndx.start + ndx;
        assert!(abs_ndx < range_ndx.end);

        PhiViewMut {
            dest: &mut self.dest[abs_ndx],
            args: &mut self.args
                [abs_ndx * self.max_preds_count..(abs_ndx + 1) * self.max_preds_count],
        }
    }
}

struct PhisBuilder {
    preds_count: cfg::BlockMap<usize>,
    phi_ndx_offset: cfg::BlockMap<Range<usize>>,
    dest: Vec<mil::Reg>,
    args: Vec<mil::Reg>,
    max_preds_count: usize,
}

impl PhisBuilder {
    const REG_UNINIT: mil::Reg = mil::Reg::Nor(u16::MAX);

    fn new(preds_count: cfg::BlockMap<usize>) -> Self {
        let block_count = preds_count.len();
        let max_preds_count = preds_count.iter().max().copied().unwrap_or(0);
        PhisBuilder {
            preds_count,
            phi_ndx_offset: cfg::BlockMap::new(0..0, block_count),
            dest: Vec::new(),
            args: Vec::new(),
            max_preds_count,
        }
    }

    fn nodes_count(&self) -> usize {
        assert_eq!(0, self.args.len() % self.max_preds_count);
        self.args.len() / self.max_preds_count
    }
    fn set_start(&mut self, bid: cfg::BasicBlockID) {
        let init_ndx = self.nodes_count();
        self.phi_ndx_offset[bid] = init_ndx..init_ndx;
    }
    fn set_end(&mut self, bid: cfg::BasicBlockID) {
        self.phi_ndx_offset[bid].end = self.nodes_count();
    }
    /// Add a phi node of the form `reg <- phi reg, reg, reg...` (with as many arguments as
    /// there are predecessors)
    fn push_init(&mut self, reg: mil::Reg) {
        self.dest.push(reg);
        for _ in 0..self.max_preds_count {
            self.args.push(reg);
        }
    }

    fn build(self) -> Phis {
        assert_eq!(self.preds_count.len(), self.phi_ndx_offset.len());
        assert_eq!(self.args.len() % self.max_preds_count, 0);
        assert_eq!(self.args.len() / self.max_preds_count, self.dest.len());
        Phis {
            preds_count: self.preds_count,
            phi_ndx_offset: self.phi_ndx_offset,
            dest: self.dest.into_boxed_slice(),
            args: self.args.into_boxed_slice(),
            max_preds_count: self.max_preds_count,
        }
    }
}

// TODO cache this info somewhere. it's so weird to recompute it twice!
fn count_variables(program: &mil::Program) -> usize {
    1 + program
        .iter()
        .map(|insn| insn.dest.as_nor().expect(ERR_NON_NOR))
        .max()
        .unwrap() as usize
}

type DomTree = cfg::BlockMap<Option<cfg::BasicBlockID>>;

pub fn compute_dom_tree(cfg: &cfg::Graph) -> DomTree {
    let block_count = cfg.block_count();
    let rpo = cfg::traverse_reverse_postorder(cfg);

    let mut parent = cfg::BlockMap::new(None, block_count);

    // process the entry node "manually", so the algorithm can rely on it for successors
    parent[cfg::ENTRY_BID] = Some(cfg::ENTRY_BID);

    let mut changed = true;
    while changed {
        changed = false;

        for &bid in rpo.order().iter() {
            let preds = cfg.predecessors(bid);
            if preds.is_empty() {
                continue;
            }

            // start with the first unprocessed predecessor
            let (idom_init_ndx, &(mut idom)) = preds
                .iter()
                .enumerate()
                .find(|(pred_ndx, _)| parent[preds[*pred_ndx]].is_some())
                .expect("rev. postorder bug: all predecessors are yet to be processed");

            for (pred_ndx, &pred) in preds.iter().enumerate() {
                if pred_ndx == idom_init_ndx {
                    continue;
                }

                if parent[pred].is_some() {
                    idom = common_ancestor(
                        &parent,
                        |id_a, id_b| rpo.position_of(id_a) < rpo.position_of(id_b),
                        pred,
                        idom,
                    );
                }
            }

            let prev_idom = parent[bid].replace(idom);
            if prev_idom != Some(idom) {
                changed = true;
            }
        }
    }

    // we hand the tree out with a slightly different convention: the root node has no parent in
    // the tree, so the corresponding item is None.  up to this point the root is linked to itself,
    // as required by the algorithm by how it's formulated
    parent[cfg::ENTRY_BID] = None;
    parent
}

/// Find the common ancestor of two nodes in a tree.
///
/// The tree is presumed to have progressively numbered nodes. It is represented as an array
/// `parent_of` such that, for each node with index _i_, parent_of[i] is the index of the parent
/// node (or _i_, the same index, for the root node).
fn common_ancestor<LT>(
    parent_of: &DomTree,
    is_lt: LT,
    mut ndx_a: cfg::BasicBlockID,
    mut ndx_b: cfg::BasicBlockID,
) -> cfg::BasicBlockID
where
    LT: Fn(cfg::BasicBlockID, cfg::BasicBlockID) -> bool,
{
    while ndx_a != ndx_b {
        let mut count = parent_of.len();
        while is_lt(ndx_a, ndx_b) {
            ndx_b = parent_of[ndx_b].unwrap();
            count -= 1;
        }
        let mut count = parent_of.len();
        while is_lt(ndx_b, ndx_a) {
            ndx_a = parent_of[ndx_a].unwrap();
            count -= 1;
        }
    }

    ndx_a
}

fn find_dominance_frontier(
    graph: &cfg::Graph,
    dom_tree: &cfg::BlockMap<Option<cfg::BasicBlockID>>,
    node: cfg::BasicBlockID,
    mut on_found: impl FnMut(cfg::BasicBlockID),
) {
    let preds = graph.predecessors(node);
    if preds.len() < 2 {
        return;
    }

    let runner_stop = dom_tree[node].unwrap();
    for &pred in preds {
        let mut runner = pred;
        while runner != runner_stop {
            on_found(runner);
            runner = dom_tree[runner].unwrap();
        }
    }
}

fn dump_tree_dot(dom_tree: cfg::BlockMap<Option<cfg::BasicBlockID>>) {
    println!("digraph {{");
    for (bid, _) in dom_tree.items() {
        let bid = bid.as_number();
        println!("  block{} [label=\"{}\"]", bid, bid);
    }
    for (bid, parent) in dom_tree.items() {
        if let Some(parent) = parent {
            println!("  block{} -> block{}", bid.as_number(), parent.as_number());
        }
    }
    println!("}}");
}

pub fn eliminate_dead_code(prog: &mut Program) {}
