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
    is_alive: RegMap<bool>,
    phis: Phis,
    rdr_count: ReaderCount,
}

impl Program {
    pub fn dump(&self) {
        let count: usize = self.is_alive.iter().map(|x| *x as usize).sum();
        println!("ssa program  {} instrs", count);

        for bid in self.cfg.block_ids() {
            let ndxs = self.cfg.insns_ndx_range(bid);
            let block_addr = self.inner.get(ndxs.start).unwrap().addr;

            println!(".B{}:   ;; 0x{:x}", bid.as_usize(), block_addr);

            for phi_ndx in 0..self.phis.nodes_count(bid) {
                let phi = self.phis.node(bid, phi_ndx);
                if !*self.is_alive.get(*phi.dest).unwrap() {
                    continue;
                }
                print!("         {:?} <- phi", phi.dest);
                for (pred, arg) in self.cfg.predecessors(bid).iter().zip(phi.args) {
                    print!(" .B{}:{:?}", pred.as_usize(), arg);
                }
                println!();
            }

            for ndx in ndxs {
                let nor_ndx = ndx.try_into().unwrap();
                if !*self.is_alive.get(mil::Reg::Nor(nor_ndx)).unwrap() {
                    continue;
                }

                let item = self.inner.get(ndx).unwrap();
                let rdr_count = self.rdr_count.get(item.dest);
                if rdr_count > 1 {
                    print!("  ({:3})  ", rdr_count);
                } else {
                    print!("         ");
                }
                print!("{:?} <- ", item.dest);
                item.insn.dump();
                println!();
            }
        }
    }
}

pub fn convert_to_ssa(mut program: mil::Program) -> Program {
    let cfg = cfg::analyze_mil(&program);
    let dom_tree = cfg::compute_dom_tree(&cfg);
    cfg.dump_graphviz(Some(&dom_tree));

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

    // TODO rework this with a more efficient data structure
    struct VarMap {
        count: usize,
        stack: Vec<Box<[mil::Reg]>>,
    }

    impl VarMap {
        fn new(vars_count: usize) -> Self {
            VarMap {
                count: vars_count,
                stack: Vec::new(),
            }
        }

        fn push(&mut self) {
            if self.stack.is_empty() {
                self.stack
                    .push(vec![mil::Reg::Und; self.count].into_boxed_slice());
            } else {
                let new_elem = self.stack.last().unwrap().clone();
                self.stack.push(new_elem);
            }

            self.check_invariants()
        }

        fn check_invariants(&self) {
            for elem in &self.stack {
                assert_eq!(elem.len(), self.count);
            }
        }

        fn pop(&mut self) {
            self.stack.pop();
            self.check_invariants();
        }

        fn current(&self) -> &[mil::Reg] {
            &self.stack.last().expect("no mappings!")[..]
        }
        fn current_mut(&mut self) -> &mut [mil::Reg] {
            &mut self.stack.last_mut().expect("no mappings!")[..]
        }
        fn get(&self, reg: mil::Reg) -> mil::Reg {
            let reg_num = reg.as_nor().expect("only Reg::Nor can be mapped") as usize;
            self.current()[reg_num]
        }

        fn set(&mut self, old: mil::Reg, new: mil::Reg) {
            let reg_num = old.as_nor().expect("only Reg::Nor can be mapped") as usize;
            self.current_mut()[reg_num] = new;
        }
    }

    let var_count = count_variables(&program);
    let mut var_map = VarMap::new(var_count);

    enum Cmd {
        Finish,
        Start(BasicBlockID),
    }
    let mut queue = vec![Cmd::Finish, Cmd::Start(cfg::ENTRY_BID)];

    while let Some(cmd) = queue.pop() {
        match cmd {
            Cmd::Start(bid) => {
                var_map.push();

                // -- patch current block

                // phi nodes are *used* here for their destination; their arguments are fixed up
                // while processing the predecessors
                for block_phi_ndx in 0..phis.nodes_count(bid) {
                    let glob_phi_ndx = phis.flat_ndx(bid, block_phi_ndx);
                    let phi = phis.flat_get_mut(glob_phi_ndx);
                    let new_phi = mil::Reg::Phi(glob_phi_ndx.try_into().unwrap());
                    var_map.set(*phi.dest, new_phi);
                    *phi.dest = new_phi;
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
                        *reg = var_map.get(*reg);
                    }

                    // in the output SSA, each destination register corrsponds to the instruction's
                    // index. this way, the numeric value of a register can also be used as
                    // instruction ID, to locate a register/variable's defining instruction.
                    let new_dest = mil::Reg::Nor(insn_ndx.try_into().unwrap());
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

                // -- patch successor's phi nodes
                // The algorithm is the following:
                // >  for s in block's successors:
                // >    for p in s's ϕ-nodes:
                // >      Assuming p is for a variable v, make it read from stack[v].
                //
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
    let is_alive = RegMap::new(true, var_count, phis.len());
    let rdr_count = ReaderCount::new(var_count, phis.len());

    // establish SSA invariants
    // the returned Program will no longer change (just some instructions are going to be marked as
    // "dead" and ignored)
    for (ndx, insn) in program.iter().enumerate() {
        let ndx = ndx.try_into().unwrap();
        assert_eq!(insn.dest, mil::Reg::Nor(ndx));
    }
    for (ndx, phi_dest) in phis.dest.iter().enumerate() {
        let ndx = ndx.try_into().unwrap();
        assert_eq!(phi_dest, &mil::Reg::Phi(ndx));
    }

    let mut program = Program {
        inner: program,
        cfg,
        is_alive,
        phis,
        rdr_count,
    };
    eliminate_dead_code(&mut program);
    program
}

const ERR_NON_NOR: &str = "input program must not mention any non-Nor Reg";

fn place_phi_nodes(program: &mil::Program, cfg: &cfg::Graph, dom_tree: &cfg::DomTree) -> Phis {
    let block_count = cfg.block_count();

    if program.len() == 0 {
        return Phis::empty();
    }

    // matrix [B * B] where B = #blocks
    // matrix[i, j] = true iff block j is in block i's dominance frontier
    let is_dom_front = compute_dominance_frontier(cfg, dom_tree);

    // the rule is:
    //   if variable v is written in block b,
    //   then we have to add `v <- phi v, v, ...` on each block in b's dominance frontier
    // (phis_set is to avoid having multiple phis for the same var)
    let var_count = count_variables(program);
    let mut phis_set = Mat::new(false, block_count, var_count);
    for bid in cfg.block_ids() {
        let bid_ndx = bid.as_usize();

        for insn_ndx in cfg.insns_ndx_range(bid) {
            let insn = program.get(insn_ndx).unwrap();
            if let mil::Reg::Nor(dest_ndx) = insn.dest {
                for target_ndx in 0..cfg.block_count() {
                    if *is_dom_front.item(bid_ndx, target_ndx) {
                        *phis_set.item_mut(target_ndx, dest_ndx as usize) = true;
                    }
                }
            }
        }
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
        let block_ndx = bid.as_usize();

        for var_ndx in 0..var_count {
            if *phis_set.item(block_ndx, var_ndx) {
                let var_ndx = var_ndx.try_into().unwrap();
                phis_builder.push_init(mil::Reg::Nor(var_ndx));
            }
        }

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
#[derive(Debug)]
struct Phis {
    block_ndxs: cfg::BlockMap<Range<usize>>,
    dest: Box<[mil::Reg]>,
    args: Box<[mil::Reg]>,
    max_preds_count: usize,
}
struct PhiViewMut<'a> {
    dest: &'a mut mil::Reg,
    args: &'a mut [mil::Reg],
}
struct PhiView<'a> {
    dest: &'a mil::Reg,
    args: &'a [mil::Reg],
}

impl Phis {
    fn empty() -> Self {
        Phis {
            block_ndxs: cfg::BlockMap::new(0..0, 0),
            dest: Box::from([]),
            args: Box::from([]),
            max_preds_count: 0,
        }
    }

    #[inline]
    fn nodes_count(&self, bid: cfg::BasicBlockID) -> usize {
        self.block_ndxs[bid].len()
    }

    #[inline]
    fn node(&self, bid: cfg::BasicBlockID, ndx: usize) -> PhiView {
        let flat_ndx = self.flat_ndx(bid, ndx);
        self.flat_get(flat_ndx)
    }

    #[inline]
    fn node_mut(&mut self, bid: cfg::BasicBlockID, ndx: usize) -> PhiViewMut {
        let flat_ndx = self.flat_ndx(bid, ndx);
        self.flat_get_mut(flat_ndx)
    }

    #[inline]
    fn flat_ndx(&self, bid: BasicBlockID, ndx: usize) -> usize {
        let range_ndx = &self.block_ndxs[bid];
        let abs_ndx = range_ndx.start + ndx;
        assert!(abs_ndx < range_ndx.end);
        abs_ndx
    }

    fn len(&self) -> mil::Index {
        self.dest.len().try_into().unwrap()
    }
    fn flat_get(&self, flat_ndx: usize) -> PhiView {
        PhiView {
            dest: &self.dest[flat_ndx],
            args: &self.args
                [flat_ndx * self.max_preds_count..(flat_ndx + 1) * self.max_preds_count],
        }
    }
    fn flat_get_mut(&mut self, flat_ndx: usize) -> PhiViewMut<'_> {
        PhiViewMut {
            dest: &mut self.dest[flat_ndx],
            args: &mut self.args
                [flat_ndx * self.max_preds_count..(flat_ndx + 1) * self.max_preds_count],
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
        assert!(matches!(reg, mil::Reg::Nor(_)));
        self.dest.push(reg);
        for _ in 0..self.max_preds_count {
            self.args.push(reg);
        }
    }

    fn build(self) -> Phis {
        assert_eq!(self.preds_count.len(), self.phi_ndx_offset.len());
        assert_eq!(self.args.len() % self.max_preds_count, 0);
        assert_eq!(self.args.len() / self.max_preds_count, self.dest.len());
        for (i, range_a) in self.phi_ndx_offset.iter().enumerate() {
            for (j, range_b) in self.phi_ndx_offset.iter().enumerate() {
                if i == j {
                    continue;
                }
                assert!(range_b.start >= range_a.end || range_b.end <= range_a.start);
            }
        }
        Phis {
            block_ndxs: self.phi_ndx_offset,
            dest: self.dest.into_boxed_slice(),
            args: self.args.into_boxed_slice(),
            max_preds_count: self.max_preds_count,
        }
    }
}

// TODO cache this info somewhere. it's so weird to recompute it twice!
fn count_variables(program: &mil::Program) -> usize {
    use std::iter::once;

    let max_reg_ndx = program
        .iter()
        .flat_map(|insn| {
            once(insn.dest).chain(insn.insn.input_regs().into_iter().flatten().copied())
        })
        .map(|reg| reg.as_nor().expect(ERR_NON_NOR))
        .max()
        .unwrap() as usize;

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

fn compute_dominance_frontier(
    graph: &cfg::Graph,
    dom_tree: &cfg::BlockMap<Option<cfg::BasicBlockID>>,
) -> Mat<bool> {
    let count = graph.block_count();
    let mut mat = Mat::new(false, count, count);

    for bid in graph.block_ids() {
        let preds = graph.predecessors(bid);
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
            let dest = match item.dest {
                mil::Reg::Nor(_) | mil::Reg::Phi(_) => item.dest,
                mil::Reg::Und => {
                    continue;
                }
            };

            let is_alive = prog.is_alive.get_mut(dest).unwrap();
            *is_alive = item.insn.is_control_flow() || prog.rdr_count.get(dest) > 0;
            if !*is_alive {
                // this insn's reads don't count
                continue;
            }

            for &input in item.insn.input_regs().into_iter().flatten() {
                prog.rdr_count.inc(input);
            }
        }

        // order of processing of phi nodes should not matter
        for phi_ndx in 0..prog.phis.nodes_count(bid) {
            let phi = prog.phis.node(bid, phi_ndx);
            let dest = *phi.dest;
            let is_alive = prog.is_alive.get_mut(dest).unwrap();
            *is_alive = prog.rdr_count.get(dest) > 0;
            if !*is_alive {
                // this insn's reads don't count
                continue;
            }
            for &arg in phi.args {
                prog.rdr_count.inc(arg);
            }
        }
    }
}

struct RegMap<T> {
    nor: Box<[T]>,
    phi: Box<[T]>,
}
impl<T: Clone> RegMap<T> {
    fn new(init: T, nor_count: mil::Index, phi_count: mil::Index) -> Self {
        RegMap {
            nor: vec![init.clone(); nor_count as usize].into_boxed_slice(),
            phi: vec![init; phi_count as usize].into_boxed_slice(),
        }
    }

    fn fill(&mut self, value: T) {
        self.nor.fill(value.clone());
        self.phi.fill(value);
    }

    fn get(&self, reg: mil::Reg) -> Option<&T> {
        match reg {
            mil::Reg::Nor(nor_ndx) => Some(&self.nor[nor_ndx as usize]),
            mil::Reg::Phi(phi_ndx) => Some(&self.phi[phi_ndx as usize]),
            mil::Reg::Und => None,
        }
    }

    fn get_mut(&mut self, reg: mil::Reg) -> Option<&mut T> {
        match reg {
            mil::Reg::Nor(nor_ndx) => Some(&mut self.nor[nor_ndx as usize]),
            mil::Reg::Phi(phi_ndx) => Some(&mut self.phi[phi_ndx as usize]),
            mil::Reg::Und => None,
        }
    }

    fn iter(&self) -> impl Iterator<Item = &T> {
        self.nor.iter().chain(self.phi.iter())
    }
}

struct ReaderCount(RegMap<usize>);
impl ReaderCount {
    fn new(nor_count: mil::Index, phi_count: mil::Index) -> Self {
        ReaderCount(RegMap::new(0, nor_count, phi_count))
    }
    fn reset(&mut self) {
        self.0.fill(0);
    }
    fn get(&self, reg: mil::Reg) -> usize {
        *self.0.get(reg).unwrap()
    }
    fn inc(&mut self, reg: mil::Reg) {
        if let Some(elm) = self.0.get_mut(reg) {
            *elm += 1;
        }
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
            pb.push(Reg::Nor(0), Insn::Const8(123));
            pb.push(
                Reg::Nor(1),
                Insn::JmpIfK {
                    cond: Reg::Nor(0),
                    target: 0xf2,
                },
            );

            pb.set_input_addr(0xf1);
            pb.push(Reg::Nor(2), Insn::Const1(4));
            pb.push(Reg::Nor(3), Insn::JmpK(0xf3));

            pb.set_input_addr(0xf2);
            pb.push(Reg::Nor(2), Insn::Const1(8));

            pb.set_input_addr(0xf3);
            pb.push(Reg::Nor(4), Insn::AddK(Reg::Nor(2), 456));
            pb.push(Reg::Nor(5), Insn::Ret(Reg::Nor(4)));

            pb.build()
        };

        eprintln!("-- mil:");
        prog.dump();

        eprintln!("-- ssa:");
        let prog = super::mil_to_ssa(prog);
        prog.dump();

        let bid = prog.cfg.block_starting_at(5).unwrap();
        let phi_count = prog.phis.nodes_count(bid);
        assert_eq!(phi_count, 2);
        // assert_eq!(prog.phis.node(bid, 0),);
        assert!(false);
    }
}
