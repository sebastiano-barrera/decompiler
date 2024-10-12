/// Control Flow Graph.
///
/// Routines and data types to extract and represent the control-flow graph (basic blocks and their
/// sequence relationships).
use std::{
    collections::HashMap,
    ops::{Index, IndexMut, Range},
};

use crate::mil;

/// Control Flow Graph
#[derive(Debug)]
pub struct Graph {
    bounds: Vec<mil::Index>,
    // successors[bndx] = successors to block #bndx
    successors: BlockMap<BlockCont>,
    predecessors: Vec<BasicBlockID>,
    pred_ndx_range: Vec<Range<usize>>,
}

#[derive(Debug)]
enum BlockCont {
    End,
    Jmp(BasicBlockID),
    Alt(BasicBlockID, BasicBlockID),
}

impl BlockCont {
    #[inline]
    fn as_array(&self) -> [Option<BasicBlockID>; 2] {
        match self {
            BlockCont::End => [None, None],
            BlockCont::Jmp(d) => [Some(*d), None],
            BlockCont::Alt(d, e) => [Some(*d), Some(*e)],
        }
    }
}

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
pub struct BasicBlockID(u16);

pub const ENTRY_BID: BasicBlockID = BasicBlockID(0);

impl BasicBlockID {
    #[inline(always)]
    pub fn as_number(&self) -> u16 {
        self.0
    }

    #[inline(always)]
    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }
}

impl Graph {
    #[inline(always)]
    pub fn block_count(&self) -> usize {
        self.bounds.len() - 1
    }

    pub fn block_ids(&self) -> impl Iterator<Item = BasicBlockID> {
        (0..self.block_count()).map(|ndx| BasicBlockID(ndx.try_into().unwrap()))
    }

    pub fn predecessors(&self, bndx: BasicBlockID) -> &[BasicBlockID] {
        let range = &self.pred_ndx_range[bndx.0 as usize];
        &self.predecessors[range.start..range.end]
    }

    pub fn successors(&self, bid: BasicBlockID) -> [Option<BasicBlockID>; 2] {
        self.successors[bid].as_array()
    }

    pub fn insns_ndx_range(&self, bid: BasicBlockID) -> Range<mil::Index> {
        let ndx = bid.as_usize();
        let start = self.bounds[ndx];
        let end = self.bounds[ndx + 1];
        // must contain at least 1 instruction
        // TODO hoist this assertion somewhere else?
        assert!(end > start);
        start..end
    }
}

pub fn analyze_mil(program: &mil::Program) -> Graph {
    let bounds = {
        let mut bounds = Vec::with_capacity(program.len() as usize / 5);
        bounds.push(0);

        for ndx in 0..program.len() {
            match dest_of_insn(program, ndx) {
                // (straight_dest, side_dest)
                (None, None) => panic!("all instructions must go *somewhere*!"),
                (_, None) => {
                    // here we don't care about straight_dest, as it's the default and does not influence
                    // basic block structure
                }
                (None, Some(a)) => {
                    bounds.push(a);
                }
                (Some(a), Some(b)) => {
                    bounds.push(a);
                    bounds.push(b);
                }
            }
        }

        bounds.push(program.len());
        bounds.sort();
        bounds.dedup();
        bounds
    };

    let block_count = bounds.len() - 1;

    let block_at: HashMap<_, _> = bounds[..block_count]
        .iter()
        .enumerate()
        .map(|(bndx, start_ndx)| {
            let bndx = bndx.try_into().unwrap();
            (*start_ndx, BasicBlockID(bndx))
        })
        .collect();

    // if bounds == [b0, b1, b2, b3, ...]
    // then the basic blocks span instructions at indices [b0, b1), [b1, b2), [b2, b3), ...
    // in particular, basic block at index bndx spans [bounds[bndx], bounds[bndx+1])

    let successors: Vec<_> = bounds[1..]
        .iter()
        .map(|end_ndx| {
            let last_ndx = end_ndx - 1;
            match dest_of_insn(program, last_ndx) {
                (None, None) => panic!("all instructions must lead *somewhere*!"),
                (None, Some(dest)) if dest == program.len() => BlockCont::End,
                (Some(dest), None) | (None, Some(dest)) => {
                    BlockCont::Jmp(*block_at.get(&dest).unwrap())
                }
                (Some(straight_dest), Some(side_dest)) => BlockCont::Alt(
                    *block_at.get(&straight_dest).unwrap(),
                    *block_at.get(&side_dest).unwrap(),
                ),
            }
        })
        .collect();
    let successors = BlockMap(successors);

    let mut pred_ndx_range = Vec::with_capacity(block_count);
    let mut predecessors = Vec::with_capacity(block_count * 2);

    // quadratic, but you know how life goes
    for bndx in 0..block_count {
        let bid = Some(BasicBlockID(bndx.try_into().unwrap()));

        let pred_offset = predecessors.len();
        let mut pred_count = 0;

        for (pred_ndx, cont) in successors.iter().enumerate() {
            let pred_ndx = pred_ndx.try_into().unwrap();
            let [a, b] = cont.as_array();
            if a == bid || b == bid {
                predecessors.push(BasicBlockID(pred_ndx));
                pred_count += 1;
            }
        }

        pred_ndx_range.push(pred_offset..pred_offset + pred_count);
    }

    #[cfg(debug_assertions)]
    {
        let is_covered = {
            let mut is = vec![false; program.len() as usize];
            for (&start_ndx, &end_ndx) in bounds.iter().zip(bounds[1..].iter()) {
                let start_ndx = start_ndx as usize;
                let end_ndx = end_ndx as usize;
                for item in &mut is[start_ndx..end_ndx] {
                    *item = true;
                }
            }
            is
        };

        let uncovered: Vec<_> = is_covered
            .into_iter()
            .enumerate()
            .filter(|(_, is)| !*is)
            .map(|(ndx, _)| ndx)
            .collect();

        // this way if the assertion fails we can see which indices are uncovered
        debug_assert_eq!(&uncovered, &[]);

        let invalid: Vec<_> = bounds
            .iter()
            .copied()
            .filter(|&i| i > program.len())
            .collect();
        debug_assert_eq!(&invalid, &[]);

        // no duplicates (<=> no 0-length blocks)
        debug_assert!(bounds.iter().zip(bounds[1..].iter()).all(|(a, b)| a != b));
    }

    Graph {
        bounds,
        successors,
        predecessors,
        pred_ndx_range,
    }
}

fn dest_of_insn(
    program: &mil::Program,
    ndx: mil::Index,
) -> (Option<mil::Index>, Option<mil::Index>) {
    let insn = program.get(ndx).unwrap().insn;
    match insn {
        mil::Insn::JmpK(target) => {
            let index = program.index_of_addr(*target).unwrap();
            (None, Some(index))
        }
        mil::Insn::JmpIfK { target, .. } => {
            let index = program.index_of_addr(*target).unwrap();
            (Some(ndx + 1), Some(index))
        }
        mil::Insn::Ret(_) => {
            // one-past-the-end of the program is a valid index; signifies "exit the function"
            (None, Some(program.len()))
        }
        mil::Insn::Jmp(_) => todo!("indirect jmp"),
        _ => (Some(ndx + 1), None),
    }
}

impl Graph {
    pub fn dump_graphviz(&self, dom_tree: Option<&DomTree>) {
        let count = self.block_count();
        println!("digraph {{");
        println!("  // {} blocks", count);

        for bndx in 0..count {
            let start_ndx = self.bounds[bndx];
            let end_ndx = self.bounds[bndx + 1];

            println!(
                "  block{} [label=\"{}\\n{}..{}\"];",
                bndx, bndx, start_ndx, end_ndx,
            );

            let bid = BasicBlockID(bndx.try_into().unwrap());
            match self.successors[bid] {
                BlockCont::End => println!("  block{} -> end", bndx),
                BlockCont::Jmp(BasicBlockID(dest)) => println!("  block{} -> block{}", bndx, dest),
                BlockCont::Alt(BasicBlockID(a), BasicBlockID(b)) => {
                    println!("  block{} -> block{};", bndx, a);
                    println!("  block{} -> block{};", bndx, b);
                }
            }

            println!();
        }

        if let Some(dom_tree) = dom_tree {
            println!("  // dominator tree");
            for (bid, parent) in dom_tree.items() {
                if let Some(parent) = parent {
                    println!(
                        "  block{} -> block{} [style=\"dotted\"];",
                        bid.as_number(),
                        parent.as_number()
                    );
                }
            }
        }

        println!("}}");
    }
}

pub type DomTree = BlockMap<Option<BasicBlockID>>;

pub fn compute_dom_tree(cfg: &Graph) -> DomTree {
    let block_count = cfg.block_count();
    let rpo = traverse_reverse_postorder(cfg);

    let mut parent = BlockMap::new(None, block_count);

    // process the entry node "manually", so the algorithm can rely on it for successors
    parent[ENTRY_BID] = Some(ENTRY_BID);

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
    parent[ENTRY_BID] = None;
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
    mut ndx_a: BasicBlockID,
    mut ndx_b: BasicBlockID,
) -> BasicBlockID
where
    LT: Fn(BasicBlockID, BasicBlockID) -> bool,
{
    while ndx_a != ndx_b {
        while is_lt(ndx_a, ndx_b) {
            ndx_b = parent_of[ndx_b].unwrap();
        }
        while is_lt(ndx_b, ndx_a) {
            ndx_a = parent_of[ndx_a].unwrap();
        }
    }

    ndx_a
}

//
// Traversals
//
pub fn traverse_reverse_postorder(graph: &Graph) -> Ordering {
    Ordering::new(reverse_postorder(graph))
}

pub fn traverse_postorder(graph: &Graph) -> Ordering {
    let mut order = reverse_postorder(graph);
    order.reverse();
    Ordering::new(order)
}

fn reverse_postorder(graph: &Graph) -> Vec<BasicBlockID> {
    let count = graph.block_count();

    // Remaining predecessors count
    // let mut rem_preds_count = BlockMap::new(0, count);
    // for bid in graph.block_ids() {
    //     rem_preds_count[bid] = graph.predecessors(bid).len();
    // }
    let mut rem_preds_count = count_nonbackedge_predecessors(graph);

    let mut order = Vec::with_capacity(count);
    let mut queue = Vec::with_capacity(count / 2);
    // we must avoid re-visiting a fully processed node. this happens when processing backedges.
    let mut visited = BlockMap::new(false, count);

    queue.push(ENTRY_BID);
    // formally incorrect, but aligns it with the rest of the algorithm
    rem_preds_count[ENTRY_BID] = 1;

    while let Some(bid) = queue.pop() {
        // each node X must be processed (added to the ordering) only after all its P predecessors
        // have been processed.  we achieve this by discarding X queue items corresponding to (P-1)
        // times before processing it the P-th time.

        rem_preds_count[bid] -= 1;
        if rem_preds_count[bid] > 0 {
            continue;
        }

        assert_eq!(rem_preds_count[bid], 0);
        order.push(bid);
        visited[bid] = true;

        let block_succs: [_; 2] = graph.successors[bid].as_array();
        for succ in block_succs.into_iter().flatten() {
            if !visited[succ] {
                queue.push(succ);
            }
        }
    }

    // all incoming edges have been processed
    if let Some(max) = rem_preds_count.iter().max() {
        assert_eq!(&0, max);
    }
    assert_eq!(order.len(), count);
    order
}

/// Count, for each node, the number of incoming edges (or, equivalently, predecessor nodes) that
/// are not back-edges (i.e. don't form a cycle).
fn count_nonbackedge_predecessors(graph: &Graph) -> BlockMap<u16> {
    let count = graph.block_count();

    #[derive(Debug, PartialEq, Eq, Clone, Copy)]
    enum Color {
        Unvisited,
        Visiting,
        Finished,
    }

    let mut incoming_count = BlockMap::new(0, count);
    let mut color = BlockMap::new(Color::Unvisited, count);

    let mut queue = Vec::with_capacity(count / 2);
    queue.push(ENTRY_BID);

    while let Some(bid) = queue.pop() {
        match color[bid] {
            Color::Unvisited => {
                // schedule Visiting -> Finished after all children have been visited
                queue.push(bid);
                for succ in graph.successors[bid].as_array().into_iter().flatten() {
                    if color[succ] == Color::Unvisited {
                        queue.push(succ);
                    }
                    if color[succ] != Color::Visiting {
                        incoming_count[succ] += 1;
                    }
                }

                color[bid] = Color::Visiting;
            }
            Color::Visiting => {
                color[bid] = Color::Finished;
            }
            Color::Finished => panic!("unreachable!"),
        }
    }

    incoming_count
}

pub struct Ordering {
    order: Vec<BasicBlockID>,
    pos_of: BlockMap<usize>,
}

impl Ordering {
    pub fn new(order: Vec<BasicBlockID>) -> Self {
        let mut pos_of = BlockMap::new(0, order.len());
        let mut occurs_count = BlockMap::new(0, order.len());
        for (pos, &bid) in order.iter().enumerate() {
            occurs_count[bid] += 1;
            pos_of[bid] = pos;
        }

        assert!(occurs_count.iter().all(|count| *count == 1));

        Ordering { order, pos_of }
    }

    pub fn order(&self) -> &[BasicBlockID] {
        &self.order
    }

    pub fn position_of(&self, bid: BasicBlockID) -> usize {
        self.pos_of[bid]
    }
}

//
// Utilities
//
#[derive(Debug)]
pub struct BlockMap<T>(Vec<T>);

impl<T: Clone> BlockMap<T> {
    pub fn new(init: T, count: usize) -> Self {
        let vec = vec![init; count];
        BlockMap(vec)
    }

    pub fn items(&self) -> impl ExactSizeIterator<Item = (BasicBlockID, &T)> {
        self.0.iter().enumerate().map(|(ndx, item)| {
            let ndx = ndx.try_into().unwrap();
            (BasicBlockID(ndx), item)
        })
    }
}

impl<T> Into<Vec<T>> for BlockMap<T> {
    fn into(self) -> Vec<T> {
        self.0
    }
}

impl<T> Index<BasicBlockID> for BlockMap<T> {
    type Output = T;

    fn index(&self, index: BasicBlockID) -> &Self::Output {
        self.0.index(index.0 as usize)
    }
}
impl<T> IndexMut<BasicBlockID> for BlockMap<T> {
    fn index_mut(&mut self, index: BasicBlockID) -> &mut Self::Output {
        self.0.index_mut(index.0 as usize)
    }
}
impl<T> std::ops::Deref for BlockMap<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T> std::ops::DerefMut for BlockMap<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
