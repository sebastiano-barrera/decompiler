/// Control Flow Graph.
///
/// Routines and data types to extract and represent the control-flow graph (basic blocks and their
/// sequence relationships).
use std::{
    collections::HashMap,
    ops::{Index, IndexMut, Range},
};

use crate::{cfg, mil};

/// Control Flow Graph
pub struct Graph {
    bounds: Vec<mil::Index>,
    // successors[bndx] = successors to block #bndx
    successors: BlockMap<BlockCont>,
    block_at: HashMap<mil::Index, BasicBlockID>,
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

    pub fn insns_ndx_range(&self, bid: BasicBlockID) -> Range<usize> {
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
        let mut bounds = Vec::with_capacity(program.len() / 5);
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
            let mut is = vec![false; program.len()];
            for (&start_ndx, &end_ndx) in bounds.iter().zip(bounds[1..].iter()) {
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
        block_at,
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
    pub fn dump_graphviz(&self, program: &mil::Program) {
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
        println!("}}");
    }
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
    rem_preds_count[cfg::ENTRY_BID] = 1;

    while let Some(bid) = queue.pop() {
        // each node X must be processed (added to the ordering) only after all its P predecessors
        // have been processed.  we achieve this by discarding X queue items corresponding to (P-1)
        // times before processing it the P-th time.

        rem_preds_count[bid] -= 1;
        if rem_preds_count[bid] > 0 {
            eprintln!(
                "node {}: delaying ({})",
                bid.as_usize(),
                rem_preds_count[bid]
            );
            continue;
        }

        assert_eq!(rem_preds_count[bid], 0);
        eprintln!("node {}: processing", bid.as_usize());
        order.push(bid);
        visited[bid] = true;

        let block_succs: [_; 2] = graph.successors[bid].as_array();
        for succ in block_succs.into_iter().flatten() {
            if !visited[succ] {
                eprintln!("{} -> {}", bid.as_usize(), succ.as_usize());
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
