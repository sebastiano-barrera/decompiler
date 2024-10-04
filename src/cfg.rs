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
pub struct Graph {
    bounds: Vec<mil::Index>,
    // successors[bndx] = successors to block #bndx
    successors: Vec<BlockCont>,
    block_at: HashMap<mil::Index, BasicBlockID>,
    predecessors: Vec<BasicBlockID>,
    pred_ndx_range: Vec<Range<usize>>,

    postorder_index: BlockMap<usize>,
    postorder_ordering: Vec<BasicBlockID>,
}

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

    pub fn predecessors(&self, bndx: BasicBlockID) -> Option<&[BasicBlockID]> {
        let range = self.pred_ndx_range.get(bndx.0 as usize)?;
        Some(&self.predecessors[range.start..range.end])
    }

    // TODO Change return value to BlockIndex
    pub fn postorder_ordering<'s>(&'s self) -> &[BasicBlockID] {
        &self.postorder_ordering
    }

    // TODO Change return value to BlockIndex
    pub fn postorder_index(&self) -> &BlockMap<usize> {
        &self.postorder_index
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

    let postorder_ordering = {
        let mut postorder_order = Vec::with_capacity(block_count);
        let mut visited = vec![false; block_count].into_boxed_slice();

        let mut queue = Vec::with_capacity(block_count);
        queue.push(0);

        while let Some(ndx) = queue.pop() {
            debug_assert!(!visited[ndx as usize]);
            postorder_order.push(BasicBlockID(ndx.try_into().unwrap()));

            let block_succs: [_; 2] = successors.get(ndx as usize).unwrap().as_array();
            // insert into the queue in reverse order
            for succ in block_succs.into_iter().rev() {
                if let Some(BasicBlockID(succ)) = succ {
                    let succ = succ as usize;
                    if !visited[succ] {
                        queue.push(succ);
                    }
                }
            }

            visited[ndx] = true;
        }

        postorder_order
    };

    let postorder_index = {
        let mut postorder_index = BlockMap(vec![0; block_count]);
        for (order_ndx, &bndx) in postorder_ordering.iter().enumerate() {
            let order_ndx = order_ndx.try_into().unwrap();
            postorder_index[bndx] = order_ndx;
        }
        postorder_index
    };

    #[cfg(debug_assertions)]
    {
        let is_covered = {
            let mut is = vec![false; program.len()];
            for (&start_ndx, &end_ndx) in bounds.iter().zip(bounds[1..].iter()) {
                for i in start_ndx..end_ndx {
                    is[i] = true;
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
            .filter(|&i| !(i <= program.len()))
            .collect();
        debug_assert_eq!(&invalid, &[]);

        // no duplicates (<=> no 0-length blocks)
        debug_assert!(bounds.iter().zip(bounds[1..].iter()).all(|(a, b)| a != b));

        for (i, ndx) in postorder_ordering.iter().enumerate() {
            debug_assert_eq!(postorder_index[*ndx], i);
        }
        for (bid, order) in postorder_index.items() {
            debug_assert_eq!(postorder_ordering[*order], bid);
        }
    }

    Graph {
        bounds,
        block_at,
        successors,
        predecessors,
        pred_ndx_range,
        postorder_ordering,
        postorder_index,
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

            match self.successors[bndx] {
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
