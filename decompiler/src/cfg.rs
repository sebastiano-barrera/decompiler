/// Control Flow Graph.
///
/// Routines and data types to extract and represent the control-flow graph (basic blocks and their
/// sequence relationships).
use std::{
    collections::HashMap,
    ops::{Index, IndexMut, Range},
};

use arrayvec::ArrayVec;

use crate::{
    mil,
    pp::{self, PP},
    util,
};

/// A graph where nodes are blocks, and edges are successors/predecessors relationships.
#[derive(Clone)]
pub struct Graph {
    // successors[bndx] = successors to block #bndx
    direct: Edges,
    predecessors: BlockMultiMap<BlockID>,
    dom_tree: DomTree,
    reverse_postorder: Ordering,
}

#[derive(Clone)]
pub struct Edges {
    entry_bid: BlockID,
    successors: BlockMap<BlockCont>,
}

impl std::ops::Index<BlockID> for Edges {
    type Output = BlockCont;

    fn index(&self, bid: BlockID) -> &Self::Output {
        &self.successors[bid]
    }
}

impl Edges {
    fn assert_invariants(&self) {
        let block_count = self.successors.block_count();

        assert!(self.entry_bid.as_number() < block_count);
        for (_, succs) in self.successors.items() {
            for succ in succs.block_dests() {
                assert!(succ.as_number() < block_count);
            }
        }
    }

    /// Create an `Edges` structure, starting from a BlockMap that
    /// associates each block to its BlockCont.
    fn from_successors(successors: BlockMap<BlockCont>, entry_bid: BlockID) -> Self {
        let edges = Edges {
            entry_bid,
            successors,
        };
        edges.assert_invariants();
        edges
    }

    pub fn block_ids(&self) -> impl DoubleEndedIterator<Item = BlockID> {
        (0..self.block_count()).map(BlockID)
    }
    fn block_count(&self) -> u16 {
        self.successors.block_count()
    }

    pub fn successors(&self, bndx: BlockID) -> ArrayVec<BlockID, 2> {
        self.successors[bndx].block_dests()
    }

    pub fn entry_bid(&self) -> BlockID {
        self.entry_bid
    }

    pub fn dump<W: pp::PP>(&self, out: &mut W) -> std::io::Result<()> {
        writeln!(out, "digraph {{\n  ")?;
        out.open_box();

        for bid in self.block_ids() {
            let dests = self.successors(bid);
            for (succ_ndx, dest) in dests.iter().enumerate() {
                let color = match (dests.len(), succ_ndx) {
                    (2, 0) => "darkred",
                    (2, 1) => "darkgreen",
                    _ => "black",
                };
                writeln!(
                    out,
                    "block{} -> block{} [color=\"{}\"];",
                    bid.0, dest.0, color
                )?;
            }
        }

        out.close_box();
        writeln!(out, "\n}}\n")?;
        Ok(())
    }
}

impl std::fmt::Debug for Edges {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "digraph {{")?;
        for (bid, succs) in self.successors.items() {
            for succ in succs.block_dests() {
                writeln!(f, "  block{} -> block{};", bid.0, succ.0)?;
            }
        }
        writeln!(f, "}}")
    }
}

/// Block continuation destination.
///
/// Describes where control flow may continue to after running through a specific basic
/// block (i.e. the block's successors).
#[derive(Debug, Clone, Copy)]
pub enum BlockCont {
    /// Jump to the associated destination unconditionally.
    Always(Dest),
    /// Jump to either associated destination, conditionally (i.e. based on the
    /// value of the implicit condition register).
    Conditional { pos: Dest, neg: Dest },
}

impl Default for BlockCont {
    fn default() -> Self {
        BlockCont::Always(Dest::Undefined)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dest {
    /// A machine address, external to the program/function
    Ext(u64),
    /// A basic block within the program/function
    Block(BlockID),
    /// The destination is stored in a machine register (or SSA value), and
    /// therefore only known at runtime
    Indirect,
    /// Return to the calling function
    Return,
    /// No information about where the block is going to jump to (e.g.
    /// the decompiler couldn't figure it out due to an internal bug or a
    /// limitation).
    Undefined,
}

impl BlockCont {
    #[inline]
    pub fn block_dests(&self) -> ArrayVec<BlockID, 2> {
        self.dests()
            .into_iter()
            .filter_map(|dest| match dest {
                Dest::Block(bid) => Some(bid),
                _ => None,
            })
            .collect()
    }

    #[inline]
    pub fn dests(&self) -> ArrayVec<Dest, 2> {
        match self {
            BlockCont::Always(tgt) => [*tgt].into_iter().collect(),
            BlockCont::Conditional { pos, neg } => [*pos, *neg].into_iter().collect(),
        }
    }
}

#[derive(Clone)]
pub struct Schedule(BlockMap<Vec<mil::Index>>);

impl Schedule {
    /// Construct a BlockSpans from the an array representation of the block's spans.
    ///
    /// bounds[i] == start of block for BlockID(i) == end of block BlockID(i-1)
    fn from_bounds(bounds: &[mil::Index]) -> Self {
        let block_count = (bounds.len() - 1).try_into().unwrap();
        let mut bmap = BlockMap::new_sized(Vec::new(), block_count);

        for (bndx, (&start, &end)) in bounds.iter().zip(bounds[1..].iter()).enumerate() {
            assert!(end >= start);
            let bid = BlockID(bndx.try_into().unwrap());
            bmap[bid].extend(start..end);
        }

        assert_eq!(bmap.block_count(), block_count);

        let bs = Schedule(bmap);
        bs.assert_invariants();
        bs
    }

    pub fn block_count(&self) -> u16 {
        self.0.block_count()
    }

    pub fn of_block(&self, bid: BlockID) -> &[mil::Index] {
        &self.0[bid]
    }

    pub fn insert(&mut self, ndx: mil::Index, bid: BlockID, ndx_in_block: u16) {
        self.0[bid].insert(ndx_in_block as usize, ndx);
    }
    pub fn append(&mut self, ndx: mil::Index, bid: BlockID) {
        self.0[bid].push(ndx);
    }

    fn assert_invariants(&self) {
        // check: no duplicates
        let mut is_insn_scheduled = Vec::new();

        for (bid, ndxs) in self.0.items() {
            for &ndx in ndxs {
                let ndx = ndx as usize;
                if ndx >= is_insn_scheduled.len() {
                    is_insn_scheduled.resize(ndx + 1, false);
                }

                assert!(!is_insn_scheduled[ndx]);
                is_insn_scheduled[ndx] = true;
            }

            assert!(ndxs.len() > 0, "block is empty: {:?}", bid);
        }
    }
}

#[derive(PartialEq, Eq, Debug, Copy, Clone, Hash)]
pub struct BlockID(u16);

impl BlockID {
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
    pub fn block_count(&self) -> u16 {
        self.direct.block_count()
    }

    pub fn block_ids(&self) -> impl Iterator<Item = BlockID> {
        (0..self.block_count()).map(BlockID)
    }

    pub fn block_preds(&self, bid: BlockID) -> &[BlockID] {
        &self.predecessors[bid]
    }

    pub fn block_cont(&self, bid: BlockID) -> BlockCont {
        self.direct.successors[bid]
    }

    pub fn dom_tree(&self) -> &DomTree {
        &self.dom_tree
    }

    pub fn direct(&self) -> &Edges {
        &self.direct
    }

    pub fn entry_block_id(&self) -> BlockID {
        self.direct.entry_bid
    }

    /// Iterate through the IDs of the blocks in this graph, in reverse post order.
    ///
    /// In this ordering, entry blocks are yielded first; then each block is
    /// only yielded after all of its predecessors.
    pub fn block_ids_rpo(&self) -> impl '_ + DoubleEndedIterator<Item = BlockID> {
        self.reverse_postorder.block_ids().iter().copied()
    }

    /// Iterate through the IDs of the blocks in this graph, in post order.
    ///
    /// In this ordering, exit blocks are yielded first; then each block is
    /// only yielded after all of its children.
    pub fn block_ids_postorder(&self) -> impl '_ + DoubleEndedIterator<Item = BlockID> {
        self.block_ids_rpo().rev()
    }
}

pub fn analyze_mil(program: &mil::Program) -> (Graph, Schedule) {
    // if bounds == [b0, b1, b2, b3, ...]
    // then the basic blocks span instructions at indices [b0, b1), [b1, b2), [b2, b3), ...
    // in general, basic block at index bndx spans [bounds[bndx], bounds[bndx+1])
    let mut bounds = Vec::with_capacity(program.len() as usize / 5);
    bounds.push(0);

    for ndx in 0..program.len() {
        if let Some(iv) = program.get(ndx) {
            if let mil::Insn::Control(ctl) = iv.insn.get() {
                // the instruction jumps or otherwise affects control flow, and is a
                // "block ender", i.e. the last instruction in the block; equivalently,
                // ndx+1 must be a block's start.
                //
                // This is regardless of whether the jump (if any) happens
                // conditionally. For some weird programs, nothing ever jumps to ndx+1
                // even , but we don't care here.
                if let mil::Control::Jmp(target_ndx) | mil::Control::JmpIf(target_ndx) = ctl {
                    // we *also* have a block starting at target_ndx
                    bounds.push(target_ndx);
                }
                bounds.push(ndx + 1);
            }
        }
    }
    bounds.push(program.len());
    bounds.sort();
    bounds.dedup();

    // number of blocks
    //
    // this number is not going to change, even though we're going to remove the
    // unreachable blocks.
    let block_count: u16 = (bounds.len() - 1).try_into().unwrap();
    let block_at_map: HashMap<_, _> = bounds[..block_count as usize]
        .iter()
        .enumerate()
        .map(|(bndx, start_ndx)| {
            let bndx = bndx.try_into().unwrap();
            (*start_ndx, BlockID(bndx))
        })
        .collect();

    let dest_at = |ndx| {
        block_at_map
            .get(&ndx)
            .map(|&bid| Dest::Block(bid))
            // mostly happens when ndx == program.len()
            .unwrap_or(Dest::Undefined)
    };

    let mut direct = {
        let mut block_conts = BlockMap::new_sized(BlockCont::default(), block_count);

        for (blk_ndx, (&insn_ndx_start, &insn_ndx_end)) in
            bounds.iter().zip(&bounds[1..]).enumerate()
        {
            assert!(insn_ndx_end >= insn_ndx_start);
            let bid = BlockID(blk_ndx.try_into().unwrap());

            if insn_ndx_end > insn_ndx_start {
                // non-empty block
                let last_ndx = insn_ndx_end - 1;
                block_conts[bid] = match program.get(last_ndx).unwrap().insn.get() {
                    mil::Insn::Control(ctl) => match ctl {
                        mil::Control::Ret => BlockCont::Always(Dest::Return),
                        mil::Control::Jmp(tgt_ndx) => {
                            let dest_bid = *block_at_map.get(&tgt_ndx).unwrap();
                            BlockCont::Always(Dest::Block(dest_bid))
                        }
                        mil::Control::JmpIf(pos_ndx) => {
                            let pos = dest_at(pos_ndx);
                            let neg = dest_at(last_ndx + 1);
                            BlockCont::Conditional { pos, neg }
                        }
                        mil::Control::JmpExt(addr) => BlockCont::Always(Dest::Ext(addr)),
                        mil::Control::JmpExtIf(addr) => {
                            let pos = Dest::Ext(addr);
                            let neg = dest_at(last_ndx + 1);
                            BlockCont::Conditional { pos, neg }
                        }
                        mil::Control::JmpIndirect => BlockCont::Always(Dest::Indirect),
                    },

                    _ => BlockCont::Always(dest_at(last_ndx + 1)),
                };
            }
        }

        let entry_bid = BlockID(0);
        Edges::from_successors(block_conts, entry_bid)
    };

    let (order, is_reachable) = reverse_postorder(&direct);
    assert_eq!(is_reachable.block_count(), direct.block_count());

    // prevent "unreachable blocks" from being any other block's predecessor
    assert!(is_reachable[direct.entry_bid]);
    for (bid, &is_reachable_bid) in is_reachable.items() {
        if !is_reachable_bid {
            direct.successors[bid] = BlockCont::Always(Dest::Undefined);
        }
    }

    let reverse_postorder = Ordering::new(order, direct.block_count());

    let predecessors = compute_predecessors(&direct.successors);
    let dom_tree = compute_dom_tree(&direct, &reverse_postorder, &predecessors);

    let graph = Graph {
        direct,
        predecessors,
        dom_tree,
        reverse_postorder,
    };
    graph.assert_invariants();

    let block_spans = Schedule::from_bounds(&bounds);

    assert_eq!(graph.block_count(), block_spans.block_count());
    (graph, block_spans)
}

fn compute_predecessors(successors: &BlockMap<BlockCont>) -> BlockMultiMap<BlockID> {
    let mut builder = BlockMultiMapSorter::new(successors.block_count());
    for (pred, succs) in successors.items() {
        for succ in succs.block_dests() {
            builder.add(succ, pred);
        }
    }
    builder.build()
}

#[derive(Clone)]
struct BlockMultiMap<T> {
    ndx_range: BlockMap<Range<usize>>,
    items: Vec<T>,
}
impl<T> BlockMultiMap<T> {
    fn block_count(&self) -> u16 {
        self.ndx_range.block_count()
    }

    fn items(&self) -> impl Iterator<Item = (BlockID, &[T])> {
        self.ndx_range
            .items()
            .map(|(bid, ndx_range)| (bid, &self.items[ndx_range.clone()]))
    }

    fn assert_invariants(&self) {
        for (_, range) in self.ndx_range.items() {
            assert!(range.end <= self.items.len());
        }
    }
}
impl<T> std::ops::Index<BlockID> for BlockMultiMap<T> {
    type Output = [T];
    fn index(&self, bid: BlockID) -> &Self::Output {
        let range = self.ndx_range[bid].clone();
        &self.items[range]
    }
}

pub struct BlockMultiMapSorter<T> {
    block_count: u16,
    items: Vec<(BlockID, T)>,
}
impl<T> BlockMultiMapSorter<T> {
    fn new(block_count: u16) -> Self {
        BlockMultiMapSorter {
            items: Vec::new(),
            block_count,
        }
    }
    fn add(&mut self, bid: BlockID, value: T) {
        self.items.push((bid, value));
    }
    fn build(self) -> BlockMultiMap<T> {
        let mut items = self.items;
        items.sort_by_key(|(bid, _)| bid.0);

        let mut mm: BlockMultiMap<T> = BlockMultiMap {
            ndx_range: BlockMap::new_sized(0..0, self.block_count),
            items: Vec::with_capacity(items.len()),
        };

        // there is for sure a ready-made algorithm for doing this...
        let mut iter = items.into_iter().peekable();
        while let Some((cur_bid, value1)) = iter.next() {
            let ndx_start = mm.items.len();
            mm.items.push(value1);

            while let Some((bid, _)) = iter.peek() {
                if *bid != cur_bid {
                    break;
                }

                let (_, value) = iter.next().unwrap();
                mm.items.push(value);
            }

            let ndx_end = mm.items.len();
            mm.ndx_range[cur_bid] = ndx_start..ndx_end;
        }

        mm
    }
}

impl Graph {
    fn assert_invariants(&self) {
        self.direct.assert_invariants();
    }

    pub fn dump_graphviz<W: PP + ?Sized>(
        &self,
        out: &mut W,
        dom_tree: Option<&DomTree>,
    ) -> std::io::Result<()> {
        let count = self.block_count();
        writeln!(out, "digraph {{")?;
        writeln!(out, "  // {} blocks", count)?;

        for bid in self.block_ids() {
            writeln!(out, "  block{};", bid.0)?;

            match self.direct.successors(bid).as_slice() {
                [] => writeln!(out, "  block{} -> end", bid.0)?,
                dests => {
                    for (succ_ndx, dest) in dests.iter().enumerate() {
                        let color = match (dests.len(), succ_ndx) {
                            (1, _) => "black",
                            (2, 0) => "darkred",
                            (2, 1) => "darkgreen",
                            _ => panic!("max 2 successors!"),
                        };
                        writeln!(
                            out,
                            "  block{} -> block{} [color=\"{}\"];",
                            bid.0, dest.0, color
                        )?;
                    }
                }
            }

            writeln!(out,)?;
        }

        if let Some(dom_tree) = dom_tree {
            writeln!(out, "  // dominator tree")?;
            for (bid, parent) in dom_tree.items() {
                if let Some(parent) = parent {
                    writeln!(
                        out,
                        "  block{} -> block{} [style=\"dotted\"];",
                        bid.as_number(),
                        parent.as_number()
                    )?;
                }
            }
        }

        writeln!(out, "}}")?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DomTree {
    parent: BlockMap<Option<BlockID>>,
    children_ndx_range: BlockMap<Range<usize>>,
    children: Vec<BlockID>,
}

impl DomTree {
    fn from_parent(parent: BlockMap<Option<BlockID>>) -> DomTree {
        let count = parent.block_count();

        let mut children_ndx_range = BlockMap::new_sized(0..0, count);
        let mut children = Vec::with_capacity(count as usize);

        for bid in parent.block_ids() {
            let clen_before = children.len();
            for (child_bid, parent) in parent.items() {
                if parent == &Some(bid) {
                    children.push(child_bid);
                }
            }
            children_ndx_range[bid] = clen_before..children.len();
        }

        // entry nodes are nobody's child in the dom tree
        let entries_count = parent.items().filter(|(_, p)| p.is_none()).count();
        assert_eq!(children.len(), count as usize - entries_count);
        DomTree {
            parent,
            children_ndx_range,
            children,
        }
    }

    pub fn items(&self) -> impl ExactSizeIterator<Item = (BlockID, &Option<BlockID>)> {
        self.parent.items()
    }

    /// Get an iterator of immediate dominators of the given block
    pub fn imm_doms(&self, bid: BlockID) -> impl '_ + Iterator<Item = BlockID> {
        let mut cur = self.parent[bid];
        let mut visited = BlockMap::new_sized(false, self.parent.block_count());
        std::iter::from_fn(move || {
            let ret = cur?;
            cur = self.parent[ret];

            assert!(!visited[ret]);
            visited[ret] = true;

            Some(ret)
        })
    }

    pub fn children_of(&self, bid: BlockID) -> &[BlockID] {
        let ndx_range = self.children_ndx_range[bid].clone();
        &self.children[ndx_range]
    }

    pub fn parent_of(&self, bid: BlockID) -> Option<BlockID> {
        self[bid]
    }

    pub fn dump<W: std::io::Write + ?Sized>(&self, out: &mut W) -> std::io::Result<()> {
        for (bid, parent) in self.parent.items() {
            if parent.is_none() {
                self.dump_subtree(out, bid, 0)?
            }
        }
        Ok(())
    }
    fn dump_subtree<W: std::io::Write + ?Sized>(
        &self,
        out: &mut W,
        bid: BlockID,
        depth: usize,
    ) -> std::io::Result<()> {
        for _ in 0..depth {
            write!(out, "|  ")?;
        }
        writeln!(out, "{:?}", bid)?;

        for &child_bid in self.children_of(bid) {
            self.dump_subtree(out, child_bid, depth + 1)?;
        }

        Ok(())
    }
}

impl Index<BlockID> for DomTree {
    type Output = Option<BlockID>;

    fn index(&self, index: BlockID) -> &Self::Output {
        self.parent.index(index)
    }
}

fn compute_dom_tree(
    fwd_edges: &Edges,
    rpo: &Ordering,
    predecessors: &BlockMultiMap<BlockID>,
) -> DomTree {
    let block_count = fwd_edges.block_count();
    assert_eq!(block_count, predecessors.block_count());

    let mut parent = BlockMap::new_sized(None, block_count);

    // process the entry node(s) "manually", so the algorithm can rely on it for successors
    // TODO remove this. should no longer be useful with our `common_ancestor`
    // that includes the `changed` flag
    parent[fwd_edges.entry_bid] = Some(fwd_edges.entry_bid);

    let mut changed = true;
    while changed {
        changed = false;

        for &bid in rpo.block_ids().iter() {
            let preds = &predecessors[bid];
            if preds.is_empty() {
                continue;
            }

            #[derive(PartialEq, Eq)]
            enum RefPred {
                Uninit,
                Block(BlockID),
                NoCommonAncestor,
            }
            let mut ref_pred = RefPred::Uninit;

            // compare all predecessor against one (arbitrary) predecessor that
            // was already processed (parent already assigned in dominator
            // tree), herein "reference predecessor" (ref. pred.)
            for &pred in preds.iter() {
                if parent[pred].is_none() {
                    continue;
                }

                // pred was already processed;
                // ref_pred <- common ancestor of ref_pred and pred
                ref_pred = match ref_pred {
                    RefPred::Uninit => RefPred::Block(pred),
                    RefPred::Block(ref_pred) => {
                        let found = common_ancestor(
                            &parent,
                            |id_a, id_b| rpo.position_of(id_a) < rpo.position_of(id_b),
                            pred,
                            ref_pred,
                        );
                        match found {
                            Some(com_anc) => RefPred::Block(com_anc),
                            None => RefPred::NoCommonAncestor,
                        }
                    }
                    RefPred::NoCommonAncestor => RefPred::NoCommonAncestor,
                };
            }

            match ref_pred {
                RefPred::Uninit => {
                    // the basic guarantee of reverse postorder: at least one
                    // predecessor has already been processed (the ones that get
                    // to `bid` via a backedge come later in the order)
                    assert!(preds.is_empty());
                }
                RefPred::Block(idom) => {
                    let prev_idom = parent[bid].replace(idom);
                    if prev_idom != Some(idom) {
                        changed = true;
                    }
                }
                RefPred::NoCommonAncestor => {
                    // TODO Add a comment explaining why this is right
                    assert!(parent[bid].is_none());
                    // No need to ever do: parent[bid] = None;
                }
            };
        }
    }

    // we hand the tree out with a slightly different convention: the entry node(s) has no parent in
    // the tree, so the corresponding item is None.  up to this point the root is linked to itself,
    // as required by the algorithm by how it's formulated
    let entry = fwd_edges.entry_bid;
    assert_eq!(parent[entry], Some(entry));
    parent[entry] = None;

    for (bid, parent) in parent.items() {
        assert_ne!(*parent, Some(bid));
    }
    DomTree::from_parent(parent)
}

/// Find the common ancestor of two nodes in a tree.
///
/// The tree is presumed to have progressively numbered nodes. It is represented as an array
/// `parent_of` such that, for each node with index _i_, parent_of[i] is the index of the parent
/// node (or _i_, the same index, for the root node).
fn common_ancestor<LT>(
    parent_of: &BlockMap<Option<BlockID>>,
    is_lt: LT,
    ndx_a: BlockID,
    ndx_b: BlockID,
) -> Option<BlockID>
where
    LT: Fn(BlockID, BlockID) -> bool,
{
    struct AdHocTree<'a, LT> {
        parent_of: &'a BlockMap<Option<BlockID>>,
        is_lt: LT,
    }
    impl<'a, LT> util::NumberedTree for AdHocTree<'a, LT>
    where
        LT: Fn(BlockID, BlockID) -> bool,
    {
        type Key = BlockID;

        fn parent_of(&self, k: &BlockID) -> Option<BlockID> {
            self.parent_of[*k]
        }

        fn key_lt(&self, a: BlockID, b: BlockID) -> bool {
            (self.is_lt)(a, b)
        }
    }

    util::common_ancestor(&AdHocTree { parent_of, is_lt }, ndx_a, ndx_b)
}

/// Visit the blocks in the given graph in reverse postorder, and a return the
/// resulting ordering.
///
/// The returned ordering will only contain the blocks that are reachable from
/// the entry block.
fn reverse_postorder(edges: &Edges) -> (Vec<BlockID>, BlockMap<bool>) {
    let count = edges.block_count();

    // Remaining predecessors count
    let mut rem_preds_count = count_nonbackedge_predecessors(edges);

    let mut order = Vec::with_capacity(count as usize);
    let mut queue = Vec::with_capacity(count as usize / 2);
    // we must avoid re-visiting a fully processed node. this happens when processing backedges.
    let mut visited = BlockMap::new_sized(false, count);

    queue.push(edges.entry_bid);
    // formally incorrect, but aligns it with the rest of the algorithm
    rem_preds_count[edges.entry_bid] = 1;

    while let Some(bid) = queue.pop() {
        // each node X must be processed (added to the ordering) only after
        // all its P predecessors have been processed.  we achieve this by
        // discarding X's queue items  (P-1) times corresponding to (P-1)
        // predecessors before finally allowing processing it the P-th time.

        rem_preds_count[bid] -= 1;
        if rem_preds_count[bid] > 0 {
            continue;
        }

        assert_eq!(rem_preds_count[bid], 0);
        order.push(bid);
        visited[bid] = true;

        for succ in edges.successors(bid) {
            if !visited[succ] {
                queue.push(succ);
            }
        }
    }

    // all incoming edges have been processed
    assert!(rem_preds_count.items().all(|(_, &count)| count == 0));
    // it may be that count > order.len(), if any blocks are unrechable from the
    // entry.
    (order, visited)
}

/// Count, for each node, the number of incoming edges (or, equivalently, predecessor nodes) that
/// are not back-edges (i.e. don't form a cycle).
///
/// Returns a tuple of two BlockMaps, associating each block to:
///  - the number of non-backedge predecessors (main purpose of this function)
///  - whether it's is reachable from the entry block (as a boolean)
fn count_nonbackedge_predecessors(edges: &Edges) -> BlockMap<u16> {
    let count = edges.block_count();

    let mut incoming_count = BlockMap::new_sized(0, count);
    let mut in_path = BlockMap::new_sized(false, count);
    let mut finished = BlockMap::new_sized(false, count);

    enum Cmd {
        Enter(BlockID),
        Exit(BlockID),
    }
    let mut queue = Vec::with_capacity(count as usize / 2);
    queue.push(Cmd::Exit(edges.entry_bid));
    queue.push(Cmd::Enter(edges.entry_bid));

    while let Some(cmd) = queue.pop() {
        match cmd {
            Cmd::Enter(bid) => {
                assert!(!in_path[bid]);
                in_path[bid] = true;

                if finished[bid] {
                    // was marked finished in the meantime!
                    // keep in the path for a later Cmd::Exit
                    continue;
                }

                for succ in edges.successors(bid) {
                    if !in_path[succ] {
                        incoming_count[succ] += 1;
                        queue.push(Cmd::Exit(succ));
                        queue.push(Cmd::Enter(succ));
                    } else {
                        // it's a backedge, don't count it
                    }
                }
            }
            Cmd::Exit(bid) => {
                assert!(in_path[bid]);
                in_path[bid] = false;
                // may be false or already true; the latter if the node is
                // visited multiple times
                finished[bid] = true;
            }
        }
    }

    incoming_count
}

#[derive(Clone)]
pub struct Ordering {
    order: Vec<BlockID>,
    pos_of: BlockMap<usize>,
}

impl Ordering {
    /// Create a new Ordering representing the same as `order`.
    ///
    /// `total_count` is the total number of blocks existing in the graph,
    /// whether or not they are included in the ordering. This number equals 1 +
    /// the maximum numeric value (.as_number()) of the existing BlockIDs.
    ///
    /// Panics if any BlockID appears more than once in `order`.
    pub fn new(order: Vec<BlockID>, total_count: u16) -> Self {
        let mut pos_of = BlockMap::new_sized(0, total_count);
        let mut occurred = BlockMap::new_sized(false, total_count);
        for (pos, &bid) in order.iter().enumerate() {
            assert!(!occurred[bid]);
            occurred[bid] = true;
            pos_of[bid] = pos;
        }

        Ordering { order, pos_of }
    }

    pub fn block_ids(&self) -> &[BlockID] {
        &self.order
    }

    pub fn position_of(&self, bid: BlockID) -> usize {
        self.pos_of[bid]
    }
}

//
// Utilities
//
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockMap<T>(Vec<T>);

impl<T: Clone> BlockMap<T> {
    pub fn new_sized(init: T, count: u16) -> Self {
        let vec = vec![init; count as usize];
        BlockMap(vec)
    }
    pub fn new(cfg: &Graph, init: T) -> Self {
        Self::new_sized(init, cfg.block_count())
    }

    pub fn new_with<F>(cfg: &Graph, init_item: F) -> Self
    where
        F: Fn(BlockID) -> T,
    {
        Self(cfg.block_ids().map(init_item).collect())
    }

    pub fn block_count(&self) -> u16 {
        self.0.len().try_into().unwrap()
    }
}

impl<T> BlockMap<T> {
    pub fn block_ids(&self) -> impl ExactSizeIterator<Item = BlockID> {
        (0..self.0.len()).map(|ndx| BlockID(ndx.try_into().unwrap()))
    }

    pub fn items(&self) -> impl ExactSizeIterator<Item = (BlockID, &T)> {
        self.0.iter().enumerate().map(|(ndx, item)| {
            let ndx = ndx.try_into().unwrap();
            (BlockID(ndx), item)
        })
    }
    pub fn items_mut(&mut self) -> impl ExactSizeIterator<Item = (BlockID, &mut T)> {
        self.0.iter_mut().enumerate().map(|(ndx, item)| {
            let ndx = ndx.try_into().unwrap();
            (BlockID(ndx), item)
        })
    }
}

impl<T> From<BlockMap<T>> for Vec<T> {
    fn from(val: BlockMap<T>) -> Self {
        val.0
    }
}

impl<T> Index<BlockID> for BlockMap<T> {
    type Output = T;

    fn index(&self, index: BlockID) -> &Self::Output {
        self.0.index(index.0 as usize)
    }
}
impl<T> IndexMut<BlockID> for BlockMap<T> {
    fn index_mut(&mut self, index: BlockID) -> &mut Self::Output {
        self.0.index_mut(index.0 as usize)
    }
}
