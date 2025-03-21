/// Control Flow Graph.
///
/// Routines and data types to extract and represent the control-flow graph (basic blocks and their
/// sequence relationships).
use std::{
    collections::HashMap,
    ops::{Index, IndexMut, Range},
};

use crate::{mil, pp::PP};

/// A graph where nodes are blocks, and edges are successors/predecessors relationships.
#[derive(Clone)]
pub struct Graph {
    bounds: Vec<mil::Index>,
    // successors[bndx] = successors to block #bndx
    direct: Edges,
    inverse: Edges,

    dom_tree: DomTree,
    inv_dom_tree: DomTree,

    reverse_postorder: Ordering,
}

#[derive(Clone)]
pub struct Edges {
    entry_bid: BlockID,
    exit_bid: BlockID,
    succ_ndxr: BlockMap<Range<usize>>,
    // these probably should be Box<[T]>
    succ: Vec<BlockID>,
    // corresponds to 'succs'. each item is the "predecessor index" of the edge
    // a->b, such that if the value is p, then a is the pth predecessor of b.
    succ_pred_ndx: Vec<u8>,
    nonbackedge_preds_count: BlockMap<u16>,
}

impl std::ops::Index<BlockID> for Edges {
    type Output = [BlockID];

    fn index(&self, bid: BlockID) -> &Self::Output {
        let range = self.succ_ndxr[bid].clone();
        &self.succ[range]
    }
}

impl Edges {
    fn assert_invariants(&self) {
        assert_eq!(self.succ_pred_ndx.len(), self.succ.len());
        assert!(self.entry_bid.as_number() < self.succ_ndxr.block_count());
        assert_eq!(
            self.succ_ndxr.block_count(),
            self.nonbackedge_preds_count.block_count()
        );
        assert!(self.successors(self.exit_bid).len() == 0);
    }

    pub fn block_ids(&self) -> impl DoubleEndedIterator<Item = BlockID> {
        (0..self.block_count()).map(BlockID)
    }
    fn block_count(&self) -> u16 {
        self.succ_ndxr.block_count()
    }

    pub fn nonbackedge_predecessor_count(&self, bid: BlockID) -> u16 {
        self.nonbackedge_preds_count[bid]
    }

    pub fn successors(&self, bndx: BlockID) -> &[BlockID] {
        let range = &self.succ_ndxr[bndx];
        &self.succ[range.start..range.end]
    }
}

impl std::fmt::Debug for Edges {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "digraph {{")?;
        for (bid, _) in self.succ_ndxr.items() {
            for succ in &self[bid] {
                writeln!(f, "  block{} -> block{};", bid.0, succ.0)?;
            }
        }
        writeln!(f, "}}")
    }
}

type Jump = (PredIndex, BlockID);
type PredIndex = u8;

#[derive(Debug)]
pub enum BlockCont {
    End,
    Jmp(Jump),
    Alt { straight: Jump, side: Jump },
}

impl BlockCont {
    #[inline]
    pub fn as_array(&self) -> [Option<Jump>; 2] {
        match self {
            BlockCont::End => [None, None],
            BlockCont::Jmp(d) => [Some(*d), None],
            BlockCont::Alt { straight, side } => [Some(*straight), Some(*side)],
        }
    }

    #[inline]
    pub fn as_array_mut(&mut self) -> [Option<&mut Jump>; 2] {
        match self {
            BlockCont::End => [None, None],
            BlockCont::Jmp(d) => [Some(d), None],
            BlockCont::Alt { straight, side } => [Some(straight), Some(side)],
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
        (self.bounds.len() - 1).try_into().unwrap()
    }

    pub fn block_ids(&self) -> impl Iterator<Item = BlockID> {
        (0..self.block_count()).map(|ndx| BlockID(ndx.try_into().unwrap()))
    }

    pub fn block_preds(&self, bid: BlockID) -> &[BlockID] {
        self.inverse.successors(bid)
    }

    pub fn block_cont(&self, bid: BlockID) -> BlockCont {
        let range = &self.direct.succ_ndxr[bid];
        let successors = &self.direct.succ[range.clone()];
        let pred_ndxs = &self.direct.succ_pred_ndx[range.clone()];

        match successors {
            [] => BlockCont::End,
            [cons] => BlockCont::Jmp((pred_ndxs[0], *cons)),
            [alt, cons] => BlockCont::Alt {
                straight: (pred_ndxs[0], *alt),
                side: (pred_ndxs[1], *cons),
            },
            _ => panic!("blocks must have 2 successors max"),
        }
    }

    /// Get the range of instructions covered by this basic block.
    ///
    /// The range for any given basic block is guaranteed not to intersect with
    /// the range of any other basic block.
    ///
    /// NOTE: the returned indices ONLY make sense in the context of the MIL
    /// program that this CFG was originally built on. It has NO meaning in any
    /// other program and most importantly on SSA, where instructions are wired
    /// in a graph and not collected in compact sequences.
    pub fn insns_ndx_range(&self, bid: BlockID) -> Range<mil::Index> {
        let ndx = bid.as_usize();
        let start = self.bounds[ndx];
        let end = self.bounds[ndx + 1];
        // must contain at least 1 instruction
        // TODO hoist this assertion somewhere else?
        assert!(end > start);
        start..end
    }

    pub fn dom_tree(&self) -> &DomTree {
        &self.dom_tree
    }

    pub fn inv_dom_tree(&self) -> &DomTree {
        &self.inv_dom_tree
    }

    pub fn direct(&self) -> &Edges {
        &self.direct
    }
    pub fn inverse(&self) -> &Edges {
        &self.inverse
    }

    pub fn entry_block_id(&self) -> BlockID {
        self.direct.entry_bid
    }

    /// Iterate through the IDs of the blocks in this graph, in reverse post order.
    ///
    /// In this ordering, entry blocks are yielded first; then each block is
    /// only yielded after all of its predecessors.
    pub fn block_ids_rpo<'s>(&'s self) -> impl 's + DoubleEndedIterator<Item = BlockID> {
        self.reverse_postorder.block_ids().iter().copied()
    }

    /// Iterate through the IDs of the blocks in this graph, in post order.
    ///
    /// In this ordering, exit blocks are yielded first; then each block is
    /// only yielded after all of its children.
    pub fn block_ids_postorder<'s>(&'s self) -> impl 's + DoubleEndedIterator<Item = BlockID> {
        self.block_ids_rpo().rev()
    }
}

pub fn analyze_mil(program: &mil::Program) -> Graph {
    // graphs with multiple exit nodes are converted into graphs with exactly 1
    // exit node. this algorithm makes sure to add exactly 1 virtual exit node,
    // and to link the "real" exits from the program to these.
    //
    // we already assume that the program can only start "at the beginning", so
    // there IS always a single entry node.
    //
    // many algorithms assume one single entry node per graph, including the
    // inverse graph. This yield significantly simpler algorithms (and easier to
    // get right, too).

    // if bounds == [b0, b1, b2, b3, ...]
    // then the basic blocks span instructions at indices [b0, b1), [b1, b2), [b2, b3), ...
    // in particular, basic block at index bndx spans [bounds[bndx], bounds[bndx+1])
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

        // virtual exit node has bounds coinciding with the program's exit
        bounds.push(program.len());
        bounds.push(program.len());
        bounds.sort();
        bounds.dedup();
        bounds
    };

    // "real" because it does not include the 2 virtual blocks
    let block_count: u16 = (bounds.len() - 1).try_into().unwrap();

    let block_at: HashMap<_, _> = bounds[..block_count as usize]
        .iter()
        .enumerate()
        .map(|(bndx, start_ndx)| {
            let bndx = bndx.try_into().unwrap();
            (*start_ndx, BlockID(bndx))
        })
        .collect();

    let direct = {
        let mut target = Vec::new();
        let mut ndx_range = BlockMap::new(0..0, block_count);
        // the running counter of predecessors for each block, used to compute
        // the pred ndx
        let mut pred_count = BlockMap::new(0, block_count);
        let mut pred_ndx = Vec::new();

        let exit_bid = BlockID(block_count - 1);

        for (insn_end_ndx, blk_ndx) in bounds[1..].iter().zip(0..) {
            let ndx_range_start = target.len();

            debug_assert_eq!(*insn_end_ndx, bounds[blk_ndx + 1]);

            if *insn_end_ndx - bounds[blk_ndx] != 0 {
                // non-empty block

                let last_ndx = insn_end_ndx - 1;
                // predecessor indices are fixed up later
                match dest_of_insn(program, last_ndx) {
                    (None, None) => panic!("all instructions must lead *somewhere*!"),

                    (Some(dest), None) | (None, Some(dest)) => {
                        let dest = if dest == program.len() {
                            // exit node
                            exit_bid
                        } else {
                            *block_at.get(&dest).unwrap()
                        };
                        target.push(dest);
                        pred_ndx.push(pred_count[dest]);
                        pred_count[dest] += 1;
                    }

                    (Some(straight_dest), Some(side_dest)) => {
                        for dest in [straight_dest, side_dest] {
                            let dest = *block_at.get(&dest).unwrap();
                            target.push(dest);
                            pred_ndx.push(pred_count[dest]);
                            pred_count[dest] += 1;
                        }
                    }
                };
            }

            let bid = BlockID(blk_ndx.try_into().unwrap());
            ndx_range[bid] = ndx_range_start..target.len();
        }

        ndx_range[exit_bid] = target.len()..target.len();

        let mut edges = Edges {
            entry_bid: BlockID(0),
            exit_bid,
            succ_ndxr: ndx_range,
            succ_pred_ndx: pred_ndx,
            succ: target,
            nonbackedge_preds_count: BlockMap::new(0, block_count),
        };
        edges.assert_invariants();
        recount_nonbackedge_predecessors(&mut edges);
        edges
    };

    let inverse = {
        let mut ndx_range = BlockMap::new(0..0, block_count);
        let mut target = Vec::with_capacity(block_count as usize * 2);
        let mut pred_ndx = Vec::with_capacity(direct.succ.len());

        // quadratic, but you know how life goes
        for succ in direct.block_ids() {
            let ndx_range_start = target.len();

            for pred in direct.block_ids() {
                for pred_succ in direct.successors(pred) {
                    if *pred_succ == succ {
                        pred_ndx.push((target.len() - ndx_range_start).try_into().unwrap());
                        target.push(pred);
                    }
                }
            }

            ndx_range[succ] = ndx_range_start..target.len();
        }

        let mut edges = Edges {
            entry_bid: direct.exit_bid,
            exit_bid: direct.entry_bid,
            succ_ndxr: ndx_range,
            succ_pred_ndx: pred_ndx,
            succ: target,
            nonbackedge_preds_count: BlockMap::new(0, block_count),
        };
        edges.assert_invariants();
        recount_nonbackedge_predecessors(&mut edges);
        edges
    };

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

    let dom_tree = compute_dom_tree(&direct, &inverse);
    let inv_dom_tree = compute_dom_tree(&inverse, &direct);
    let reverse_postorder = Ordering::new(reverse_postorder(&direct));

    Graph {
        bounds,
        direct,
        inverse,
        dom_tree,
        inv_dom_tree,
        reverse_postorder,
    }
}

/// Returns the indices of the instruction(s) that may follow the given one.
///
/// The return value is a tuple of two elements:
///
///  - the first is the "straight" destination, the index to follow when the
///    branch (if any) IS NOT taken;
///
///  - the second is only set for branches and is the "side" destination, the
///    index to follow when the branch IS taken.
fn dest_of_insn(
    program: &mil::Program,
    ndx: mil::Index,
) -> (Option<mil::Index>, Option<mil::Index>) {
    let insn = program.get(ndx).unwrap().insn.get();
    match insn {
        mil::Insn::Jmp(ndx) => (None, Some(ndx)),
        mil::Insn::JmpIf { target, .. } => (Some(ndx + 1), Some(target)),
        // indirect jumps (Insn::JmpI, x86_64: jmp [reg]) are treated like
        // "jumps to somewhere else".  They potentially exit the function, or
        // re-enter it at some unknown location; we can only know at runtime.
        // With the little information we have, we can only treat it as "exit".
        mil::Insn::Ret(_) | mil::Insn::JmpInd(_) => {
            // one-past-the-end of the program is a valid index; signifies "exit the function"
            (None, Some(program.len()))
        }
        // external are currently handled as non-control-flow instruction
        // (mil::Insn::JmpExt(_) | mil::Insn::JmpExtIf { .. })
        _ => (Some(ndx + 1), None),
    }
}

impl Graph {
    pub fn dump_graphviz<W: PP + ?Sized>(
        &self,
        out: &mut W,
        dom_tree: Option<&DomTree>,
    ) -> std::io::Result<()> {
        let count = self.block_count();
        writeln!(out, "digraph {{")?;
        writeln!(out, "  // {} blocks", count)?;

        for bid in self.block_ids() {
            let Range { start, end } = self.insns_ndx_range(bid);

            writeln!(
                out,
                "  block{} [label=\"{}\\n{}..{}\"];",
                bid.0, bid.0, start, end,
            )?;

            match self.direct.successors(bid) {
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

        let mut children_ndx_range = BlockMap::new(0..0, count);
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
    pub fn imm_doms<'s>(&'s self, bid: BlockID) -> impl 's + Iterator<Item = BlockID> {
        let mut cur = self.parent[bid];
        let mut visited = BlockMap::new(false, self.parent.block_count());
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
}

impl Index<BlockID> for DomTree {
    type Output = Option<BlockID>;

    fn index(&self, index: BlockID) -> &Self::Output {
        self.parent.index(index)
    }
}

pub fn compute_dom_tree(fwd_edges: &Edges, bwd_edges: &Edges) -> DomTree {
    let block_count = fwd_edges.block_count();
    assert_eq!(block_count, bwd_edges.block_count());
    let rpo = Ordering::new(reverse_postorder(fwd_edges));

    let mut parent = BlockMap::new(None, block_count);

    // process the entry node(s) "manually", so the algorithm can rely on it for successors
    // TODO remove this. should no longer be useful with our `common_ancestor`
    // that includes the `changed` flag
    parent[fwd_edges.entry_bid] = Some(fwd_edges.entry_bid);

    let mut changed = true;
    while changed {
        changed = false;

        for &bid in rpo.block_ids().iter() {
            let preds = &bwd_edges[bid];
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
    mut ndx_a: BlockID,
    mut ndx_b: BlockID,
) -> Option<BlockID>
where
    LT: Fn(BlockID, BlockID) -> bool,
{
    // This func is called with a graph encoded with such that each entry point
    // `e` will have parent_of[e] == Some(e). If there are multiple entry
    // points, we will never have ndx_a == ndx_b. Rather, we would walk through
    // parent_of infinitely, so we have to add specific termination conditions
    // for those.
    while ndx_a != ndx_b {
        while parent_of[ndx_b] != Some(ndx_b) && is_lt(ndx_a, ndx_b) {
            ndx_b = parent_of[ndx_b].unwrap();
        }
        while parent_of[ndx_a] != Some(ndx_a) && is_lt(ndx_b, ndx_a) {
            ndx_a = parent_of[ndx_a].unwrap();
        }

        // When the graph has multiple entry nodes, each `e` of them has
        // parent_of[e] == Some(e). If ndx_a and ndx_b represent two such nodes,
        // we're never going to find a common ancestor, and we must stop (or we
        // go into an infinite loop).
        if ndx_a != ndx_b && parent_of[ndx_a] == Some(ndx_a) && parent_of[ndx_b] == Some(ndx_b) {
            return None;
        }
    }

    Some(ndx_a)
}

fn reverse_postorder(edges: &Edges) -> Vec<BlockID> {
    let count = edges.block_count();

    // Remaining predecessors count
    let mut rem_preds_count = edges.nonbackedge_preds_count.clone();

    let mut order = Vec::with_capacity(count as usize);
    let mut queue = Vec::with_capacity(count as usize / 2);
    // we must avoid re-visiting a fully processed node. this happens when processing backedges.
    let mut visited = BlockMap::new(false, count);

    queue.push(edges.entry_bid);
    // formally incorrect, but aligns it with the rest of the algorithm
    rem_preds_count[edges.entry_bid] = 1;

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

        for &succ in edges.successors(bid) {
            if !visited[succ] {
                queue.push(succ);
            }
        }
    }

    // all incoming edges have been processed
    if let Some(max) = rem_preds_count.items().map(|(_, count)| *count).max() {
        assert_eq!(0, max);
    }
    assert_eq!(order.len(), count as usize);
    order
}

/// Count, for each node, the number of incoming edges (or, equivalently, predecessor nodes) that
/// are not back-edges (i.e. don't form a cycle).
fn recount_nonbackedge_predecessors(edges: &mut Edges) {
    let count = edges.block_count();

    let mut incoming_count = BlockMap::new(0, count);
    let mut in_path = BlockMap::new(false, count);
    let mut finished = BlockMap::new(false, count);

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

                for &succ in edges.successors(bid) {
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

    edges.nonbackedge_preds_count = incoming_count;
}

#[derive(Clone)]
pub struct Ordering {
    order: Vec<BlockID>,
    pos_of: BlockMap<usize>,
}

impl Ordering {
    pub fn new(order: Vec<BlockID>) -> Self {
        let count = order.len().try_into().unwrap();
        let mut pos_of = BlockMap::new(0, count);
        let mut occurs_count = BlockMap::new(0, count);
        for (pos, &bid) in order.iter().enumerate() {
            occurs_count[bid] += 1;
            pos_of[bid] = pos;
        }

        assert!(occurs_count.items().all(|(_, count)| *count == 1));

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
    pub fn new(init: T, count: u16) -> Self {
        let vec = vec![init; count as usize];
        BlockMap(vec)
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
