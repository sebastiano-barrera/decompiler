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
    original: mil::Program,
}

impl Program {
    pub fn original(&self) -> &mil::Program {
        &self.original
    }
}

pub fn convert_to_ssa(program: &mil::Program, cfg: &cfg::Graph) -> Program {
    let phi_blocks = place_phi_nodes(program, cfg);
    todo!()
}

fn place_phi_nodes(program: &mil::Program, cfg: &cfg::Graph) {
    if program.len() == 0 {
        return;
    }

    const ERR_NON_NOR: &'static str = "input program must not mention any non-Nor Reg";
    let dom_tree = compute_dom_tree(cfg);

    let var_count = program
        .iter()
        .map(|insn| insn.dest.as_nor().expect(ERR_NON_NOR))
        .max()
        .unwrap() as usize
        + 1;
    let mut var_written = vec![false; var_count];

    // capacity is a heuristic
    let mut phis = vec![false; var_count * cfg.block_count()];
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
                &mut phis[ofs..ofs + var_count]
            };
            assert_eq!(block_phis.len(), var_count);
            assert_eq!(var_written.len(), var_count);
            for i in 0..var_count {
                block_phis[i] |= var_written[i];
            }
        });
    }

    todo!()
}

type DomTree = cfg::BlockMap<Option<cfg::BasicBlockID>>;

pub fn compute_dom_tree(cfg: &cfg::Graph) -> DomTree {
    let block_count = cfg.block_count();
    let rpo = cfg::traverse_reverse_postorder(&cfg);

    let mut parent = cfg::BlockMap::new(None, block_count);

    // process the entry node "manually", so the algorithm can rely on it for successors
    parent[cfg::ENTRY_BID] = Some(cfg::ENTRY_BID);

    let mut changed = true;
    while changed {
        changed = false;

        for &bid in rpo.order().iter() {
            let preds = cfg.predecessors(bid);
            if preds.len() == 0 {
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
