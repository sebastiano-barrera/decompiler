/// Static Single-Assignment representation of a program (and conversion from direct multiple
/// assignment).
///
/// The algorithms in this module are mostly derived from the descriptions in:
/// > Cooper, Keith & Harvey, Timothy & Kennedy, Ken. (2006).
/// > A Simple, Fast Dominance Algorithm.
/// > Rice University, CS Technical Report 06-33870.
use crate::{cfg, mil};

pub struct Program {
    original: mil::Program,
}

impl Program {
    pub fn original(&self) -> &mil::Program {
        &self.original
    }
}

pub fn convert_to_ssa(program: &mil::Program, cfg: &cfg::Graph) -> Program {
    let phi_blocks = place_phi_nodes(cfg);
    todo!()
}

fn place_phi_nodes(cfg: &cfg::Graph) -> Vec<cfg::BasicBlockID> {
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
