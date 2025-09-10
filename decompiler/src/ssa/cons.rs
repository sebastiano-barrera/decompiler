//! SSA construction (conversion from MIL)
//!
//! This module is fully internal. No API is exposed for consumption outside
//! of the super module (ssa).

use std::cell::Cell;

use tracing::{event, Level};

use super::Program;
use crate::{cfg, mil};

pub(super) fn mil_to_ssa(mut program: mil::Program) -> super::Program {
    // NOTE: the mil program is going to progressively become closer to the final SSA form.
    //
    // some notes:
    //
    // - the properties checked by `validate` may be violated by the time this function
    //   finishes (typically: Phi insns are going to be inserted, even though `validate`
    //   forbids them.)
    //
    // - the value u64::MAX is used for the 'addrs' array for all insns inserted
    //   from now on. it's a dummy value for instructions that have no direct
    //   connection to the source machine program

    program.validate();
    program.convert_jmp_ext_to_int();

    let var_count = program.count_distinct_regs();
    let vars = move || (0..var_count).map(mil::Reg);

    let cfg::MILAnalysis {
        graph: cfg,
        mut schedule,
        ..
    } = cfg::analyze_mil(&program);
    assert_eq!(cfg.block_count(), schedule.block_count());

    let dom_tree = cfg.dom_tree();
    let is_phi_needed = compute_phis_set(&program, &cfg, &schedule, dom_tree);

    // from now on, `program` is going to progressively become closer to the SSA result

    // create all required phi nodes, and link them to basic blocks and
    // variables, to be "wired" later to the data flow graph
    let phis = {
        let mut phis = RegMat::with_size(None, var_count, cfg.block_count());
        for bid in cfg.block_ids() {
            for var in vars() {
                if *is_phi_needed.get(bid, var) {
                    // just set `var` as the dest; the regular renaming process
                    // will process this insn just like the others
                    let ndx = program.len().try_into().unwrap();
                    program.push(var, mil::Insn::Phi);

                    // in `phis`, we need a stable ID for this insn (which
                    // survives the renaming)
                    phis.set(bid, var, Some(ndx));
                    schedule.insert(ndx, bid, 0);
                }
            }
        }
        phis
    };

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

    let mut var_map = VarMap::new(var_count);

    enum Cmd {
        Finish,
        Start(cfg::BlockID),
    }
    let mut queue = vec![Cmd::Finish, Cmd::Start(cfg.entry_block_id())];

    while let Some(cmd) = queue.pop() {
        match cmd {
            Cmd::Start(bid) => {
                var_map.push();

                // -- patch current block
                //
                // insns are treated as if already renamed so that their dest
                // becomes mil::Reg(index) (this replacement will physically
                // happen later on).
                for &insn_ndx in schedule.of_block(bid) {
                    let iv = program.get(insn_ndx).unwrap();
                    let mut insn = iv.insn.get();
                    for reg in insn.input_regs() {
                        *reg = var_map.get(*reg).expect("value not initialized in pre-ssa");
                    }

                    // in the output SSA, each destination register corresponds to the instruction's
                    // index. this way, the numeric value of a register can also be used as
                    // instruction ID, to locate a register/variable's defining instruction.
                    let new_dest = mil::Reg(insn_ndx);
                    // iv.dest is rewritten later (always mil::Reg(index))
                    // we only update `var_map`
                    let old_name = iv.dest.get();

                    let new_name = if let mil::Insn::Get(input_reg) = insn {
                        // exception: for Get(_) instructions, we just reuse the input reg for the
                        // output
                        input_reg
                    } else {
                        new_dest
                    };
                    var_map.set(old_name, new_name);
                    iv.insn.set(insn);
                }

                // -- patch successor's phi nodes
                // The algorithm is the following:
                // >  for s in block's successors:
                // >    for p in s's ϕ-nodes:
                // >      Assuming p is for a variable v, make it read from stack[v].
                //
                // Thanks to the Upsilon formulation of phi nodes, it's
                // sufficient to add one such insn to obtain the final correct
                // representation. Algorithms that want to know the operands of
                // the phi node will iterate in reverse through the
                // predecessor's instructions.

                for succ in cfg.block_cont(bid).block_dests() {
                    for var in vars() {
                        if let &Some(phi_ndx) = phis.get(succ, var) {
                            let value = var_map
                                .get(var)
                                .expect("value not initialized in pre-ssa (phi)");

                            let ndx = program.len().try_into().unwrap();
                            program.push(
                                mil::Reg(ndx),
                                mil::Insn::Upsilon {
                                    value,
                                    phi_ref: mil::Reg(phi_ndx),
                                },
                            );
                            schedule.append(ndx, bid);
                        }
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

    // convert into plain data, fully accessible form
    let program = program.unwrap();
    let tyids = program
        .insns
        .iter()
        .map(|insn| match insn {
            mil::Insn::Ancestral { anc_name, size: _ } => {
                program.ancestral_tyids.get(anc_name).copied()
            }
            _ => None,
        })
        .collect();

    let ssa = Program {
        insns: program.insns.into_iter().map(Cell::new).collect(),
        addrs: program.addrs,
        tyids,
        schedule,
        cfg,
    };
    event!(Level::TRACE, ?ssa, "ssa constructed");
    ssa.assert_invariants();
    ssa
}

fn compute_phis_set(
    program: &mil::Program,
    cfg: &cfg::Graph,
    block_spans: &cfg::Schedule,
    dom_tree: &cfg::DomTree,
) -> RegMat<bool> {
    // matrix [B * B] where B = #blocks
    // matrix[i, j] = true iff block j is in block i's dominance frontier
    let is_dom_front = compute_dominance_frontier(cfg, dom_tree);
    let block_uses_var = find_received_vars(program, block_spans, cfg);
    let mut phis_set = RegMat::for_program(program, cfg, false);

    // the rule is:
    //   if variable v is written in block b,
    //   then we have to add `v <- phi v, v, ...` on each block in b's dominance frontier
    // (phis_set is to avoid having multiple phis for the same var)
    for bid in cfg.block_ids() {
        for &ndx in block_spans.of_block(bid) {
            let dest = program.get(ndx).unwrap().dest.get();
            for target_bid in cfg.block_ids() {
                if cfg.block_preds(target_bid).len() < 2 {
                    continue;
                }

                let is_blk_in_df = *is_dom_front.item(bid.as_usize(), target_bid.as_usize());
                let is_used = *block_uses_var.get(target_bid, dest);

                if is_blk_in_df && is_used {
                    phis_set.set(target_bid, dest, true);
                }
            }
        }
    }

    phis_set
}

/// Find the set of "received variables" for each block.
///
/// This is the set of variables which are read before any write in the block.
/// In other words, for these are the variables, the block observes the values
/// left there by other blocks.
fn find_received_vars(
    prog: &mil::Program,
    block_spans: &cfg::Schedule,
    graph: &cfg::Graph,
) -> RegMat<bool> {
    let mut is_received = RegMat::for_program(prog, graph, false);
    for bid in graph.block_ids_postorder() {
        for &ndx in block_spans.of_block(bid).iter().rev() {
            let iv = prog.get(ndx).unwrap();
            let dest = iv.dest.get();
            let mut insn = iv.insn.get();

            is_received.set(bid, dest, false);
            for &mut input in insn.input_regs_iter() {
                is_received.set(bid, input, true);
            }
        }
    }
    is_received
}

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

#[derive(Debug)]
struct RegMat<T>(Mat<T>);

impl<T: Clone> RegMat<T> {
    fn for_program(program: &mil::Program, graph: &cfg::Graph, value: T) -> Self {
        // NOTE(perf) this function is called a few times, and each time `count_distinct_regs`
        // is going to iterate through the program.  maybe wasteful, but I bet it's fine.
        let var_count = program.count_distinct_regs();
        let block_count = graph.block_count();
        Self::with_size(value, var_count, block_count)
    }

    fn with_size(value: T, var_count: u16, block_count: u16) -> RegMat<T> {
        RegMat(Mat::new(value, block_count as usize, var_count as usize))
    }

    fn get(&self, bid: cfg::BlockID, reg: mil::Reg) -> &T {
        self.0
            .item(bid.as_number() as usize, reg.reg_index() as usize)
    }

    fn set(&mut self, bid: cfg::BlockID, reg: mil::Reg, value: T) {
        *self
            .0
            .item_mut(bid.as_number() as usize, reg.reg_index() as usize) = value;
    }
}

struct Mat<T> {
    items: Box<[T]>,
    rows: usize,
    cols: usize,
}

impl<T> Mat<T> {
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

impl<T: Clone> Mat<T> {
    fn new(init: T, rows: usize, cols: usize) -> Self {
        let items = vec![init; rows * cols].into_boxed_slice();
        Mat { items, rows, cols }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Mat<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::io::Write;

        let mut item_buf = Vec::new();

        writeln!(f, "mat. {}×{}", self.rows, self.cols)?;
        for i in 0..self.rows {
            for j in 0..self.cols {
                let value = self.item(i, j);
                item_buf.clear();
                write!(item_buf, "{:?}", value).unwrap();
                write!(f, "{:10} ", std::str::from_utf8(&item_buf).unwrap())?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

fn compute_dominance_frontier(graph: &cfg::Graph, dom_tree: &cfg::DomTree) -> Mat<bool> {
    let count = graph.block_count() as usize;
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
