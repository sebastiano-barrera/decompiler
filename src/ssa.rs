use std::{
    collections::{HashMap, HashSet},
    ops::Range,
};

use slotmap::SlotMap;

/// Static Single-Assignment representation of a program (and conversion from direct multiple
/// assignment).
///
/// The algorithms in this module are mostly derived from the descriptions in:
/// > Cooper, Keith & Harvey, Timothy & Kennedy, Ken. (2006).
/// > A Simple, Fast Dominance Algorithm.
/// > Rice University, CS Technical Report 06-33870.
use crate::{cfg, mil};

use mil::{AncestralName, ArithOp, BoolOp, CmpOp};

#[derive(Clone)]
pub struct Program {
    // Sea-of-Nodes representation
    control_graph: SlotMap<ControlNID, ControlNode>,
    data_graph: SlotMap<DataNID, DataNode>,
}

slotmap::new_key_type! { struct ControlNID; }
slotmap::new_key_type! { struct DataNID; }

#[derive(Clone)]
enum ControlNode {
    /// Start of the function. The only control node without control dependencies.
    Start,

    /// End of the function.
    End {
        /// Predecessors
        pred: ControlNID,
        /// Return value
        ret: DataNID,
    },

    Merge {
        // TODO more efficient repr (SmallVec?)
        preds: Vec<ControlNID>,
    },

    /// Jump to the address obtained by evaluating the `addr` expression.
    ///
    /// The target address is to be considered external to the function
    /// represented by this graph, as the compiler attempts to translate all jumps
    /// that "fall into" this function to other node types.
    JumpIndirect {
        pred: ControlNID,
        addr: DataNID,
    },
    /// If branch, where the target address is determined by evaluating the `addr` expression.
    ///
    /// The target address is to be considered external to the function
    /// represented by this graph, as the compiler attempts to translate all jumps
    /// that "fall into" this function to other node types.
    BranchIndirect {
        pred: ControlNID,
        cond: DataNID,
        addr: DataNID,
    },

    /// The Jump's destination is encoded in another node (this node shows up as
    /// the other node's predecessor).
    Jump {
        pred: ControlNID,
    },

    Branch {
        cond: DataNID,
    },
    /// The consequent ("if condition true") branch of a Branch node
    IfTrue(ControlNID),
    /// The alternate ("if condition false") branch of a Branch node
    IfFalse(ControlNID),

    Call {
        pred: ControlNID,
        callee: DataNID,
        args: Vec<DataNID>,
    },

    Load {
        pred: ControlNID,
        addr: DataNID,
    },
    Store {
        pred: ControlNID,
        addr: DataNID,
        value: DataNID,
    },
}

/// Size of a value in the IR, expressed in bytes.
///
/// Typically mirrors the size of an operand's size in the original assembly, so
/// the maximum is the size of a machine register (for x86_64: 8 bytes).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct ValueSize(u8);

/// A Phi node.
#[derive(Clone)]
struct Phi {
    /// ID to the Merge node which determines this Phi's value, based on the
    /// predecessors that the program enters it from at runtime.
    merge_nid: ControlNID,

    /// This nodes assumes value `values[i]` at runtime whenever the
    /// corresponding Merge control node `Merge { preds }` is entered from
    /// predecessor `preds[i]`
    ///
    /// Invariant: all input values are of the same type
    // TODO more efficient repr (SmallVec?)
    values: Vec<DataNID>,
}

#[derive(Clone)]
enum DataNode {
    ConstBool(bool),
    ConstInt {
        size: ValueSize,
        value: i64,
    },

    Part {
        src: DataNID,
        offset: u8,
        size: u8,
    },
    Concat {
        lo: DataNID,
        hi: DataNID,
    },

    Widen {
        input: DataNID,
        out_size: ValueSize,
    },
    Arith(ArithOp, DataNID, DataNID),
    ArithK(ArithOp, DataNID, i64),
    Cmp(CmpOp, DataNID, DataNID),
    Bool(BoolOp, DataNID, DataNID),
    Not(DataNID),

    OverflowOf(DataNID),
    CarryOf(DataNID),
    SignOf(DataNID),
    IsZero(DataNID),
    Parity(DataNID),

    Ancestral(AncestralName),

    /// A call's return value.
    ///
    /// The ControlNID must correspond to a ControlNode::Call node
    ReturnValueOf(ControlNID),

    Phi(Phi),
}

impl Program {
    /// Get the defining instruction for the given register.
    ///
    /// (Note that it's not allowed to fetch instructions by position.)
    pub fn get(&self, reg: mil::Reg) -> Option<mil::InsnView> {
        todo!()
    }

    pub fn assert_invariants(&self) {
        // TODO collect & check invariants
        self.assert_inputs_alive();
        self.assert_no_circular_refs();
    }

    fn assert_inputs_alive(&self) {
        // TODO reimplement
    }

    fn assert_no_circular_refs(&self) {
        // TODO reimplement
    }
}

#[cfg(test)]
#[test]
#[should_panic]
fn test_assert_no_circular_refs() {
    use mil::{ArithOp, Insn, Reg};

    let prog = {
        let mut pb = mil::ProgramBuilder::new();
        pb.set_input_addr(0xf0);
        pb.push(Reg(0), Insn::Const8(123));
        pb.push(Reg(1), Insn::Arith8(ArithOp::Add, Reg(0), Reg(2)));
        pb.push(Reg(2), Insn::Arith8(ArithOp::Add, Reg(0), Reg(1)));
        pb.build()
    };
    let prog = mil_to_ssa(&prog);
    prog.assert_no_circular_refs();
}

impl std::fmt::Debug for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count: usize = self.live_regs().count();
        writeln!(f, "ssa program  {} instrs", count)?;

        for bid in self.cfg.block_ids_rpo() {
            let phis = self.block_phi(bid);
            let block_addr = {
                let nor_ndxs = self.cfg.insns_ndx_range(bid);
                self.inner.get(nor_ndxs.start).unwrap().addr
            };
            let insns = self.block_normal_insns(bid).unwrap();

            write!(f, ".B{}:    ;;  ", bid.as_usize())?;
            let preds = self.cfg.block_preds(bid);
            if preds.len() > 0 {
                write!(f, "preds:")?;
                for (ndx, pred) in preds.iter().enumerate() {
                    if ndx > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "B{}", pred.as_number())?;
                }
                write!(f, "  ")?;
            }
            writeln!(
                f,
                "addr:0x{:x}; {} insn {} phis",
                block_addr,
                insns.insns.len(),
                phis.phi_count(),
            )?;

            let print_rdr_count = |f: &mut std::fmt::Formatter, reg| -> std::fmt::Result {
                let rdr_count = self.rdr_count.get(reg);
                if rdr_count > 1 {
                    write!(f, "  ({:3})  ", rdr_count)?;
                } else {
                    write!(f, "         ")?;
                }
                Ok(())
            };

            if phis.phi_count() > 0 {
                write!(f, "                  ɸ  ")?;
                for pred in preds {
                    write!(f, "B{:<5} ", pred.as_number())?;
                }
                writeln!(f)?;
                for phi in phis.phi_regs() {
                    print_rdr_count(f, phi)?;
                    write!(f, "  r{:<5} <- ", phi.0)?;
                    for arg in self.get_phi_args(phi) {
                        write!(f, "r{:<5} ", arg.0)?;
                    }
                    writeln!(f)?;
                }
            }

            for (dest, mut insn) in insns.iter_copied() {
                if self.is_alive(dest) {
                    // modify insn (our copy) so that the registers skip/dereference any Get
                    for input in insn.input_regs_iter_mut() {
                        loop {
                            let input_def = self.get(*input).unwrap().insn.get();
                            if let mil::Insn::Get(x) = input_def {
                                *input = x;
                            } else {
                                break;
                            }
                        }
                    }

                    print_rdr_count(f, dest)?;
                    writeln!(f, "{:?} <- {:?}", dest, insn)?;
                }
            }
        }

        Ok(())
    }
}

pub fn mil_to_ssa(program: &mil::Program) -> Program {
    let cfg = cfg::analyze_mil(&program);

    let dom_tree = cfg.dom_tree();
    let phis_set = compute_phis_set(program, &cfg, dom_tree);

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

    // let var_count = program.reg_count();
    // let mut var_map = VarMap::new(var_count);

    //
    // (+) for register renaming, we can rename dest[i] (assignment target of the i-th instruction)
    //     to Reg(i) (register whose ID is nominally equal to the insn's index)
    //
    //      <- each mil insn results in exactly one assignment
    //              [trick] actually, it results in *at most* one assignment,
    //              but we can merge the two cases by saying that non-assigning instructions
    //              result in a "unit/side-effect-only" type, which is what we usually want in
    //              SSA anyway (we can identify the side-effect-only insn using the "unit" register)
    //
    //      <- in SSA, each assignment targets a unique variable (might as well be the insn's index)
    //
    // (!) what remains is connecting use->def; then the 'use' input reg can be
    //   renamed to 'def' as specified above
    //
    // (+) scan each basic block; determine the order by scanning the dominators tree in preorder.
    //     for each basic block, start with the variable mapping at the end of the immediate dominator,
    //     and update it based on the phi nodes; then for each insn, update it with the insn's assignment;
    //     transform each insn into a data and/or control node, renaming registers based on the current
    //     state of the mapping.
    //
    //      <- each use->def link can be defined as: the def is the "last visible" NID corresponding
    //         to the use's mil register ("last" in order of insn execution)
    //
    //      <- within a single basic block, each insn "updates" the variables mapping
    //         based on its assignment (dest[i] mapped to Reg(i))
    //
    //      <- at the beginning of each basic block, the mapping is the one that was valid
    //         at the end of the immediate dominator block, updated based on the block's
    //         phi nodes.
    //
    //         <- this is because only the assignments happened before the end of the imm. dom
    //            are guaranteed to have happened and to still be valid at the begining of "this"
    //            block; everything else needs a phi node
    //
    //      <- the mapping is Ø at the beginning of the entry basic block
    //

    let mut control_graph = SlotMap::with_key();
    let mut data_graph = SlotMap::with_key();

    enum Cmd {
        Block(cfg::BlockID),
    }
    let mut queue = vec![Cmd::Block(cfg.entry_block_id())];

    let mut var_maps = BlockRegMat::for_program(program, &cfg, None);
    let var_count = program.reg_count() as usize;

    let cnid_of_bid: Vec<_> = todo!();
    let mut ends = Vec::new();

    while let Some(Cmd::Block(bid)) = queue.pop() {
        let mut var_map = if let Some(imm_dom_bid) = cfg.dom_tree().parent_of(bid) {
            var_maps.of_block(imm_dom_bid).to_vec()
        } else {
            vec![None; var_count]
        };

        // each block starts with a control node
        let preds = cfg.block_preds(bid);
        let control_node = if preds.is_empty() {
            ControlNode::Start
        } else {
            ControlNode::Merge {
                preds: preds
                    .iter()
                    .map(|pred| cnid_of_bid[pred.as_usize()].unwrap())
                    .collect(),
            }
        };
        let cnid = control_graph.insert(control_node);

        for (var_ndx, &is_phi_needed) in phis_set.of_block(bid).iter().enumerate() {
            let reg = mil::Reg(var_ndx.try_into().unwrap());
            if is_phi_needed {
                let dnid = data_graph.insert(DataNode::Phi(Phi {
                    merge_nid: cnid,
                    values: preds
                        .iter()
                        .map(|&pred| var_maps.get(pred, reg).expect("unmapped phi input reg"))
                        .collect(),
                }));
                var_map[var_ndx] = Some(dnid);
            }
        }

        let insns = program.slice(cfg.insns_ndx_range(bid)).expect("broken cfg");
        let mut cnid = cnid;
        let mut iter = insns.iter().peekable();
        while let Some((dest, insn)) = iter.next() {
            let dnid_of_reg =
                |reg: mil::Reg| var_map[reg.reg_index() as usize].expect("unmapped reg");

            let data_node = match insn.get() {
                mil::Insn::True => Some(DataNode::ConstBool(true)),
                mil::Insn::False => Some(DataNode::ConstBool(false)),
                mil::Insn::Const1(_) => todo!(),
                mil::Insn::Const2(_) => todo!(),
                mil::Insn::Const4(_) => todo!(),
                mil::Insn::Const8(_) => todo!(),
                mil::Insn::Get(reg) => todo!(),
                mil::Insn::Part { src, offset, size } => todo!(),
                mil::Insn::Concat { lo, hi } => todo!(),
                mil::Insn::Widen1_2(reg) => todo!(),
                mil::Insn::Widen1_4(reg) => todo!(),
                mil::Insn::Widen1_8(reg) => todo!(),
                mil::Insn::Widen2_4(reg) => todo!(),
                mil::Insn::Widen2_8(reg) => todo!(),
                mil::Insn::Widen4_8(reg) => todo!(),
                mil::Insn::Arith1(arith_op, reg, reg1) => todo!(),
                mil::Insn::Arith2(arith_op, reg, reg1) => todo!(),
                mil::Insn::Arith4(arith_op, reg, reg1) => todo!(),
                mil::Insn::Arith8(arith_op, reg, reg1) => todo!(),
                mil::Insn::ArithK1(arith_op, reg, _) => todo!(),
                mil::Insn::ArithK2(arith_op, reg, _) => todo!(),
                mil::Insn::ArithK4(arith_op, reg, _) => todo!(),
                mil::Insn::ArithK8(arith_op, reg, _) => todo!(),
                mil::Insn::Cmp(cmp_op, reg, reg1) => todo!(),
                mil::Insn::Bool(bool_op, reg, reg1) => todo!(),
                mil::Insn::Not(reg) => todo!(),

                mil::Insn::Call(callee) => {
                    let mut args = Vec::new();
                    while let Some((dest, insn)) = iter.peek() {
                        if let mil::Insn::CArg(reg) = insn.get() {
                            args.push(dnid_of_reg(reg));
                            iter.next();
                        } else {
                            break;
                        }
                    }

                    cnid = control_graph.insert(ControlNode::Call {
                        pred: cnid,
                        callee: dnid_of_reg(callee),
                        args,
                    });

                    Some(DataNode::ReturnValueOf(cnid))
                }
                mil::Insn::CArg(_) => {
                    panic!("compiler bug or malformed mil: CArg node should be skipped here")
                }

                mil::Insn::Ret(reg) => {
                    cnid = control_graph.insert(ControlNode::Jump { pred: cnid });
                    ends.push((cnid, dnid_of_reg(reg)));
                    None
                }
                mil::Insn::JmpInd(reg) => {
                    cnid = control_graph.insert(ControlNode::JumpIndirect {
                        pred: cnid,
                        addr: dnid_of_reg(reg),
                    });
                    None
                }
                mil::Insn::Jmp(target) => {
                    let target_bid = cfg
                        .block_starting_at(target)
                        .expect("inconsistent mil/cfg: no block at jmp target");

                    None
                }
                mil::Insn::JmpIf { cond, target } => todo!(),
                mil::Insn::JmpExt(_) => todo!(),
                mil::Insn::JmpExtIf { cond, addr } => todo!(),
                mil::Insn::TODO(_) => todo!(),
                mil::Insn::LoadMem1(reg) => todo!(),
                mil::Insn::LoadMem2(reg) => todo!(),
                mil::Insn::LoadMem4(reg) => todo!(),
                mil::Insn::LoadMem8(reg) => todo!(),
                mil::Insn::StoreMem(reg, reg1) => todo!(),
                mil::Insn::OverflowOf(reg) => todo!(),
                mil::Insn::CarryOf(reg) => todo!(),
                mil::Insn::SignOf(reg) => todo!(),
                mil::Insn::IsZero(reg) => todo!(),
                mil::Insn::Parity(reg) => todo!(),
                mil::Insn::Undefined => todo!(),
                mil::Insn::Ancestral(ancestral_name) => todo!(),
                mil::Insn::Phi => todo!(),
                mil::Insn::PhiBool => todo!(),
                mil::Insn::PhiArg(reg) => todo!(),
            };

            if let Some(data_node) = data_node {
                let dnid = data_graph.insert(data_node);
                var_map[dest.get().reg_index() as usize] = Some(dnid);
            }
        }
    }

    let final_merge = control_graph.insert(ControlNode::Merge {
        preds: ends.iter().map(|(cnid, _)| *cnid).collect(),
    });
    let final_phi = data_graph.insert(DataNode::Phi(Phi {
        merge_nid: final_merge,
        values: ends.into_iter().map(|(_, dnid)| dnid).collect(),
    }));
    control_graph.insert(ControlNode::End {
        pred: final_merge,
        ret: final_phi,
    });

    // establish SSA invariants
    // the returned Program will no longer change (just some instructions are going to be marked as
    // "dead" and ignored)
    for (ndx, insn) in program.iter().enumerate() {
        let ndx = ndx.try_into().unwrap();
        assert_eq!(
            insn.dest.get(),
            mil::Reg(ndx),
            "insn unvisited: {:?} <- {:?}",
            insn.dest,
            insn.insn
        );
    }

    let mut ssa = Program {
        control_graph,
        data_graph,
    };
    eliminate_dead_code(&mut ssa);
    narrow_phi_nodes(&mut ssa);
    ssa
}

#[derive(Clone)]
struct BlockPhis {}

impl BlockPhis {
    fn empty() -> BlockPhis {}
}

/// Add phi nodes to a MIL program, where required.
///
/// Some notes:
///
/// - This function acts on MIL programs not yet in SSA form. Phi nodes are
/// added in preparation for an SSA conversion, but no part of the actual
/// conversion is performed at all.
///
/// - The added Phi nodes are all with the largest result width (Phi8). They
/// must be later swapped for the proper width variants (Phi1, Phi2, Phi4, ...)
/// by calling `narrow_phi_nodes` on the SSA-converted program.
fn place_phi_nodes(
    program: &mil::Program,
    cfg: &cfg::Graph,
    dom_tree: &cfg::DomTree,
) -> cfg::BlockMap<BlockPhis> {
    let block_count = cfg.block_count();

    if program.len() == 0 {
        return cfg::BlockMap::new(BlockPhis::empty(), block_count);
    }

    let phis_set = compute_phis_set(program, cfg, dom_tree);
    let var_count = phis_set.0.cols.try_into().unwrap();
    assert_eq!(var_count, program.reg_count());

    // phis_set[B][reg] == true
    //   <==> Block B has a phi node for register reg

    // Translate `phis_set` in a form such that the phi nodes inputs and outputs
    // can undergo register renaming during SSA translation later
    let mut phis = cfg::BlockMap::new(BlockPhis::empty(), block_count);

    for bid in cfg.block_ids() {
        let pred_count: u16 = cfg.block_preds(bid).len().try_into().unwrap();

        for var_ndx in 0..var_count {
            let reg = mil::Reg(var_ndx);
            if *phis_set.get(bid, reg) {
                program.push(
                    reg,
                    Phi {
                        merge_nid: todo!(),
                        values: todo!(),
                    },
                );

                for _pred_ndx in 0..pred_count {
                    // implicitly corresponds to _pred_ndx
                    todo!();
                }
            }
        }

        phis[bid] = phi_info;
    }

    phis
}

fn compute_phis_set(
    program: &mil::Program,
    cfg: &cfg::Graph,
    dom_tree: &cfg::DomTree,
) -> BlockRegMat<bool> {
    // matrix [B * B] where B = #blocks
    // matrix[i, j] = true iff block j is in block i's dominance frontier
    let is_dom_front = compute_dominance_frontier(cfg, dom_tree);
    let block_uses_var = find_received_vars(program, cfg);
    let mut phis_set = BlockRegMat::for_program(program, cfg, false);

    // the rule is:
    //   if variable v is written in block b,
    //   then we have to add `v <- phi v, v, ...` on each block in b's dominance frontier
    // (phis_set is to avoid having multiple phis for the same var)
    for bid in cfg.block_ids() {
        let slice = program.slice(cfg.insns_ndx_range(bid)).unwrap();
        for dest in slice.dests.iter() {
            let dest = dest.get();
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

fn narrow_phi_nodes(program: &mut Program) {
    for iv in program.insns_unordered() {
        let dest = iv.dest.get();
        let insn = iv.insn.get();

        match insn {
            mil::Insn::Phi { size } => {
                if size != 8 {
                    panic!("narrow_phi_nodes: unexpected narrow phi node")
                }
            }
            _ => continue,
        }

        let mut args = program.get_phi_args(dest);
        let arg0 = args.next().unwrap();
        let rt0 = program.value_type(arg0);

        // check: all args have the same type!
        for arg in args {
            assert_eq!(
                program.value_type(arg),
                rt0,
                "malformed ssa: phi node has different type args"
            );
        }

        let repl_insn = match rt0 {
            mil::RegType::Effect => panic!("malformed ssa: phi node can't have Effect inputs"),
            mil::RegType::Bytes(size) => mil::Insn::Phi { size },
            mil::RegType::Bool => mil::Insn::PhiBool,
        };
        iv.insn.set(repl_insn);
    }
}

/// Find the set of "received variables" for each block.
///
/// This is the set of variables which are read before any write in the block.
/// In other words, for these are the variables, the block observes the values
/// left there by other blocks.
fn find_received_vars(prog: &mil::Program, graph: &cfg::Graph) -> BlockRegMat<bool> {
    let mut is_received = BlockRegMat::for_program(prog, graph, false);
    for bid in graph.block_ids_postorder() {
        for (dest, insn) in prog
            .slice(graph.insns_ndx_range(bid))
            .unwrap()
            .iter_copied()
            .rev()
        {
            is_received.set(bid, dest, false);
            for input in insn.input_regs_iter() {
                is_received.set(bid, input, true);
            }
        }
    }
    is_received
}

struct BlockRegMat<T>(Mat<T>);

impl<T: Clone> BlockRegMat<T> {
    fn for_program(program: &mil::Program, graph: &cfg::Graph, value: T) -> Self {
        let var_count = program.reg_count() as usize;
        BlockRegMat(Mat::new(value, graph.block_count() as usize, var_count))
    }

    fn of_block(&self, bid: cfg::BlockID) -> &[T] {
        self.0.row(bid.as_usize())
    }

    fn get(&self, bid: cfg::BlockID, reg: mil::Reg) -> &T {
        self.0.item(bid.as_usize(), reg.reg_index() as usize)
    }

    fn set(&mut self, bid: cfg::BlockID, reg: mil::Reg, value: T) {
        *self.0.item_mut(bid.as_usize(), reg.reg_index() as usize) = value;
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
    fn row(&self, i: usize) -> &[T] {
        &self.items[i * self.cols..(i + 1) * self.cols]
    }
}
impl<T: Clone> Mat<T> {
    fn new(init: T, rows: usize, cols: usize) -> Self {
        let items = vec![init; rows * cols].into_boxed_slice();
        Mat { items, rows, cols }
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

pub fn eliminate_dead_code(prog: &mut Program) {
    if prog.inner.len() == 0 {
        return;
    }

    let count = prog.reg_count();
    let mut rdr_count = ReaderCount::new(count);
    let mut visited = vec![false; count as usize];

    let mut queue: Vec<_> = prog
        .inner
        .iter()
        .filter(|iv| iv.insn.get().has_side_effects())
        .map(|iv| iv.dest.get())
        .collect();

    for &side_fx_reg in &queue {
        rdr_count.inc(side_fx_reg);
    }

    while let Some(reg) = queue.pop() {
        // we can assume no circular references
        visited[reg.reg_index() as usize] = true;
        let insn = prog.get(reg).unwrap().insn.get();
        match insn {
            mil::Insn::Phi { size: _ } | mil::Insn::PhiBool => {
                for input in prog.get_phi_args(reg) {
                    rdr_count.inc(input);
                    if !visited[input.reg_index() as usize] {
                        queue.push(input);
                    }
                }
            }
            _ => {
                for input in insn.input_regs_iter() {
                    rdr_count.inc(input);
                    if !visited[input.reg_index() as usize] {
                        queue.push(input);
                    }
                }
            }
        }
    }

    prog.rdr_count = rdr_count;
}

#[derive(Debug, Clone)]
struct ReaderCount(Vec<usize>);
impl ReaderCount {
    fn new(var_count: mil::Index) -> Self {
        ReaderCount(vec![0; var_count as usize])
    }
    fn reset(&mut self) {
        self.0.fill(0);
    }
    fn get(&self, reg: mil::Reg) -> usize {
        self.0[reg.0 as usize]
    }
    fn inc(&mut self, reg: mil::Reg) {
        if let Some(elm) = self.0.get_mut(reg.reg_index() as usize) {
            *elm += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::mil;

    #[test]
    fn test_phi_read() {
        use mil::{ArithOp, Insn, Reg};

        let prog = {
            let mut pb = mil::ProgramBuilder::new();

            pb.set_input_addr(0xf0);
            pb.push(Reg(0), Insn::Const8(123));
            pb.push(
                Reg(1),
                Insn::JmpExtIf {
                    cond: Reg(0),
                    addr: 0xf2,
                },
            );

            pb.set_input_addr(0xf1);
            pb.push(Reg(2), Insn::Const1(4));
            pb.push(Reg(3), Insn::JmpExt(0xf3));

            pb.set_input_addr(0xf2);
            pb.push(Reg(2), Insn::Const1(8));

            pb.set_input_addr(0xf3);
            pb.push(Reg(4), Insn::ArithK1(ArithOp::Add, Reg(2), 456));
            pb.push(Reg(5), Insn::Ret(Reg(4)));

            pb.build()
        };

        eprintln!("-- mil:");
        eprintln!("{:?}", prog);

        eprintln!("-- ssa:");
        let prog = super::mil_to_ssa(super::ConversionParams::new(prog));
        insta::assert_debug_snapshot!(prog);
    }
}
