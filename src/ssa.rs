use std::collections::HashSet;

use slotmap::{Key, SlotMap};

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
    start_cnid: ControlNID,
    end_cnid: ControlNID,
}

slotmap::new_key_type! { pub struct ControlNID; }
slotmap::new_key_type! { pub struct DataNID; }

#[derive(Clone, Debug)]
enum ControlNode {
    /// End of the function.
    End {
        /// Predecessors
        pred: ControlNID,
        /// Return value
        ret: DataNID,
    },

    /// Control node with zero or more predecessors.
    ///
    /// With zero predecessors, it marks the start of the function. (There must be only
    /// one per function.)
    ///
    /// With > 1 predecessors, it represents a control flow merge (join point). Phi nodes
    /// may then be associated to this as their Merge node, to also merge data flow in
    /// addition to control.
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
        pred: ControlNID,
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

    TODO {
        pred: ControlNID,
        label: &'static str,
    },
}

pub struct Predecessors(Vec<ControlNID>);
impl IntoIterator for Predecessors {
    type Item = ControlNID;
    type IntoIter = <Vec<ControlNID> as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

pub struct Inputs(Vec<DataNID>);
impl IntoIterator for Inputs {
    type Item = DataNID;
    type IntoIter = <Vec<DataNID> as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl ControlNode {
    fn predecessors(&self) -> Predecessors {
        // TODO remove this waste! change data structure
        let preds = match self {
            ControlNode::Merge { preds } => preds.clone(),
            ControlNode::End { pred, .. }
            | ControlNode::JumpIndirect { pred, .. }
            | ControlNode::BranchIndirect { pred, .. }
            | ControlNode::Jump { pred }
            | ControlNode::Branch { pred, .. }
            | ControlNode::IfTrue(pred)
            | ControlNode::IfFalse(pred)
            | ControlNode::Call { pred, .. }
            | ControlNode::Load { pred, .. }
            | ControlNode::Store { pred, .. }
            | ControlNode::TODO { pred, .. } => vec![*pred],
        };
        Predecessors(preds)
    }

    fn data_inputs(&self) -> Inputs {
        let inputs = match self {
            ControlNode::End { ret, .. } => vec![*ret],
            ControlNode::Merge { .. } => vec![],
            ControlNode::JumpIndirect { addr, .. } => vec![*addr],
            ControlNode::BranchIndirect { cond, addr, .. } => vec![*cond, *addr],
            ControlNode::Jump { .. } => vec![],
            ControlNode::Branch { cond, .. } => vec![*cond],
            ControlNode::IfTrue(_) => vec![],
            ControlNode::IfFalse(_) => vec![],
            ControlNode::Call { callee, args, .. } => {
                let mut inputs = args.clone();
                inputs.push(*callee);
                inputs
            }
            ControlNode::Load { addr, .. } => vec![*addr],
            ControlNode::Store { addr, value, .. } => vec![*addr, *value],
            ControlNode::TODO { .. } => vec![],
        };
        Inputs(inputs)
    }

    /// Return a representation of the node that doesn't include data/control dependencies.
    fn implicit_repr(&self) -> String {
        match self {
            ControlNode::End { .. } => format!("End"),
            ControlNode::Merge { .. } => format!("Merge"),
            ControlNode::JumpIndirect { .. } => format!("JumpIndirect"),
            ControlNode::BranchIndirect { .. } => format!("BranchIndirect"),
            ControlNode::Jump { .. } => format!("Jump"),
            ControlNode::Branch { .. } => format!("Branch"),
            ControlNode::IfTrue(_) => format!("IfTrue"),
            ControlNode::IfFalse(_ontrol_nid) => format!("IfFalse"),
            ControlNode::Call { .. } => format!("Call"),
            ControlNode::Load { .. } => format!("Load"),
            ControlNode::Store { .. } => format!("Store"),
            ControlNode::TODO { label, .. } => format!("TODO({:?})", label),
        }
    }
}

/// Size of a value in the IR, expressed in bytes.
///
/// Typically mirrors the size of an operand's size in the original assembly, so
/// the maximum is the size of a machine register (for x86_64: 8 bytes).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct ValueSize(u8);

/// A Phi node.
#[derive(Clone, Debug)]
struct Phi {
    /// ID to the Merge node which determines this Phi's value, based on the
    /// predecessors that the program enters it from at runtime.
    merge_nid: ControlNID,

    /// Each entry (cnid, dnid) in this vector means that the Phi node latches
    /// the value from dnid whenever merge_nid is entered from
    ///
    /// Invariant: all ControlNIDs that appear here also appear in merge_nid's
    /// `preds`, and vice-versa.
    // TODO more efficient repr (SmallVec?)
    values: Vec<(ControlNID, DataNID)>,
}

#[derive(Clone, Debug)]
enum DataNode {
    Undefined,
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

    LoadedValueOf(ControlNID),

    Phi(Phi),
}

impl DataNode {
    fn control_inputs(&self) -> Predecessors {
        // TODO remove this waste! choose a different data structure
        let cnids = match self {
            DataNode::Undefined => vec![],
            DataNode::ConstBool(_) => vec![],
            DataNode::ConstInt { .. } => vec![],
            DataNode::Part { .. } => vec![],
            DataNode::Concat { .. } => vec![],
            DataNode::Widen { .. } => vec![],
            DataNode::Arith(_, _, _) => vec![],
            DataNode::ArithK(_, _, _) => vec![],
            DataNode::Cmp(_, _, _) => vec![],
            DataNode::Bool(_, _, _) => vec![],
            DataNode::Not(_) => vec![],
            DataNode::OverflowOf(_) => vec![],
            DataNode::CarryOf(_) => vec![],
            DataNode::SignOf(_) => vec![],
            DataNode::IsZero(_) => vec![],
            DataNode::Parity(_) => vec![],
            DataNode::Ancestral(_) => vec![],

            DataNode::ReturnValueOf(cnid)
            | DataNode::Phi(Phi {
                merge_nid: cnid, ..
            })
            | DataNode::LoadedValueOf(cnid) => vec![*cnid],
        };
        Predecessors(cnids)
    }

    fn data_inputs(&self) -> Inputs {
        // TODO remove this waste! choose a different data structure
        let inputs = match self {
            DataNode::Undefined => vec![],
            DataNode::ConstBool(_) => vec![],
            DataNode::ConstInt { .. } => vec![],
            DataNode::Part { src, .. } => vec![*src],
            DataNode::Concat { lo, hi } => vec![*lo, *hi],
            DataNode::Widen { input, .. } => vec![*input],
            DataNode::Arith(_, a, b) => vec![*a, *b],
            DataNode::ArithK(_, a, _) => vec![*a],
            DataNode::Cmp(_, a, b) => vec![*a, *b],
            DataNode::Bool(_, a, b) => vec![*a, *b],
            DataNode::Not(x) => vec![*x],
            DataNode::OverflowOf(x) => vec![*x],
            DataNode::CarryOf(x) => vec![*x],
            DataNode::SignOf(x) => vec![*x],
            DataNode::IsZero(x) => vec![*x],
            DataNode::Parity(x) => vec![*x],
            DataNode::Ancestral(_) => vec![],
            DataNode::ReturnValueOf(_) => vec![],
            DataNode::LoadedValueOf(_) => vec![],
            DataNode::Phi(Phi { values, .. }) => values.iter().map(|(_, dnid)| *dnid).collect(),
        };
        Inputs(inputs)
    }

    /// Return a representation of the node that doesn't include data/control dependencies.
    fn implicit_repr(&self) -> String {
        match self {
            DataNode::Undefined => format!("Undefined"),
            DataNode::ConstBool(value) => format!("{:?}", value),
            DataNode::ConstInt { size: _, value } => format!("{:?}", value),
            DataNode::Part {
                src: _,
                offset,
                size,
            } => format!("[{}:{}]", offset, size),
            DataNode::Concat { .. } => format!("Concat"),
            DataNode::Widen { input: _, out_size } => format!("Widen({})", out_size.0),
            DataNode::Arith(arith_op, _, _) => format!("{:?}", arith_op),
            DataNode::ArithK(arith_op, _, _) => format!("{:?}", arith_op),
            DataNode::Cmp(cmp_op, _, _) => format!("{:?}", cmp_op),
            DataNode::Bool(bool_op, _, _) => format!("{:?}", bool_op),
            DataNode::Not(_) => format!("not"),
            DataNode::OverflowOf(_) => format!("OverflowOf"),
            DataNode::CarryOf(_) => format!("CarryOf"),
            DataNode::SignOf(_) => format!("SignOf"),
            DataNode::IsZero(_) => format!("IsZero"),
            DataNode::Parity(_) => format!("Parity"),
            DataNode::Ancestral(ancestral_name) => format!("{:?}", ancestral_name),
            DataNode::ReturnValueOf(_) => format!("ReturnValue"),
            DataNode::LoadedValueOf(_) => format!("LoadedValue"),
            DataNode::Phi(phi) => format!("Phi"),
        }
    }
}

#[derive(Default)]
pub struct Uses {
    pub control: Vec<ControlNID>,
    pub data: Vec<DataNID>,
}

pub type InverseGraph<NID> = slotmap::SecondaryMap<NID, Uses>;
pub type InverseDataGraph = InverseGraph<DataNID>;
pub type InverseControlGraph = InverseGraph<ControlNID>;

impl Program {
    // TODO inefficient data structures!
    pub fn find_uses_data(&self) -> InverseDataGraph {
        let mut uses_data: slotmap::SecondaryMap<DataNID, Uses> = self
            .data_graph
            .keys()
            .map(|dnid| (dnid, Uses::default()))
            .collect();

        for (dnid, dn) in self.data_graph.iter() {
            for input_dnid in dn.data_inputs() {
                uses_data[input_dnid].data.push(dnid);
            }
        }

        for (cnid, cn) in self.control_graph.iter() {
            for input_dnid in cn.data_inputs() {
                uses_data[input_dnid].control.push(cnid);
            }
        }

        uses_data

    }

    pub fn assert_invariants(&self) {
        // TODO collect & check invariants
        self.assert_inputs_alive();
        self.assert_no_circular_refs();
        // TODO: only one start node per function (Merge with zero preds)
        // TODO: data nodes point to the correct controlnode (e.g. phi to merge, returnvalueof to return, etc.)
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
        writeln!(f, "ssa program")?;
        writeln!(
            f,
            "  {} control nodes, {} data nodes",
            self.control_graph.len(),
            self.data_graph.len()
        )?;

        enum Cmd {
            StartC(ControlNID),
            EndC(ControlNID),
            StartD(DataNID),
            EndD(DataNID),
        }

        let mut visited_data = HashSet::new();
        let mut visited_control = HashSet::new();

        let mut queue = Vec::with_capacity(2 * (self.control_graph.len() + self.data_graph.len()));
        queue.push(Cmd::StartC(self.end_cnid));
        while let Some(cmd) = queue.pop() {
            match cmd {
                Cmd::StartC(cnid) => {
                    visited_control.insert(cnid);
                    queue.push(Cmd::EndC(cnid));

                    let cn = self.control_graph.get(cnid).unwrap();
                    for dnid in cn.data_inputs() {
                        if !visited_data.contains(&dnid) {
                            queue.push(Cmd::StartD(dnid));
                        }
                    }
                    for cnid in cn.predecessors() {
                        if !visited_control.contains(&cnid) {
                            queue.push(Cmd::StartC(cnid));
                        }
                    }
                }
                Cmd::StartD(dnid) => {
                    visited_data.insert(dnid);
                    queue.push(Cmd::EndD(dnid));

                    let dn = self.data_graph.get(dnid).unwrap();
                    for dnid in dn.data_inputs() {
                        if !visited_data.contains(&dnid) {
                            queue.push(Cmd::StartD(dnid));
                        }
                    }
                    for cnid in dn.control_inputs() {
                        if !visited_control.contains(&cnid) {
                            queue.push(Cmd::StartC(cnid));
                        }
                    }
                }

                Cmd::EndC(cnid) => {
                    let cn = self.control_graph.get(cnid).unwrap();
                    writeln!(f, "  {:?} -- {:?}", cnid, cn)?;
                }
                Cmd::EndD(dnid) => {
                    let dn = self.data_graph.get(dnid).unwrap();
                    writeln!(f, "    {:?} -- {:?}", dnid, dn)?;
                }
            }
        }

        self.dump_graphviz().unwrap();

        Ok(())
    }
}
impl Program {
    fn dump_graphviz(&self) -> std::io::Result<()> {
        use std::io::Write;

        let mut out = std::fs::File::create("ssa.dot").unwrap();
        writeln!(out, "digraph {{")?;

        writeln!(out, "subgraph cluster_control {{")?;

        for (cnid, cn) in self.control_graph.iter() {
            let gvid = format!("c{}", cnid.data().as_ffi());
            let label = cn.implicit_repr();
            writeln!(out, "  {} [label={:?}];", gvid, label)?;

            for pred_cnid in cn.predecessors() {
                let pred_gvid = format!("c{}", pred_cnid.data().as_ffi());
                writeln!(out, "  {} -> {} [color=red];", pred_gvid, gvid,)?;
            }

            for dep_dnid in cn.data_inputs() {
                let dep_gvid = format!("d{}", dep_dnid.data().as_ffi());
                writeln!(out, "  {} -> {} [color=blue];", gvid, dep_gvid)?;
            }
        }
        writeln!(out, "}}")?;

        writeln!(out, "subgraph cluster_data {{")?;
        for (dnid, dn) in self.data_graph.iter() {
            let gvid = format!("d{}", dnid.data().as_ffi());
            let label = dn.implicit_repr();
            writeln!(out, "  {} [label={:?}];", gvid, label)?;

            for dep_cnid in dn.control_inputs() {
                let pred_gvid = format!("c{}", dep_cnid.data().as_ffi());
                writeln!(out, "  {} -> {} [color=red];", pred_gvid, gvid,)?;
            }

            for dep_dnid in dn.data_inputs() {
                let dep_gvid = format!("d{}", dep_dnid.data().as_ffi());
                writeln!(out, "  {} -> {} [color=blue];", gvid, dep_gvid)?;
            }
        }
        writeln!(out, "}}")?;

        writeln!(out, "}}")
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

    let mut var_maps = BlockRegMat::for_program(program, &cfg, None);

    let mut cnid_of_bid = cfg::BlockMap::new_with(&cfg, |_| {
        control_graph.insert(ControlNode::Merge { preds: Vec::new() })
    });

    /// Temporary data structure, which is going to be translated into a Phi
    /// node when the block it belongs to is processed
    ///
    /// Specific to a block.
    #[derive(Clone)]
    struct PhiNote {
        pred: ControlNID,
        reg: mil::Reg,
        value: DataNID,
    }
    // relying on the fact that Vec::new does not actually allocate any memory until the first insertion, so this is cheap
    let mut phis: cfg::BlockMap<Vec<PhiNote>> = cfg::BlockMap::new(Vec::new(), cfg.block_count());

    let mut ends = Vec::new();

    for bid in cfg.block_ids_rpo() {
        // Map of mil reg -> ssa data node ID
        //
        // - starts with the state saved at the end of the immediate dominator, if any;
        // - updated as we scan the block insn by insn;
        // - gets reused as the initial mapping by immediately dominated blocks
        if let Some(imm_dom_bid) = cfg.dom_tree().parent_of(bid) {
            // copy from imm dom
            for reg in program.regs_iter() {
                var_maps.set(bid, reg, *var_maps.get(imm_dom_bid, reg));
            }
        }

        // each block starts with a control node
        //
        // some MIL instructions 'cut' the control flow with another control node
        let cnid = cnid_of_bid[bid];

        // Add phi nodes as necessary in this block
        {
            let notes = &mut phis[bid];
            notes.sort_by_key(|note| note.reg);

            for chunk in notes.chunk_by(|a, b| a.reg == b.reg) {
                // all PhiNote's in this chunk have the same reg
                let reg = chunk.first().unwrap().reg;

                let values = chunk
                    .iter()
                    .cloned()
                    .map(|note| (note.pred, note.value))
                    .collect();

                let dnid = data_graph.insert(DataNode::Phi(Phi {
                    merge_nid: cnid,
                    values,
                }));
                var_maps.set(bid, reg, Some(dnid));
            }
        }

        let insns = program.slice(cfg.insns_ndx_range(bid)).expect("broken cfg");
        // The current control node ID.
        //
        // Starts with the merge block that *starts* the control block.
        // As we encounter instructions that are translated with a new control
        // node, the new control nod ID is set into `cnid`.
        // At the end, cnid is written to cnid_of_bid to allow successors to link to the
        // right control block.
        let mut cnid = cnid;
        let mut iter = insns.iter().peekable();
        while let Some((dest, insn)) = iter.next() {
            let dnid_of_reg = |reg: mil::Reg| var_maps.get(bid, reg).expect("unmapped reg");

            let data_node = match insn.get() {
                mil::Insn::True => Some(DataNode::ConstBool(true)),
                mil::Insn::False => Some(DataNode::ConstBool(false)),
                mil::Insn::Const1(value) => Some(DataNode::ConstInt {
                    size: ValueSize(1),
                    value: value as i64,
                }),
                mil::Insn::Const2(value) => Some(DataNode::ConstInt {
                    size: ValueSize(2),
                    value: value as i64,
                }),
                mil::Insn::Const4(value) => Some(DataNode::ConstInt {
                    size: ValueSize(4),
                    value: value as i64,
                }),
                mil::Insn::Const8(value) => Some(DataNode::ConstInt {
                    size: ValueSize(8),
                    value: value as i64,
                }),
                mil::Insn::Get(arg) => {
                    var_maps.set(bid, dest.get(), Some(dnid_of_reg(arg)));
                    continue;
                }
                mil::Insn::Part { src, offset, size } => Some(DataNode::Part {
                    src: dnid_of_reg(src),
                    offset,
                    size,
                }),
                mil::Insn::Concat { lo, hi } => Some(DataNode::Concat {
                    lo: dnid_of_reg(lo),
                    hi: dnid_of_reg(hi),
                }),
                mil::Insn::Widen1_2(arg) => Some(DataNode::Widen {
                    input: dnid_of_reg(arg),
                    out_size: ValueSize(2),
                }),
                mil::Insn::Widen1_4(arg) => Some(DataNode::Widen {
                    input: dnid_of_reg(arg),
                    out_size: ValueSize(4),
                }),
                mil::Insn::Widen1_8(arg) => Some(DataNode::Widen {
                    input: dnid_of_reg(arg),
                    out_size: ValueSize(8),
                }),
                mil::Insn::Widen2_4(arg) => Some(DataNode::Widen {
                    input: dnid_of_reg(arg),
                    out_size: ValueSize(4),
                }),
                mil::Insn::Widen2_8(arg) => Some(DataNode::Widen {
                    input: dnid_of_reg(arg),
                    out_size: ValueSize(8),
                }),
                mil::Insn::Widen4_8(arg) => Some(DataNode::Widen {
                    input: dnid_of_reg(arg),
                    out_size: ValueSize(8),
                }),
                mil::Insn::Arith1(arith_op, a, b)
                | mil::Insn::Arith2(arith_op, a, b)
                | mil::Insn::Arith4(arith_op, a, b)
                | mil::Insn::Arith8(arith_op, a, b) => {
                    Some(DataNode::Arith(arith_op, dnid_of_reg(a), dnid_of_reg(b)))
                }
                mil::Insn::ArithK1(arith_op, reg, k)
                | mil::Insn::ArithK2(arith_op, reg, k)
                | mil::Insn::ArithK4(arith_op, reg, k)
                | mil::Insn::ArithK8(arith_op, reg, k) => {
                    Some(DataNode::ArithK(arith_op, dnid_of_reg(reg), k))
                }
                mil::Insn::Cmp(op, a, b) => Some(DataNode::Cmp(op, dnid_of_reg(a), dnid_of_reg(b))),
                mil::Insn::Bool(op, a, b) => {
                    Some(DataNode::Bool(op, dnid_of_reg(a), dnid_of_reg(b)))
                }
                mil::Insn::Not(x) => Some(DataNode::Not(dnid_of_reg(x))),

                mil::Insn::Call(callee) => {
                    let mut args = Vec::new();
                    while let Some((_, insn)) = iter.peek() {
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
                    panic!("compiler bug or malformed mil: CArg node should be adjacent to the corresponding call, and processed in the context of the Call insn")
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

                    // make sure that the target is already correctly registered
                    // in the cfg; we're going to add the edge in the SoN anyway
                    // later
                    assert!(cfg.direct().successors(bid).contains(&target_bid));

                    cnid = control_graph.insert(ControlNode::Jump { pred: cnid });
                    None
                }
                mil::Insn::JmpIf { cond, target } => {
                    let branch_cnid = control_graph.insert(ControlNode::Branch {
                        pred: cnid,
                        cond: dnid_of_reg(cond),
                    });

                    let cons_cnid = control_graph.insert(ControlNode::IfTrue(branch_cnid));

                    let target_bid = cfg
                        .block_starting_at(target)
                        .expect("inconsistent mil/cfg: no block at jmpif target");
                    let target_cnid = cnid_of_bid[target_bid];
                    add_predecessor(&mut control_graph, cons_cnid, target_cnid);

                    cnid = control_graph.insert(ControlNode::IfFalse(branch_cnid));

                    None
                }
                mil::Insn::JmpExt(value) => {
                    let addr = data_graph.insert(DataNode::ConstInt {
                        size: ValueSize(8),
                        value: value as i64,
                    });
                    cnid = control_graph.insert(ControlNode::JumpIndirect { pred: cnid, addr });
                    None
                }
                mil::Insn::JmpExtIf { cond, addr } => {
                    let addr = data_graph.insert(DataNode::ConstInt {
                        size: ValueSize(8),
                        value: addr as i64,
                    });
                    cnid = control_graph.insert(ControlNode::BranchIndirect {
                        pred: cnid,
                        addr,
                        cond: dnid_of_reg(cond),
                    });
                    None
                }
                mil::Insn::TODO(descr) => {
                    cnid = control_graph.insert(ControlNode::TODO {
                        pred: cnid,
                        label: descr,
                    });
                    None
                }
                mil::Insn::LoadMem1(reg)
                | mil::Insn::LoadMem2(reg)
                | mil::Insn::LoadMem4(reg)
                | mil::Insn::LoadMem8(reg) => {
                    cnid = control_graph.insert(ControlNode::Load {
                        pred: cnid,
                        addr: dnid_of_reg(reg),
                    });
                    Some(DataNode::LoadedValueOf(cnid))
                }
                mil::Insn::StoreMem(addr, value) => {
                    cnid = control_graph.insert(ControlNode::Store {
                        pred: cnid,
                        addr: dnid_of_reg(addr),
                        value: dnid_of_reg(value),
                    });
                    None
                }
                mil::Insn::OverflowOf(reg) => Some(DataNode::OverflowOf(dnid_of_reg(reg))),
                mil::Insn::CarryOf(reg) => Some(DataNode::CarryOf(dnid_of_reg(reg))),
                mil::Insn::SignOf(reg) => Some(DataNode::SignOf(dnid_of_reg(reg))),
                mil::Insn::IsZero(reg) => Some(DataNode::IsZero(dnid_of_reg(reg))),
                mil::Insn::Parity(reg) => Some(DataNode::Parity(dnid_of_reg(reg))),
                mil::Insn::Undefined => Some(DataNode::Undefined),
                mil::Insn::Ancestral(ancestral_name) => Some(DataNode::Ancestral(ancestral_name)),
                mil::Insn::Phi | mil::Insn::PhiBool | mil::Insn::PhiArg(_) => {
                    panic!("phi nodes managed differently in this new impl")
                }
            };

            if let Some(data_node) = data_node {
                // TODO remove duplicates with a hashmap, save some space
                let dnid = data_graph.insert(data_node);
                var_maps.set(bid, dest.get(), Some(dnid));
            }
        }

        // Block finished:
        // - Save the state of the variable mapping, for successors
        cnid_of_bid[bid] = cnid;

        // - Update phi notes in successors (will be translated to phi nodes)
        for &succ in cfg.direct().successors(bid) {
            add_predecessor(&mut control_graph, cnid, cnid_of_bid[succ]);

            for reg in program.regs_iter() {
                if *phis_set.get(succ, reg) {
                    let value = var_maps.get(bid, reg).unwrap();
                    // a straightforward `push` is fine: each block is guaranteed to be processed just once,
                    // and each reg/bid combination is guaranteed to end up in the vec at most once.
                    phis[succ].push(PhiNote {
                        pred: cnid,
                        reg,
                        value,
                    });
                }
            }
        }
    }

    let final_merge = control_graph.insert(ControlNode::Merge {
        preds: ends.iter().map(|(cnid, _)| *cnid).collect(),
    });
    let final_phi = data_graph.insert(DataNode::Phi(Phi {
        merge_nid: final_merge,
        values: ends,
    }));
    let end_cnid = control_graph.insert(ControlNode::End {
        pred: final_merge,
        ret: final_phi,
    });

    let mut ssa = Program {
        control_graph,
        data_graph,
        start_cnid: cnid_of_bid[cfg.entry_block_id()],
        end_cnid,
    };
    ssa.assert_invariants();
    eliminate_dead_code(&mut ssa);
    ssa
}

fn add_predecessor(
    control_graph: &mut SlotMap<ControlNID, ControlNode>,
    pred_cnid: ControlNID,
    target_cnid: ControlNID,
) {
    add_predecessors(control_graph, target_cnid, std::iter::once(pred_cnid));
}

fn add_predecessors(
    control_graph: &mut SlotMap<ControlNID, ControlNode>,
    target_cnid: ControlNID,
    pred_cnids: impl Iterator<Item = ControlNID>,
) {
    let target_cn = control_graph.get_mut(target_cnid).unwrap();
    if let ControlNode::Merge { preds } = target_cn {
        preds.extend(pred_cnids);
    } else {
        // add a new Merge node that joins the target CN and the
        // current CN; then swap it with the target CN (the IDs
        // remain valid and the new Merge CN takes place of the
        // old one)
        let mut preds: Vec<_> = pred_cnids.collect();
        let new_cnid = control_graph.insert_with_key(move |new_cnid| {
            preds.push(new_cnid);
            ControlNode::Merge { preds }
        });
        let [new_cn, target_cn] = control_graph
            .get_disjoint_mut([new_cnid, target_cnid])
            .unwrap();
        std::mem::swap(new_cn, target_cn);
    }
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

// TODO restore this function
pub fn eliminate_dead_code(prog: &mut Program) {
    enum NID {
        C(ControlNID),
        D(DataNID),
    }

    let mut used_control = slotmap::SecondaryMap::with_capacity(prog.control_graph.len());
    let mut used_data = slotmap::SecondaryMap::with_capacity(prog.data_graph.len());

    let mut queue = vec![NID::C(prog.end_cnid)];

    while let Some(nid) = queue.pop() {
        match nid {
            NID::C(cnid) => {
                if used_control.insert(cnid, ()).is_some() {
                    continue;
                }
                let cn = prog.control_graph.get(cnid).unwrap();
                for dnid in cn.data_inputs() {
                    queue.push(NID::D(dnid));
                }
                for cnid in cn.predecessors() {
                    queue.push(NID::C(cnid));
                }
            }
            NID::D(dnid) => {
                if used_data.insert(dnid, ()).is_some() {
                    continue;
                }
                let dn = prog.data_graph.get(dnid).unwrap();
                for dnid in dn.data_inputs() {
                    queue.push(NID::D(dnid));
                }
                for cnid in dn.control_inputs() {
                    queue.push(NID::C(cnid));
                }
            }
        }
    }

    prog.control_graph
        .retain(|cnid, _| used_control.contains_key(cnid));
    prog.data_graph
        .retain(|dnid, _| used_data.contains_key(dnid));
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
        let prog = super::mil_to_ssa(&prog);
        insta::assert_debug_snapshot!(prog);
    }
}
