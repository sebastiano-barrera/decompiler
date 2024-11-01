use std::ops::Range;
use std::{collections::HashMap, rc::Rc};

use smallvec::SmallVec;

use crate::{
    cfg::{self, BlockID},
    mil,
    pp::PrettyPrinter,
    ssa,
};

use self::nodeset::{NodeID, NodeSet};

#[derive(Debug)]
pub struct Ast {
    root_thunk: ThunkID,
    nodes: NodeSet,
    thunks: HashMap<ThunkID, Thunk>,
}

#[derive(Debug)]
struct Thunk {
    body: NodeID,
    // there is one param per phi node
    params: SmallVec<[Ident; 2]>,
}

#[derive(Debug, PartialEq, Eq)]
struct ThunkArgs {
    // thunk's parameters (copied)
    names: SmallVec<[Ident; 2]>,
    // values, in lockstep with `params`
    values: SmallVec<[NodeID; 2]>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Ident(Rc<String>);

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ThunkID(Rc<String>);

#[derive(Debug, PartialEq, Eq)]
enum Node {
    Seq(Seq),
    If {
        cond: NodeID,
        cons: NodeID,
        alt: NodeID,
    },
    Let {
        name: Ident,
        value: NodeID,
    },
    LetMut {
        name: Ident,
        value: NodeID,
    },
    Ref(Ident),
    ContinueToThunk(ThunkID, ThunkArgs),
    ContinueToExtern(u64),

    Labeled(ThunkID, NodeID),

    Const1(u8),
    Const2(u16),
    Const4(u32),
    Const8(u64),

    L1(NodeID),
    L2(NodeID),
    L4(NodeID),

    WithL1(NodeID, NodeID),
    WithL2(NodeID, NodeID),
    WithL4(NodeID, NodeID),

    Bin {
        op: BinOp,
        a: NodeID,
        b: NodeID,
    },
    Cmp {
        op: CmpOp,
        a: NodeID,
        b: NodeID,
    },
    Not(NodeID),

    Call(NodeID, SmallVec<[NodeID; 4]>),
    Return(NodeID),
    TODO(&'static str),
    Phi(SmallVec<[(u16, NodeID); 2]>),

    LoadMem1(NodeID),
    LoadMem2(NodeID),
    LoadMem4(NodeID),
    LoadMem8(NodeID),
    StoreMem1(NodeID, NodeID),
    StoreMem2(NodeID, NodeID),
    StoreMem4(NodeID, NodeID),
    StoreMem8(NodeID, NodeID),

    OverflowOf(NodeID),
    CarryOf(NodeID),
    SignOf(NodeID),
    IsZero(NodeID),
    Parity(NodeID),
    StackBot,
    Undefined,
    Nop,
}

impl Node {
    fn as_const(&self) -> Option<u64> {
        match self {
            Node::Const1(x) => Some(*x as u64),
            Node::Const2(x) => Some(*x as u64),
            Node::Const4(x) => Some(*x as u64),
            Node::Const8(x) => Some(*x),
            _ => None,
        }
    }
}

type Seq = SmallVec<[NodeID; 2]>;

#[derive(PartialEq, Eq, Debug)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Shl,
    Shr,
    BitAnd,
    BitOr,
}
impl BinOp {
    fn precedence(&self) -> u8 {
        // higher number means higher precedence
        match self {
            BinOp::Add => 0,
            BinOp::Sub => 0,
            BinOp::Mul => 1,
            BinOp::Div => 1,
            BinOp::Shl => 2,
            BinOp::Shr => 2,
            BinOp::BitAnd => 3,
            BinOp::BitOr => 3,
        }
    }
}

#[derive(PartialEq, Eq, Debug)]
enum CmpOp {
    EQ,
    LT,
    LE,
    GE,
    GT,
}

pub fn ssa_to_ast(ssa: &ssa::Program, _pat_sel: &PatternSel) -> Ast {
    // the set of blocks in the CFG
    // is transformed
    // into a set of Thunk-s
    //
    // each thunk can result from one or more blocks, merged.

    let mut builder = Builder::init_to_ast(ssa);

    // TODO Remove thunks altogether
    builder.compile_new_thunk(cfg::ENTRY_BID);

    // TODO: inline 1-predecessor continuations (?)

    builder.finish()
}

fn mark_path_inline(builder: &mut Builder, path: &Path) {
    for (&a, &b) in path.iter().zip(&path[1..]) {
        builder.mark_edge_inline(a, b);
    }
}

struct Builder<'a> {
    ssa: &'a ssa::Program,
    name_of_value: HashMap<mil::Index, Ident>,
    thunk_id_of_block: cfg::BlockMap<ThunkID>,
    thunks: HashMap<ThunkID, Thunk>,
    nodes: NodeSet,
    edge_flags: HashMap<(BlockID, BlockID), EdgeFlags>,
    blocks_compiling: Vec<BlockID>,
    marks: cfg::BlockMap<BlockMark>,
}

// TODO replace with a proper bitmask
#[derive(Clone, Copy, Default)]
struct EdgeFlags {
    is_loop: bool,
    is_inline: bool,
}

#[derive(Clone, Debug)]
enum BlockMark {
    Simple,
    LoopHead { loop_enter_succ_ndx: u8 },
}

fn mark_blocks(cfg: &cfg::Graph) -> cfg::BlockMap<BlockMark> {
    let mut mark = cfg::BlockMap::new(BlockMark::Simple, cfg.block_count());

    // in_path[B] == Some(S)
    //   <==> block B is in the currently walked path, via successor with index S
    let mut in_path = cfg::BlockMap::new(None, cfg.block_count());
    let mut path = Vec::with_capacity(cfg.block_count() / 2);

    enum Cmd {
        Start((cfg::BlockID, usize)),
        End(cfg::BlockID),
    }
    let mut queue = Vec::with_capacity(cfg.block_count() / 2);
    queue.push(Cmd::Start((cfg::ENTRY_BID, 0)));

    while let Some(cmd) = queue.pop() {
        match cmd {
            Cmd::Start((bid, succ_ndx)) => {
                assert!(in_path[bid].is_none());
                path.push(bid);
                let succ_ndx: u8 = succ_ndx.try_into().unwrap();
                in_path[bid] = Some(succ_ndx);

                // cycles: one of the current block's successors S dominates B
                for (succ_ndx, &succ) in cfg.direct()[bid].iter().enumerate() {
                    if let Some(loop_enter_succ_ndx) = in_path[succ] {
                        mark[succ] = BlockMark::LoopHead {
                            loop_enter_succ_ndx,
                        };
                    } else {
                        queue.push(Cmd::End(succ));
                        queue.push(Cmd::Start((succ, succ_ndx)));
                    }
                }
            }
            Cmd::End(bid) => {
                assert!(in_path[bid].is_some());
                in_path[bid] = None;
            }
        }
    }

    mark
}

impl<'a> Builder<'a> {
    fn init_to_ast(ssa: &'a ssa::Program) -> Builder<'a> {
        let marks = mark_blocks(ssa.cfg());

        let name_of_value = (0..ssa.len())
            .filter_map(|ndx| {
                let insn = ssa.get(ndx).unwrap().insn;
                let is_named = match insn {
                    mil::Insn::Const1(_)
                    | mil::Insn::Const2(_)
                    | mil::Insn::Const4(_)
                    | mil::Insn::Const8(_) => false,
                    mil::Insn::LoadMem1(_)
                    | mil::Insn::LoadMem2(_)
                    | mil::Insn::LoadMem4(_)
                    | mil::Insn::LoadMem8(_) => false,
                    mil::Insn::Phi | mil::Insn::Call { .. } => true,
                    _ => ssa.readers_count(mil::Reg(ndx)) > 1,
                };
                if is_named {
                    let name = Ident(Rc::new(format!("v{}", ndx)));
                    Some((ndx, name))
                } else {
                    None
                }
            })
            .collect();

        // Thunk IDs are all preassigned so that jumps/continues can always be resolved, regardless
        // of the order in which blocks are processed / thunks generated.
        // (conservatively = even for thunks that end up not being generated)
        let cfg = ssa.cfg();
        let thunk_id_of_block =
            cfg::BlockMap::new_with(cfg, |bid| ThunkID(Rc::new(format!("T{}", bid.as_number()))));

        Builder {
            ssa,
            name_of_value,
            thunk_id_of_block,
            thunks: HashMap::new(),
            nodes: NodeSet::new(),
            edge_flags: HashMap::new(),
            blocks_compiling: Vec::new(),
            marks,
        }
    }

    fn finish(mut self) -> Ast {
        let root_tid = self.thunk_id_of_block[cfg::ENTRY_BID].clone();
        apply_peephole_substitutions(&mut self.nodes);
        Ast {
            root_thunk: root_tid,
            nodes: self.nodes,
            thunks: self.thunks,
        }
    }

    fn add_node(&mut self, node: Node) -> NodeID {
        self.nodes.add(node)
    }

    fn compile_new_thunk(&mut self, start_bid: cfg::BlockID) -> ThunkID {
        let mut seq = Seq::new();

        let params = self.thunk_params_of_block_phis(start_bid);
        self.compile_thunk_body(start_bid, &mut seq);

        let thunk = Thunk {
            params,
            body: self.add_node(Node::Seq(seq)),
        };

        let lbl = self.thunk_id_of_block[start_bid].clone();
        self.thunks.insert(lbl.clone(), thunk);
        lbl
    }

    fn thunk_params_of_block_phis(&mut self, bid: BlockID) -> SmallVec<[Ident; 2]> {
        let phis = self.ssa.block_phi(bid);
        (0..phis.phi_count())
            .map(|phi_ndx| phis.node_ndx(phi_ndx))
            .filter(|ndx| self.ssa.is_alive(*ndx))
            .map(|ndx| {
                self.name_of_value
                    .get(&ndx)
                    .expect("unnamed phi node!")
                    .clone()
            })
            .collect()
    }

    fn compile_thunk_body(&mut self, bid: BlockID, out_seq: &mut Seq) {
        assert!(!self.blocks_compiling.contains(&bid));
        self.blocks_compiling.push(bid);

        let cfg = &self.ssa.cfg();

        let nor_ndxs = cfg.insns_ndx_range(bid);

        self.compile_seq(&nor_ndxs, out_seq);

        let block_cont = cfg.block_cont(bid);
        match block_cont {
            cfg::BlockCont::End => {
                // all done!
            }
            cfg::BlockCont::Jmp((pred_ndx, tgt)) => {
                self.compile_continue(tgt, pred_ndx, out_seq);
            }
            cfg::BlockCont::Alt {
                straight: (neg_pred_ndx, neg_bid),
                side: (pos_pred_ndx, pos_bid),
            } => {
                assert!(
                    nor_ndxs.len() > 0,
                    "block with BlockCont::Alt continuation must have at least 1 insn"
                );
                let last_ndx = nor_ndxs.end - 1;
                let last_item = self.ssa.get(last_ndx).unwrap();

                let cond = match &last_item.insn {
                    mil::Insn::JmpIf { cond, target: _ } => cond,
                    _ => panic!("block with BlockCont::Alt continuation must end with a JmpIf"),
                };
                let cond = self.add_node_of_value(cond.0);
                let cons = {
                    let mut seq = Seq::new();
                    self.compile_continue(neg_bid, neg_pred_ndx, &mut seq);
                    self.add_node(Node::Seq(seq))
                };
                let alt = {
                    let mut seq = Seq::new();
                    self.compile_continue(pos_bid, pos_pred_ndx, &mut seq);
                    self.add_node(Node::Seq(seq))
                };

                out_seq.push(self.add_node(Node::If { cond, cons, alt }));
            }
        };

        let succs = &cfg.direct()[bid];
        for &dominated_bid in cfg.dom_tree().children(bid) {
            if succs.contains(&dominated_bid) {
                continue;
            }

            // dominated_bid is an "extra":
            // it's dominated by the current block, but not a direct successor
            // it's indirect successor that still inherits the current block's scope
            let node = self.compile_to_labeled(dominated_bid);
            out_seq.push(self.add_node(node));
        }

        let check = self.blocks_compiling.pop();
        assert_eq!(check, Some(bid));
    }

    fn compile_seq(&mut self, nor_ndxs: &Range<mil::Index>, out_seq: &mut Seq) {
        out_seq.extend(
            nor_ndxs
                .clone()
                .map(|ndx| (ndx, self.ssa.get(ndx).unwrap()))
                .filter_map(|(ndx, iv)| {
                    let name = self.name_of_value.get(&iv.dest.0).cloned();

                    if !iv.insn.has_side_effects() && name.is_none() {
                        return None;
                    }

                    let node = self.compile_node(ndx);
                    if node == Node::Nop {
                        return None;
                    }

                    if let Some(name) = name {
                        let value = self.add_node(node);
                        return Some(self.add_node(Node::Let { name, value }));
                    };

                    Some(self.add_node(node))
                }),
        )
    }

    fn compile_continue(&mut self, target_bid: cfg::BlockID, pred_ndx: u8, out_seq: &mut Seq) {
        let args: SmallVec<_> = {
            let target_phis = self.ssa.block_phi(target_bid);
            (0..target_phis.phi_count())
                .filter(|phi_ndx| self.ssa.is_alive(target_phis.node_ndx(*phi_ndx)))
                .map(|phi_ndx| {
                    let reg = target_phis.arg(self.ssa, phi_ndx, pred_ndx.into());
                    self.add_node_of_value(reg.0)
                })
                .collect()
        };

        let cur_bid = self.blocks_compiling.last().copied().unwrap();
        let edge = self.edge(cur_bid, target_bid);

        let cfg = self.ssa.cfg();
        let nonbackedge_count = cfg.direct().nonbackedge_predecessor_count(target_bid);
        let preds_count = cfg.block_preds(target_bid).len();

        let params = self.thunk_params_of_block_phis(target_bid);
        assert_eq!(params.len(), args.len(), "inconsistent arg/param count");

        if !edge.is_loop && (edge.is_inline || nonbackedge_count == 1) {
            if preds_count > 1 {
                out_seq.extend(
                    params
                        .into_iter()
                        .zip(args.into_iter())
                        .map(|(param, arg)| {
                            self.add_node(Node::LetMut {
                                name: param,
                                value: arg,
                            })
                        }),
                );

                let node = self.compile_to_labeled(target_bid);
                out_seq.push(self.add_node(node));
            } else {
                assert_eq!(params.len(), 0);
                self.compile_thunk_body(target_bid, out_seq);
            }
        } else {
            let thunk_id = self.thunk_id_of_block[target_bid].clone();
            let node = Node::ContinueToThunk(
                thunk_id,
                ThunkArgs {
                    names: params,
                    values: args,
                },
            );
            out_seq.push(self.add_node(node));
        }
    }

    fn compile_to_labeled(&mut self, target_bid: BlockID) -> Node {
        let mut inner_seq = SmallVec::new();
        self.compile_thunk_body(target_bid, &mut inner_seq);
        let inner_seq = self.add_node(Node::Seq(inner_seq));
        let label = self.thunk_id_of_block[target_bid].clone();
        Node::Labeled(label, inner_seq)
    }

    fn compile_node(&mut self, start_ndx: mil::Index) -> Node {
        use mil::Insn;

        let iv = self.ssa.get(start_ndx).unwrap();
        match iv.insn {
            Insn::Call { callee, arg0 } => {
                let callee = self.add_node_of_value(callee.0);
                let mut args = SmallVec::new();
                let mut arg = arg0;
                loop {
                    let arg_insn = self.ssa.get(arg.0).unwrap();
                    match arg_insn.insn {
                        Insn::CArg { value, prev } => {
                            let arg_val = self.add_node_of_value(value.0);
                            args.push(arg_val);
                            arg = prev;
                        }
                        Insn::CArgEnd => break Node::Call(callee, args),
                        other => {
                            panic!("invalid insn in call arg chain: {:?}", other)
                        }
                    };
                }
            }
            // To be handled in ::Call
            Insn::CArgEnd | Insn::CArg { .. } => Node::Nop,
            Insn::Ret(arg) => Node::Return(self.add_node_of_value(arg.0)),
            Insn::JmpExt(target) => Node::ContinueToExtern(*target),
            Insn::JmpI(_) | Insn::Jmp(_) | Insn::JmpExtIf { .. } | Insn::JmpIf { .. } => {
                // skip.  the control fllow handling in compile_thunk shall take care of this
                Node::Nop
            }
            Insn::StoreMem1(addr, val) => {
                let addr = self.add_node_of_value(addr.0);
                let val = self.add_node_of_value(val.0);
                Node::StoreMem1(addr, val)
            }
            Insn::StoreMem2(addr, val) => {
                let addr = self.add_node_of_value(addr.0);
                let val = self.add_node_of_value(val.0);
                Node::StoreMem2(addr, val)
            }
            Insn::StoreMem4(addr, val) => {
                let addr = self.add_node_of_value(addr.0);
                let val = self.add_node_of_value(val.0);
                Node::StoreMem4(addr, val)
            }
            Insn::StoreMem8(addr, val) => {
                let addr = self.add_node_of_value(addr.0);
                let val = self.add_node_of_value(val.0);
                Node::StoreMem8(addr, val)
            }

            Insn::Const1(val) => Node::Const1(*val),
            Insn::Const2(val) => Node::Const2(*val),
            Insn::Const4(val) => Node::Const4(*val),
            Insn::Const8(val) => Node::Const8(*val),

            Insn::L1(reg) => Node::L1(self.add_node_of_value(reg.0)),
            Insn::L2(reg) => Node::L2(self.add_node_of_value(reg.0)),
            Insn::L4(reg) => Node::L4(self.add_node_of_value(reg.0)),
            Insn::Get(reg) => self.node_of_value(reg.0),

            Insn::WithL1(a, b) => {
                Node::WithL1(self.add_node_of_value(a.0), self.add_node_of_value(b.0))
            }
            Insn::WithL2(a, b) => {
                Node::WithL2(self.add_node_of_value(a.0), self.add_node_of_value(b.0))
            }
            Insn::WithL4(a, b) => {
                Node::WithL4(self.add_node_of_value(a.0), self.add_node_of_value(b.0))
            }
            Insn::Add(a, b) => {
                let a = self.add_node_of_value(a.0);
                let b = self.add_node_of_value(b.0);
                self.fold_bin(BinOp::Add, a, b)
            }
            Insn::AddK(a, k) => {
                let a = self.add_node_of_value(a.0);
                let b = self.add_node(Node::Const8(*k as u64));
                self.fold_bin(BinOp::Add, a, b)
            }
            Insn::Sub(a, b) => {
                let a = self.add_node_of_value(a.0);
                let b = self.add_node_of_value(b.0);
                self.fold_bin(BinOp::Sub, a, b)
            }
            Insn::Mul(a, b) => {
                let a = self.add_node_of_value(a.0);
                let b = self.add_node_of_value(b.0);
                self.fold_bin(BinOp::Mul, a, b)
            }
            Insn::MulK32(a, k) => {
                let a = self.add_node_of_value(a.0);
                let b = self.add_node(Node::Const8(*k as u64));
                self.fold_bin(BinOp::Mul, a, b)
            }
            Insn::Shl(a, b) => {
                let b = self.add_node_of_value(b.0);
                let a = self.add_node_of_value(a.0);
                self.fold_bin(BinOp::Shl, a, b)
            }
            Insn::BitAnd(a, b) => {
                let a = self.add_node_of_value(a.0);
                let b = self.add_node_of_value(b.0);
                self.fold_bin(BinOp::BitAnd, a, b)
            }
            Insn::BitOr(a, b) => {
                let a = self.add_node_of_value(a.0);
                let b = self.add_node_of_value(b.0);
                self.fold_bin(BinOp::BitOr, a, b)
            }
            Insn::Eq(a, b) => {
                let a = self.add_node_of_value(a.0);
                let b = self.add_node_of_value(b.0);
                Node::Cmp {
                    op: CmpOp::EQ,
                    a,
                    b,
                }
            }
            Insn::Not(x) => Node::Not(self.add_node_of_value(x.0)),
            Insn::TODO(msg) => Node::TODO(msg),
            Insn::LoadMem1(addr_reg) => Node::LoadMem1(self.add_node_of_value(addr_reg.0)),
            Insn::LoadMem2(addr_reg) => Node::LoadMem2(self.add_node_of_value(addr_reg.0)),
            Insn::LoadMem4(addr_reg) => Node::LoadMem4(self.add_node_of_value(addr_reg.0)),
            Insn::LoadMem8(addr_reg) => Node::LoadMem8(self.add_node_of_value(addr_reg.0)),

            Insn::OverflowOf(arg) => Node::OverflowOf(self.add_node_of_value(arg.0)),
            Insn::CarryOf(arg) => Node::CarryOf(self.add_node_of_value(arg.0)),
            Insn::SignOf(arg) => Node::SignOf(self.add_node_of_value(arg.0)),
            Insn::IsZero(arg) => Node::IsZero(self.add_node_of_value(arg.0)),
            Insn::Parity(arg) => Node::Parity(self.add_node_of_value(arg.0)),

            Insn::Undefined => Node::Undefined,
            Insn::Ancestral(anc) => match anc {
                mil::Ancestral::StackBot => Node::StackBot,
            },

            Insn::Phi => {
                let mut args = SmallVec::new();

                let mut pred_ndx = 0;
                while let Some(mil::InsnView {
                    insn: Insn::PhiArg(value),
                    ..
                }) = self.ssa.get(start_ndx + 1 + pred_ndx)
                {
                    let value_expr = self.add_node_of_value(value.0);
                    args.push((pred_ndx, value_expr));
                    pred_ndx += 1;
                }

                Node::Phi(args)
            }

            other => {
                if other.has_side_effects() {
                    panic!(
                        "side-effecting instruction passed to collect_expr: {:?}",
                        other
                    )
                } else {
                    panic!(
                        "invalid/forbidden instruction passed to collect_expr: {:?}",
                        other
                    )
                }
            }
        }
    }

    fn add_node_of_value(&mut self, ndx: mil::Index) -> NodeID {
        let node = self.node_of_value(ndx);
        self.add_node(node)
    }
    fn node_of_value(&mut self, ndx: mil::Index) -> Node {
        if let Some(lbl) = self.name_of_value.get(&ndx) {
            Node::Ref(lbl.clone())
        } else {
            // inline
            self.compile_node(ndx)
        }
    }

    fn edge_mut(&mut self, a: BlockID, b: BlockID) -> &mut EdgeFlags {
        self.edge_flags
            .entry((a, b))
            .or_insert_with(|| EdgeFlags::default())
    }
    fn edge(&self, a: BlockID, b: BlockID) -> EdgeFlags {
        self.edge_flags.get(&(a, b)).copied().unwrap_or_default()
    }

    fn mark_edge_loop(&mut self, a: BlockID, b: BlockID) {
        self.edge_mut(a, b).is_loop = true;
    }
    fn mark_edge_inline(&mut self, a: BlockID, b: BlockID) {
        self.edge_mut(a, b).is_inline = true;
    }

    fn fold_bin(&self, op: BinOp, a: NodeID, b: NodeID) -> Node {
        // TODO!
        Node::Bin { op, a, b }
    }
}

fn apply_peephole_substitutions(nodes: &mut NodeSet) {
    for nid in nodes.node_ids() {
        match &nodes[nid] {
            Node::CarryOf(arg) => {
                let arg = &nodes[*arg];
                if let Node::Bin {
                    op: BinOp::Sub,
                    a,
                    b,
                } = arg
                {
                    let new_node = Node::Cmp {
                        op: CmpOp::LT,
                        a: *a,
                        b: *b,
                    };
                    let new_node = nodes.add(new_node);
                    nodes.swap(new_node, nid);
                }
            }
            Node::IsZero(arg) => {
                let arg = *arg;
                let zero = nodes.add(Node::Const8(0));
                let new_node = nodes.add(Node::Cmp {
                    op: CmpOp::EQ,
                    a: arg,
                    b: zero,
                });
                nodes.swap(new_node, nid);
            }
            Node::Bin {
                op: BinOp::Sub,
                a,
                b,
            } => {
                let na = &nodes[*a];
                let nb = &nodes[*b];
                match (na.as_const(), nb.as_const()) {
                    (Some(0), _) => nodes.swap(nid, *b),
                    (_, Some(0)) => nodes.swap(nid, *a),
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

impl Ast {
    pub fn pretty_print<W: std::fmt::Write>(&self, pp: &mut PrettyPrinter<W>) -> std::fmt::Result {
        use std::fmt::Write;
        for (thid, thunk) in self.thunks.iter() {
            write!(pp, "thunk {}(", thid.0.as_str())?;
            for (ndx, param_name) in thunk.params.iter().enumerate() {
                if ndx > 0 {
                    write!(pp, ", ")?;
                }
                write!(pp, "{}", param_name.0.as_str())?;
            }
            write!(pp, ") = ")?;

            self.pretty_print_node(pp, thunk.body)?;

            writeln!(pp)?;
        }
        Ok(())
    }

    fn pretty_print_node<W: std::fmt::Write>(
        &self,
        pp: &mut PrettyPrinter<W>,
        nid: NodeID,
    ) -> std::fmt::Result {
        use std::fmt::Write;

        let node = &self.nodes[nid];

        match node {
            Node::Seq(nodes) => {
                write!(pp, "{{\n    ")?;
                pp.open_box();
                for (ndx, node) in nodes.iter().enumerate() {
                    self.pretty_print_node(pp, *node)?;
                    if ndx < nodes.len() - 1 {
                        writeln!(pp, ";")?;
                    }
                }
                pp.close_box();
                write!(pp, "\n}}")
            }

            Node::If { cond, cons, alt } => {
                write!(pp, "if ")?;
                pp.open_box();
                self.pretty_print_node(pp, *cond)?;
                pp.close_box();

                write!(pp, " ")?;
                self.pretty_print_node(pp, *cons)?;

                write!(pp, " else ")?;
                self.pretty_print_node(pp, *alt)
            }

            Node::Let { name, value } => {
                write!(pp, "let {} = ", name.0.as_str())?;
                pp.open_box();
                self.pretty_print_node(pp, *value)?;
                pp.close_box();
                Ok(())
            }
            Node::LetMut { name, value } => {
                write!(pp, "let mut {} = ", name.0.as_str())?;
                pp.open_box();
                self.pretty_print_node(pp, *value)?;
                pp.close_box();
                Ok(())
            }

            Node::Ref(ident) => write!(pp, "{}", ident.0.as_str()),

            Node::Labeled(thunk_id, node) => {
                write!(pp, "'{}: ", thunk_id.0)?;
                self.pretty_print_node(pp, *node)
            }

            Node::ContinueToThunk(thunk_id, args) => {
                write!(pp, "goto {}", thunk_id.0.as_str())?;
                let args_count = args.names.len();
                assert_eq!(args_count, args.values.len());
                match args_count {
                    0 => {}
                    1 => {
                        write!(pp, " ({} = ", args.names[0].0.as_str())?;
                        self.pretty_print_node(pp, args.values[0])?;
                        write!(pp, ")")?;
                    }
                    _ => {
                        write!(pp, " with (")?;
                        pp.open_box();
                        for (ndx, (name, arg)) in args.names.iter().zip(&args.values).enumerate() {
                            pp.open_box();
                            write!(pp, "{} = ", name.0.as_str())?;
                            self.pretty_print_node(pp, *arg)?;
                            if ndx == args_count - 1 {
                                write!(pp, ")")?;
                            } else {
                                writeln!(pp, ",")?;
                            }
                            pp.close_box();
                        }
                        pp.close_box();
                    }
                };
                Ok(())
            }
            Node::ContinueToExtern(addr) => write!(pp, "jmp extern 0x{:x}", addr),
            Node::Const1(val) => write!(pp, "{}", *val as i8),
            Node::Const2(val) => write!(pp, "{}", *val as i16),
            Node::Const4(val) => write!(pp, "{}", *val as i32),
            Node::Const8(val) => write!(pp, "{}", *val as i64),

            Node::L1(arg) => {
                self.pretty_print_node(pp, *arg)?;
                write!(pp, ".l1")
            }
            Node::L2(arg) => {
                self.pretty_print_node(pp, *arg)?;
                write!(pp, ".l2")
            }
            Node::L4(arg) => {
                self.pretty_print_node(pp, *arg)?;
                write!(pp, ".l4")
            }
            Node::WithL1(a, b) => {
                write!(pp, "(")?;
                self.pretty_print_node(pp, *a)?;
                write!(pp, " with l1 = ")?;
                self.pretty_print_node(pp, *b)?;
                write!(pp, ")")
            }
            Node::WithL2(a, b) => {
                write!(pp, "(")?;
                self.pretty_print_node(pp, *a)?;
                write!(pp, " with l2 = ")?;
                self.pretty_print_node(pp, *b)?;
                write!(pp, ")")
            }
            Node::WithL4(a, b) => {
                write!(pp, "(")?;
                self.pretty_print_node(pp, *a)?;
                write!(pp, " with l4 = ")?;
                self.pretty_print_node(pp, *b)?;
                write!(pp, ")")
            }
            Node::Bin { op, a, b } => {
                let op_s = match op {
                    BinOp::Add => "+",
                    BinOp::Sub => "-",
                    BinOp::Mul => "*",
                    BinOp::Div => "/",
                    BinOp::Shl => "<<",
                    BinOp::Shr => ">>",
                    BinOp::BitAnd => "&",
                    BinOp::BitOr => "|",
                };

                for (ndx, &arg_nid) in [a, b].into_iter().enumerate() {
                    let arg = &self.nodes[arg_nid];
                    let needs_parens = match arg {
                        Node::Bin { op: child_op, .. } => child_op.precedence() < op.precedence(),
                        _ => false,
                    };

                    if ndx > 0 {
                        write!(pp, " {} ", op_s)?;
                    }

                    if needs_parens {
                        write!(pp, "(")?;
                    }
                    pp.open_box();
                    self.pretty_print_node(pp, arg_nid)?;
                    pp.close_box();
                    if needs_parens {
                        write!(pp, ")")?;
                    }
                }
                Ok(())
            }
            Node::Cmp { op, a, b } => {
                pp.open_box();
                self.pretty_print_node(pp, *a)?;
                pp.close_box();

                let op_s = match op {
                    CmpOp::EQ => "==",
                    CmpOp::LT => "<",
                    CmpOp::LE => "<=",
                    CmpOp::GE => ">=",
                    CmpOp::GT => ">",
                };
                write!(pp, " {} ", op_s)?;

                pp.open_box();
                self.pretty_print_node(pp, *b)?;
                pp.close_box();

                Ok(())
            }

            Node::Not(arg) => {
                write!(pp, "not ")?;
                self.pretty_print_node(pp, *arg)
            }
            Node::Call(callee, args) => {
                self.pretty_print_node(pp, *callee)?;
                write!(pp, "(\n  ")?;
                pp.open_box();
                for (ndx, arg) in args.iter().enumerate() {
                    if ndx > 0 {
                        writeln!(pp)?;
                    }
                    self.pretty_print_node(pp, *arg)?;
                    write!(pp, ",")?;
                }
                pp.close_box();
                write!(pp, "\n)")
            }
            Node::Return(arg) => {
                write!(pp, "return ")?;
                self.pretty_print_node(pp, *arg)
            }
            Node::TODO(msg) => write!(pp, "<-- TODO: {} -->", msg),
            Node::Phi(phi_args) => {
                write!(pp, "phi(\n  ")?;
                pp.open_box();
                let mut first = true;
                for (pred_ndx, value) in phi_args {
                    if !first {
                        writeln!(pp)?;
                    }
                    first = false;
                    write!(pp, "from pred. {} = ", pred_ndx)?;
                    self.pretty_print_node(pp, *value)?;
                    write!(pp, ",")?;
                }
                pp.close_box();
                write!(pp, "\n)")
            }

            Node::LoadMem1(arg) => self.pp_load_mem(pp, 1, *arg),
            Node::LoadMem2(arg) => self.pp_load_mem(pp, 2, *arg),
            Node::LoadMem4(arg) => self.pp_load_mem(pp, 4, *arg),
            Node::LoadMem8(arg) => self.pp_load_mem(pp, 8, *arg),

            Node::StoreMem1(dest, val) => self.pp_store_mem(pp, 1, *dest, *val),
            Node::StoreMem2(dest, val) => self.pp_store_mem(pp, 2, *dest, *val),
            Node::StoreMem4(dest, val) => self.pp_store_mem(pp, 4, *dest, *val),
            Node::StoreMem8(dest, val) => self.pp_store_mem(pp, 8, *dest, *val),

            Node::OverflowOf(arg) => {
                write!(pp, "overflow (")?;
                self.pretty_print_node(pp, *arg)?;
                write!(pp, ")")
            }
            Node::CarryOf(arg) => {
                write!(pp, "carry (")?;
                self.pretty_print_node(pp, *arg)?;
                write!(pp, ")")
            }
            Node::SignOf(arg) => {
                write!(pp, "sign (")?;
                self.pretty_print_node(pp, *arg)?;
                write!(pp, ")")
            }
            Node::IsZero(arg) => {
                write!(pp, "is0 (")?;
                self.pretty_print_node(pp, *arg)?;
                write!(pp, ")")
            }
            Node::Parity(arg) => {
                write!(pp, "parity (")?;
                self.pretty_print_node(pp, *arg)?;
                write!(pp, ")")
            }
            Node::StackBot => write!(pp, "<stackBottom>"),
            Node::Undefined => write!(pp, "<undefined>"),
            Node::Nop => write!(pp, "nop"),
        }
    }

    fn pp_load_mem<W: std::fmt::Write>(
        &self,
        pp: &mut PrettyPrinter<W>,
        src_size: u8,
        addr: NodeID,
    ) -> std::fmt::Result {
        use std::fmt::Write;
        write!(pp, "[")?;
        self.pretty_print_node(pp, addr)?;
        write!(pp, "]:{}", src_size)
    }

    fn pp_store_mem<W: std::fmt::Write>(
        &self,
        pp: &mut PrettyPrinter<W>,
        dest_size: u8,
        dest: NodeID,
        val: NodeID,
    ) -> std::fmt::Result {
        use std::fmt::Write;
        write!(pp, "[")?;
        self.pretty_print_node(pp, dest)?;
        write!(pp, "]:{} = ", dest_size)?;
        pp.open_box();
        self.pretty_print_node(pp, val)?;
        pp.close_box();
        Ok(())
    }
}

#[derive(Debug)]
pub struct PatternSet {
    // TODO replace this with something more appropriate
    pats: Vec<Pattern>,
}
#[derive(Debug)]
struct Pattern {
    key_bid: BlockID,
    pat: Pat,
}

#[derive(Debug)]
enum Pat {
    /// One control flow path in a branch (an if-like structure): the control flow
    /// branches at `key_bid`.
    IfBranch { path: Path },

    /// A cycle.
    ///
    /// The cycle starts with the `key_bid` block, and continues with the block
    /// designated by `path`. A 1-block cycle (a block that can jump to its own
    /// start) is represented with an empty `path`.
    Cycle { path: Path },
}

type Path = SmallVec<[BlockID; 4]>;

impl PatternSet {
    pub fn available_for_block<'s>(&'s self, key_bid: BlockID) -> impl 's + Iterator<Item = usize> {
        self.pats
            .iter()
            .enumerate()
            .filter(move |(_, pat)| pat.key_bid == key_bid)
            .map(|(ndx, _)| ndx)
    }
}

pub fn search_patterns(cfg: &cfg::Graph) -> PatternSet {
    let preds = cfg.inverse();

    let mut pats = Vec::new();
    let mut in_path = cfg::BlockMap::new(false, cfg.block_count());
    let mut path = Vec::with_capacity(cfg.block_count() / 2);

    enum Cmd {
        Start(cfg::BlockID),
        End(cfg::BlockID),
    }
    let mut queue = vec![Cmd::Start(cfg::ENTRY_BID)];

    while let Some(cmd) = queue.pop() {
        match cmd {
            Cmd::Start(bid) => {
                assert!(!in_path[bid]);
                path.push(bid);
                in_path[bid] = true;

                // if branch:
                //   && the current block B has ≥ 2 predecessors
                //   && an ancestor (indirect predecessor) A of B dominates B
                //   && B inverse-dominates A
                if preds[bid].len() >= 2 {
                    assert_eq!(path.last().unwrap(), &bid);

                    for idom in cfg.dom_tree().imm_doms(bid) {
                        let tail_len = match path
                            .iter()
                            .rev()
                            .enumerate()
                            .find(|(_, &step)| step == idom)
                            .map(|(rndx, _)| rndx)
                        {
                            None => break,
                            Some(l) if l < 2 => break,
                            Some(l) => l,
                        };

                        let tail_ndx = path.len() - tail_len;
                        let branch = &path[tail_ndx..path.len() - 1];
                        pats.push(Pattern {
                            key_bid: path[tail_ndx],
                            pat: Pat::IfBranch {
                                path: branch.into(),
                            },
                        });
                    }
                }

                // cycles: one of the current block's successors S dominates B
                for &succ in &cfg.direct()[bid] {
                    if in_path[succ] {
                        let cycle = {
                            let cycle_len =
                                1 + path.iter().rev().take_while(|step| **step != succ).count();
                            assert_eq!(path[path.len() - cycle_len], succ);
                            assert!(cycle_len > 0);

                            let cy = &path[path.len() - cycle_len..];
                            assert_eq!(cy.len(), cycle_len);
                            cy
                        };

                        pats.push(Pattern {
                            key_bid: cycle[0],
                            pat: Pat::Cycle { path: cycle.into() },
                        });
                    } else {
                        queue.push(Cmd::End(succ));
                        queue.push(Cmd::Start(succ));
                    }
                }
            }
            Cmd::End(bid) => {
                assert!(in_path[bid]);
                in_path[bid] = false;
                let check = path.pop();
                assert_eq!(check, Some(bid));
            }
        }
    }
    PatternSet { pats }
}

pub struct PatternSel<'a> {
    set: &'a PatternSet,
    // each item is an index into `set`
    sel: cfg::BlockMap<Option<usize>>,
}

impl<'a> PatternSel<'a> {
    pub fn new(set: &'a PatternSet, block_count: usize) -> Self {
        let sel = cfg::BlockMap::new(None, block_count);
        PatternSel { set, sel }
    }

    pub fn set(&mut self, bid: cfg::BlockID, pat_ndx: Option<usize>) {
        if let Some(pat_ndx) = pat_ndx {
            assert_eq!(bid, self.set.pats[pat_ndx].key_bid);
        }
        self.sel[bid] = pat_ndx;
    }
}

mod nodeset {
    use super::Node;

    pub(super) struct NodeSet(Vec<Node>);
    impl NodeSet {
        pub(super) fn add(&mut self, node: Node) -> NodeID {
            let new_ndx = self.0.len().try_into().unwrap();
            self.0.push(node);
            NodeID(new_ndx)
        }

        pub(super) fn new() -> NodeSet {
            NodeSet(Vec::new())
        }

        pub(super) fn node_ids(&self) -> impl Iterator<Item = NodeID> {
            (0..self.0.len()).map(|ndx| NodeID(ndx.try_into().unwrap()))
        }

        pub(super) fn iter(&self) -> impl Iterator<Item = (NodeID, &Node)> {
            self.node_ids().map(|nid| (nid, &self[nid]))
        }

        pub(crate) fn swap(&mut self, nid_a: NodeID, nid_b: NodeID) {
            let ndx_a = nid_a.0 as usize;
            let ndx_b = nid_b.0 as usize;
            self.0.swap(ndx_a, ndx_b);
        }
    }
    impl std::ops::Index<NodeID> for NodeSet {
        type Output = Node;

        fn index(&self, index: NodeID) -> &Self::Output {
            &self.0[index.0 as usize]
        }
    }

    #[derive(PartialEq, Eq, Clone, Copy)]
    pub(super) struct NodeID(u32);

    impl std::fmt::Debug for NodeSet {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            for (ndx, node) in self.0.iter().enumerate() {
                let nid = NodeID(ndx.try_into().unwrap());
                write!(f, "{:5?} = {:?}", nid, node)?;
            }
            Ok(())
        }
    }

    impl std::fmt::Debug for NodeID {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "n{}", self.0)
        }
    }
}
