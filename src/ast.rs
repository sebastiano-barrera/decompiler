use std::ops::Range;
use std::rc::Rc;

use std::{collections::HashMap, sync::Arc};

use smallvec::SmallVec;

use crate::{
    cfg::{self, BlockID},
    mil,
    pp::PP,
    ssa, ty,
};

use self::nodeset::{NodeID, NodeSet};

#[derive(Debug)]
pub struct Ast {
    root_nid: NodeID,
    nodes: NodeSet,
}

#[derive(Debug, PartialEq, Eq)]
struct ContinueArgs {
    // thunk's parameters (copied)
    names: SmallVec<[Ident; 2]>,
    // values, in lockstep with `params`
    values: SmallVec<[NodeID; 2]>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Ident(Rc<String>);

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Label(Rc<String>);

#[derive(Debug, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
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
    ContinueToLabel(Label, ContinueArgs),
    ContinueToExtern(u64),

    Labeled {
        label: Label,
        body: NodeID,
        // there is one param per phi node (from the block that generated the label)
        params: SmallVec<[Ident; 2]>,
    },

    Const1(i8),
    Const2(i16),
    Const4(i32),
    Const8(i64),
    True,
    False,

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
    StoreMem {
        loc: NodeID,
        value: NodeID,
    },

    RawDeref {
        addr: NodeID,
        size: u8,
    },
    MemberAccess {
        base: NodeID,
        // TODO replace with smallvec like in `ty`?
        path: Vec<Arc<String>>,
    },

    OverflowOf(NodeID),
    CarryOf(NodeID),
    SignOf(NodeID),
    IsZero(NodeID),
    Parity(NodeID),
    Undefined,
    Nop,
    Pre(&'static str),
    Widen {
        from_sz: u8,
        to_sz: u8,
        reg: NodeID,
    },
}

impl Node {
    fn as_const(&self) -> Option<i64> {
        match self {
            Node::Const1(x) => Some(*x as i64),
            Node::Const2(x) => Some(*x as i64),
            Node::Const4(x) => Some(*x as i64),
            Node::Const8(x) => Some(*x),
            _ => None,
        }
    }
}

type Seq = SmallVec<[NodeID; 2]>;

#[derive(PartialEq, Eq, Debug)]
#[allow(dead_code)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Shl,
    Shr,
    BitAnd,
    BitOr,
    BitXor,
    BoolAnd,
    BoolOr,
}
impl From<mil::ArithOp> for BinOp {
    fn from(value: mil::ArithOp) -> Self {
        match value {
            mil::ArithOp::Add => BinOp::Add,
            mil::ArithOp::Sub => BinOp::Sub,
            mil::ArithOp::Mul => BinOp::Mul,
            mil::ArithOp::Shl => BinOp::Shl,
            mil::ArithOp::BitAnd => BinOp::BitAnd,
            mil::ArithOp::BitOr => BinOp::BitOr,
            mil::ArithOp::BitXor => BinOp::BitXor,
        }
    }
}
impl BinOp {
    fn precedence(&self) -> u8 {
        // higher number means higher precedence
        match self {
            BinOp::Add => 1,
            BinOp::Sub => 1,
            BinOp::Mul => 2,
            BinOp::Div => 2,
            BinOp::Shl => 3,
            BinOp::Shr => 3,
            BinOp::BitAnd => 4,
            BinOp::BitOr => 4,
            BinOp::BitXor => 4,
            BinOp::BoolAnd => 0,
            BinOp::BoolOr => 0,
        }
    }
}

#[derive(PartialEq, Eq, Debug)]
#[allow(dead_code)]
enum CmpOp {
    EQ,
    LT,
    LE,
    GE,
    GT,
}

impl From<mil::CmpOp> for CmpOp {
    fn from(value: mil::CmpOp) -> Self {
        match value {
            mil::CmpOp::EQ => CmpOp::EQ,
            mil::CmpOp::LT => CmpOp::LT,
        }
    }
}

pub struct Builder<'a> {
    ssa: &'a ssa::Program,
    name_of_value: HashMap<mil::Reg, Ident>,
    label_of_block: cfg::BlockMap<Label>,
    nodes: NodeSet,
    blocks_compiling: Vec<BlockID>,
    types: Option<&'a ty::TypeSet>,
}

impl<'a> Builder<'a> {
    pub fn new(ssa: &'a ssa::Program) -> Builder<'a> {
        let name_of_value = ssa
            .insns_unordered()
            .filter_map(|iv| {
                // TODO Use Insn::has_side_effect, Insn::result_type
                let is_named = match iv.insn.get() {
                    mil::Insn::Const1(_)
                    | mil::Insn::Const2(_)
                    | mil::Insn::Const4(_)
                    | mil::Insn::Const8(_)
                    | mil::Insn::LoadMem1(_)
                    | mil::Insn::LoadMem2(_)
                    | mil::Insn::LoadMem4(_)
                    | mil::Insn::LoadMem8(_)
                    // ancestral values are akin to (named) consts
                    | mil::Insn::Ancestral(_) => false,
                    mil::Insn::Phi1
                    | mil::Insn::Phi2
                    | mil::Insn::Phi4
                    | mil::Insn::Phi8
                    | mil::Insn::Call { .. } => true,
                    _ => ssa.readers_count(iv.dest.get()) > 1,
                };
                if is_named {
                    let name = Ident(Rc::new(format!("{:?}", iv.dest.get())));
                    Some((iv.dest.get(), name))
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
            cfg::BlockMap::new_with(cfg, |bid| Label(Rc::new(format!("T{}", bid.as_number()))));

        Builder {
            ssa,
            name_of_value,
            label_of_block: thunk_id_of_block,
            nodes: NodeSet::new(),
            blocks_compiling: Vec::new(),
            types: None,
        }
    }

    pub fn use_type_set(&mut self, types: &'a ty::TypeSet) {
        self.types = Some(types);
    }

    pub fn compile(self) -> Ast {
        let start_bid = self.ssa.cfg().entry_block_id();
        self.compile_with_entry(start_bid)
    }
}

impl<'a> Builder<'a> {
    fn compile_with_entry(mut self, start_bid: cfg::BlockID) -> Ast {
        let main_node = self.compile_to_labeled(start_bid);
        let root_nid = self.add_node(main_node);
        apply_peephole_substitutions(&mut self.nodes);
        Ast {
            root_nid,
            nodes: self.nodes,
        }
    }

    fn add_node(&mut self, node: Node) -> NodeID {
        match node {
            Node::Seq(xs) if xs.len() == 1 => xs[0],
            other => self.nodes.add(other),
        }
    }

    fn compile_to_labeled(&mut self, start_bid: cfg::BlockID) -> Node {
        let mut seq = Seq::new();

        self.compile_thunk_body(start_bid, &mut seq);

        Node::Labeled {
            label: self.label_of_block[start_bid].clone(),
            params: self.block_phis_to_param_names(start_bid),
            body: self.add_node(Node::Seq(seq)),
        }
    }

    fn block_phis_to_param_names(&mut self, bid: BlockID) -> SmallVec<[Ident; 2]> {
        let phis = self.ssa.block_phi(bid);
        (0..phis.phi_count())
            .map(|phi_ndx| phis.phi_reg(phi_ndx))
            .filter(|phi_reg| self.ssa.is_alive(*phi_reg))
            .map(|phi_reg| {
                self.name_of_value
                    .get(&phi_reg)
                    .expect("unnamed phi node!")
                    .clone()
            })
            .collect()
    }

    fn compile_thunk_body(&mut self, bid: BlockID, out_seq: &mut Seq) {
        assert!(
            !self.blocks_compiling.contains(&bid),
            "compiling block again! {bid:?}"
        );
        self.blocks_compiling.push(bid);

        let cfg = self.ssa.cfg();

        let nor_insns = self.ssa.block_normal_insns(bid).unwrap();
        self.compile_seq(nor_insns, out_seq);

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
                    !nor_insns.insns.is_empty(),
                    "block with BlockCont::Alt continuation must have at least 1 insn"
                );
                let last_insn = nor_insns.insns.last().unwrap().get();

                let cond = match last_insn {
                    mil::Insn::JmpIf { cond, target: _ } => cond,
                    _ => panic!("block with BlockCont::Alt continuation must end with a JmpIf"),
                };
                let cond = self.add_node_of_value(cond);
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
        for &dominated_bid in cfg.dom_tree().children_of(bid) {
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

    fn compile_seq(&mut self, insns: mil::InsnSlice, out_seq: &mut Seq) {
        out_seq.extend(
            insns
                .dests
                .iter()
                .zip(insns.insns.iter())
                .map(|(r, i)| (r.get(), i.get()))
                .filter_map(|(reg, insn)| {
                    let name = self.name_of_value.get(&reg).cloned();

                    if !insn.has_side_effects() && name.is_none() {
                        return None;
                    }

                    let node = self.compile_node(reg);
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
                .filter(|phi_ndx| self.ssa.is_alive(target_phis.phi_reg(*phi_ndx)))
                .map(|phi_ndx| {
                    let reg = target_phis.arg(self.ssa, phi_ndx, pred_ndx.into());
                    self.add_node_of_value(reg)
                })
                .collect()
        };

        let cur_bid = self.blocks_compiling.last().copied().unwrap();

        let cfg = self.ssa.cfg();
        let dom_tree = cfg.dom_tree();

        let params = self.block_phis_to_param_names(target_bid);
        assert_eq!(params.len(), args.len(), "inconsistent arg/param count");

        if dom_tree.parent_of(target_bid) == Some(cur_bid) {
            // TODO represent initial parameter assignment some better way
            out_seq.extend(params.into_iter().zip(args).map(|(param, arg)| {
                self.add_node(Node::LetMut {
                    name: param,
                    value: arg,
                })
            }));

            let labeled_node = self.compile_to_labeled(target_bid);
            let preds_count = cfg.inverse().successors(target_bid).len();
            let node_id = match labeled_node {
                Node::Labeled {
                    label: _,
                    body,
                    params,
                } if preds_count == 1 => {
                    assert_eq!(params.as_slice(), &[]);
                    body
                }

                other => self.add_node(other),
            };

            out_seq.push(node_id);
        } else {
            let thunk_id = self.label_of_block[target_bid].clone();
            // the label definition is going to be built while processing the actual dominator node
            let node = Node::ContinueToLabel(
                thunk_id,
                ContinueArgs {
                    names: params,
                    values: args,
                },
            );
            out_seq.push(self.add_node(node));
        }
    }

    fn compile_node(&mut self, reg: mil::Reg) -> Node {
        use mil::Insn;

        let iv = self.ssa.get(reg).unwrap();
        match iv.insn.get() {
            Insn::Call(callee) => {
                let callee = self.add_node_of_value(callee);

                let mut args = SmallVec::new();
                for arg_reg in self.ssa.get_call_args(reg) {
                    let arg = self.add_node_of_value(arg_reg);
                    args.push(arg);
                }

                Node::Call(callee, args)
            }
            // To be handled in ::Call
            Insn::CArg { .. } => Node::Nop,
            Insn::Ret(arg) => Node::Return(self.add_node_of_value(arg)),
            Insn::JmpExt(target) => Node::ContinueToExtern(target),
            Insn::JmpI(_) | Insn::Jmp(_) | Insn::JmpExtIf { .. } | Insn::JmpIf { .. } => {
                // skip.  the control fllow handling in compile_thunk shall take care of this
                Node::Nop
            }
            Insn::StoreMem(addr, val) => {
                let size = self
                    .ssa
                    .value_type(val)
                    .bytes_size()
                    .expect("Insn::StoreMem: value has no size!");

                use mil::ArithOp;
                let attr_def = self.ssa.get(addr).unwrap().insn.get();
                let (base, offset) = match attr_def {
                    Insn::ArithK1(ArithOp::Add, base, offset)
                    | Insn::ArithK2(ArithOp::Add, base, offset)
                    | Insn::ArithK4(ArithOp::Add, base, offset)
                    | Insn::ArithK8(ArithOp::Add, base, offset) => (base, offset),
                    Insn::ArithK1(ArithOp::Sub, base, offset)
                    | Insn::ArithK2(ArithOp::Sub, base, offset)
                    | Insn::ArithK4(ArithOp::Sub, base, offset)
                    | Insn::ArithK8(ArithOp::Sub, base, offset) => (base, -offset),
                    _ => (addr, 0),
                };

                let offset_range = offset..offset + (size as i64);
                let loc = self
                    .add_node_derefing(base, offset_range)
                    .unwrap_or_else(|| {
                        let addr = self.add_node_of_value(addr);
                        self.add_node(Node::RawDeref { addr, size })
                    });

                let value = self.add_node_of_value(val);
                Node::StoreMem { loc, value }
            }

            Insn::Const1(val) => Node::Const1(val),
            Insn::Const2(val) => Node::Const2(val),
            Insn::Const4(val) => Node::Const4(val),
            Insn::Const8(val) => Node::Const8(val),
            Insn::True => Node::True,
            Insn::False => Node::False,

            Insn::L1(reg) => Node::L1(self.add_node_of_value(reg)),
            Insn::L2(reg) => Node::L2(self.add_node_of_value(reg)),
            Insn::L4(reg) => Node::L4(self.add_node_of_value(reg)),
            Insn::Get8(reg) => self.node_of_value(reg),

            Insn::V8WithL1(a, b) => {
                Node::WithL1(self.add_node_of_value(a), self.add_node_of_value(b))
            }
            Insn::V8WithL2(a, b) => {
                Node::WithL2(self.add_node_of_value(a), self.add_node_of_value(b))
            }
            Insn::V8WithL4(a, b) => {
                Node::WithL4(self.add_node_of_value(a), self.add_node_of_value(b))
            }
            Insn::Widen1_2(x) => {
                let x = self.add_node_of_value(x);
                Node::Widen {
                    from_sz: 1,
                    to_sz: 2,
                    reg: x,
                }
            }
            Insn::Widen1_4(x) => {
                let x = self.add_node_of_value(x);
                Node::Widen {
                    from_sz: 1,
                    to_sz: 4,
                    reg: x,
                }
            }
            Insn::Widen1_8(x) => {
                let x = self.add_node_of_value(x);
                Node::Widen {
                    from_sz: 1,
                    to_sz: 8,
                    reg: x,
                }
            }
            Insn::Widen2_4(x) => {
                let x = self.add_node_of_value(x);
                Node::Widen {
                    from_sz: 2,
                    to_sz: 4,
                    reg: x,
                }
            }
            Insn::Widen2_8(x) => {
                let x = self.add_node_of_value(x);
                Node::Widen {
                    from_sz: 2,
                    to_sz: 8,
                    reg: x,
                }
            }
            Insn::Widen4_8(x) => {
                let x = self.add_node_of_value(x);
                Node::Widen {
                    from_sz: 4,
                    to_sz: 8,
                    reg: x,
                }
            }

            Insn::Arith1(op, a, b)
            | Insn::Arith2(op, a, b)
            | Insn::Arith4(op, a, b)
            | Insn::Arith8(op, a, b) => {
                let a = self.add_node_of_value(a);
                let b = self.add_node_of_value(b);
                self.fold_bin(op.into(), a, b)
            }
            Insn::ArithK1(op, a, k)
            | Insn::ArithK2(op, a, k)
            | Insn::ArithK4(op, a, k)
            | Insn::ArithK8(op, a, k) => {
                let a = self.add_node_of_value(a);
                let b = self.add_node(Node::Const8(k));
                self.fold_bin(op.into(), a, b)
            }

            Insn::Cmp(op, a, b) => {
                let a = self.add_node_of_value(a);
                let b = self.add_node_of_value(b);
                Node::Cmp {
                    op: op.into(),
                    a,
                    b,
                }
            }
            Insn::Not(x) => Node::Not(self.add_node_of_value(x)),
            Insn::TODO(msg) => Node::TODO(msg),
            Insn::LoadMem1(addr_reg) => Node::LoadMem1(self.add_node_of_value(addr_reg)),
            Insn::LoadMem2(addr_reg) => Node::LoadMem2(self.add_node_of_value(addr_reg)),
            Insn::LoadMem4(addr_reg) => Node::LoadMem4(self.add_node_of_value(addr_reg)),
            Insn::LoadMem8(addr_reg) => Node::LoadMem8(self.add_node_of_value(addr_reg)),

            Insn::Bool(bool_op, a, b) => {
                let op = match bool_op {
                    mil::BoolOp::Or => BinOp::BoolOr,
                    mil::BoolOp::And => BinOp::BoolAnd,
                };
                let a = self.add_node_of_value(a);
                let b = self.add_node_of_value(b);
                Node::Bin { op, a, b }
            }

            Insn::OverflowOf(arg) => Node::OverflowOf(self.add_node_of_value(arg)),
            Insn::CarryOf(arg) => Node::CarryOf(self.add_node_of_value(arg)),
            Insn::SignOf(arg) => Node::SignOf(self.add_node_of_value(arg)),
            Insn::IsZero(arg) => Node::IsZero(self.add_node_of_value(arg)),
            Insn::Parity(arg) => Node::Parity(self.add_node_of_value(arg)),

            Insn::Undefined => Node::Undefined,
            Insn::Ancestral(anc) => Node::Pre(anc.name()),

            mil::Insn::Phi1
            | mil::Insn::Phi2
            | mil::Insn::Phi4
            | mil::Insn::Phi8
            | mil::Insn::PhiBool => {
                let mut args = SmallVec::new();

                for (pred_ndx, value) in self.ssa.get_phi_args(reg).enumerate() {
                    let pred_ndx = pred_ndx.try_into().unwrap();
                    let value_expr = self.add_node_of_value(value);
                    args.push((pred_ndx, value_expr));
                }

                Node::Phi(args)
            }
            mil::Insn::PhiArg(_) => panic!("PhiArg passed to collect_expr"),
        }
    }

    #[cfg(not(feature = "proto_typing"))]
    fn add_node_derefing(&mut self, addr_reg: mil::Reg, size: u32) -> Option<NodeID> {
        None
    }

    #[cfg(feature = "proto_typing")]
    fn add_node_derefing(
        &mut self,
        base_reg: mil::Reg,
        offset_range: Range<i64>,
    ) -> Option<NodeID> {
        let types = self.types?;
        let pt = self.ssa.ptr_type(base_reg)?;

        let sel = types.select(pt.pointee_tyid, offset_range).ok()?;

        let path = sel
            .path
            .into_iter()
            .map(|step| match step {
                ty::SelectStep::Index(_) => todo!(),
                ty::SelectStep::Member(rc) => Arc::clone(&rc),
            })
            .collect();

        let base = self.add_node_of_value(base_reg);
        Some(self.add_node(Node::MemberAccess { base, path }))
    }

    fn add_node_of_value(&mut self, reg: mil::Reg) -> NodeID {
        let node = self.node_of_value(reg);
        self.add_node(node)
    }
    fn node_of_value(&mut self, reg: mil::Reg) -> Node {
        if let Some(lbl) = self.name_of_value.get(&reg) {
            Node::Ref(lbl.clone())
        } else {
            // inline
            self.compile_node(reg)
        }
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
            Node::Bin {
                op: BinOp::BitAnd,
                a,
                b,
            } if a == b => {
                nodes.swap(nid, *a);
            }
            _ => {}
        }
    }
}

impl Ast {
    pub fn pretty_print<W: PP + ?Sized>(&self, pp: &mut W) -> std::fmt::Result {
        self.pretty_print_node(pp, self.root_nid)
    }

    fn pretty_print_node<W: PP + ?Sized>(&self, pp: &mut W, nid: NodeID) -> std::fmt::Result {
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

            Node::Labeled {
                params,
                label,
                body,
            } => {
                if params.is_empty() {
                    write!(pp, "\n'{}: ", label.0)?;
                } else {
                    write!(pp, "\n'{}(", label.0)?;
                    for (ndx, param) in params.iter().enumerate() {
                        if ndx > 0 {
                            write!(pp, ", ")?;
                        }
                        write!(pp, "{}", param.0.as_str())?;
                    }
                    write!(pp, "): ")?;
                }
                self.pretty_print_node(pp, *body)
            }

            Node::ContinueToLabel(thunk_id, args) => {
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
            Node::True => write!(pp, "true"),
            Node::False => write!(pp, "false"),

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
                    BinOp::BitXor => "^",
                    BinOp::BoolAnd => "&&",
                    BinOp::BoolOr => "||",
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
            Node::Widen {
                from_sz,
                to_sz,
                reg,
            } => {
                write!(pp, "(")?;
                self.pretty_print_node(pp, *reg)?;
                write!(pp, ")")?;
                write!(pp, ".widen[{}->{}]", from_sz, to_sz)
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
                write!(pp, "phi(")?;
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
                write!(pp, ")")
            }

            Node::LoadMem1(arg) => self.pp_load_mem(pp, 1, *arg),
            Node::LoadMem2(arg) => self.pp_load_mem(pp, 2, *arg),
            Node::LoadMem4(arg) => self.pp_load_mem(pp, 4, *arg),
            Node::LoadMem8(arg) => self.pp_load_mem(pp, 8, *arg),

            Node::StoreMem { loc, value } => {
                self.pretty_print_node(pp, *loc)?;
                write!(pp, " = ")?;
                pp.open_box();
                self.pretty_print_node(pp, *value)?;
                pp.close_box();
                Ok(())
            }
            Node::RawDeref { addr, size } => {
                write!(pp, "[")?;
                self.pretty_print_node(pp, *addr)?;
                write!(pp, "]:{}", *size)
            }
            Node::MemberAccess { base, path } => {
                self.pretty_print_node(pp, *base)?;
                for step in path {
                    write!(pp, ".{}", step)?;
                }
                Ok(())
            }

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
            Node::Pre(tag) => write!(pp, "<pre:{}>", tag),
            Node::Undefined => write!(pp, "<undefined>"),
            Node::Nop => write!(pp, "nop"),
        }
    }

    fn pp_load_mem<W: PP + ?Sized>(
        &self,
        pp: &mut W,
        src_size: u8,
        addr: NodeID,
    ) -> std::fmt::Result {
        write!(pp, "[")?;
        self.pretty_print_node(pp, addr)?;
        write!(pp, "]:{}", src_size)
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

#[cfg(feature = "proto_typing")]
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{ast, mil, ssa, ty};
    use mil::{ArithOp, Insn, Reg};

    crate::define_ancestral_name!(ANC_ARG0, "arg0");

    #[test]
    fn store_target_struct_member() {
        let prog = {
            let mut b = mil::ProgramBuilder::new();
            // this has undergone SSA and constant folding, but ProgramBuilder
            // is the nicer API to get a proper ssa::Program
            b.push(Reg(0), Insn::Ancestral(ANC_ARG0));
            b.push(Reg(1), Insn::ArithK8(ArithOp::Add, Reg(0), 8));
            b.push(Reg(2), Insn::Const4(123));
            b.push(Reg(3), Insn::StoreMem(Reg(1), Reg(2)));
            b.push(Reg(4), Insn::Ret(Reg(1)));
            b.build()
        };

        let mut prog = ssa::mil_to_ssa(ssa::ConversionParams::new(prog));

        let mut types = ty::TypeSet::new();
        let tyid_i32 = types.add(ty::Type {
            name: Arc::new("i32".to_owned()),
            ty: ty::Ty::Int(ty::Int {
                size: 4,
                signed: ty::Signedness::Signed,
            }),
        });
        let tyid_i8 = types.add(ty::Type {
            name: Arc::new("i8".to_owned()),
            ty: ty::Ty::Int(ty::Int {
                size: 1,
                signed: ty::Signedness::Signed,
            }),
        });
        let type_id = types.add(ty::Type {
            name: Arc::new("point".to_owned()),
            ty: ty::Ty::Struct(ty::Struct {
                size: 24,
                members: vec![
                    ty::StructMember {
                        offset: 0,
                        name: Arc::new("x".to_owned()),
                        tyid: tyid_i32,
                    },
                    // a bit of padding
                    ty::StructMember {
                        offset: 8,
                        name: Arc::new("y".to_owned()),
                        tyid: tyid_i32,
                    },
                    ty::StructMember {
                        offset: 12,
                        name: Arc::new("cost".to_owned()),
                        tyid: tyid_i8,
                    },
                ],
            }),
        });

        // TODO this shall be inferred by a dedicated SSA processing step, after
        // doing the following:
        //    let ptr_ty = ssa::Ptr { type_id, offset: 0 };
        //    prog.set_ptr_type(Reg(0), ptr_ty);
        prog.set_ptr_type(
            Reg(0),
            ssa::Ptr {
                pointee_tyid: type_id,
            },
        );

        let ast = {
            let mut builder = ast::Builder::new(&prog);
            builder.use_type_set(&mut types);
            builder.compile()
        };
        assert_ast_snapshot(&ast);
    }

    fn assert_ast_snapshot(ast: &ast::Ast) {
        use crate::pp::PrettyPrinter;
        use insta::assert_snapshot;

        let mut out = String::new();
        let mut pp = PrettyPrinter::start(&mut out);
        ast.pretty_print(&mut pp).unwrap();
        assert_snapshot!(out);
    }
}
