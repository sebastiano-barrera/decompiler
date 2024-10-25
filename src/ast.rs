use std::ops::Range;
#[allow(dead_code)]
#[allow(unused)]
use std::{collections::HashMap, rc::Rc};

use smallvec::SmallVec;

use crate::{
    cfg::{self, BlockID},
    mil,
    pp::PrettyPrinter,
    ssa,
};

#[derive(Debug)]
pub struct Ast {
    root_thunk: ThunkID,
    thunks: HashMap<ThunkID, Thunk>,
}

#[derive(Debug)]
struct Thunk {
    body: Node,
    // there is one param per phi node
    params: SmallVec<[Ident; 2]>,
}

#[derive(Debug, PartialEq, Eq)]
struct ThunkArgs {
    // thunk's parameters (copied)
    names: SmallVec<[Ident; 2]>,
    // values, in lockstep with `params`
    values: SmallVec<[NodeP; 2]>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Ident(Rc<String>);

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ThunkID(Rc<String>);

#[derive(Debug, PartialEq, Eq)]
enum Node {
    Seq(Seq),
    If {
        cond: NodeP,
        cons: NodeP,
        alt: NodeP,
    },
    Let {
        name: Ident,
        value: NodeP,
    },
    LetMut {
        name: Ident,
        value: NodeP,
    },
    Ref(Ident),
    ContinueToThunk(ThunkID, ThunkArgs),
    ContinueToExtern(u64),

    Labeled(ThunkID, NodeP),

    Const1(u8),
    Const2(u16),
    Const4(u32),
    Const8(u64),

    L1(NodeP),
    L2(NodeP),
    L4(NodeP),

    WithL1(NodeP, NodeP),
    WithL2(NodeP, NodeP),
    WithL4(NodeP, NodeP),

    Bin {
        op: BinOp,
        args: SmallVec<[NodeP; 2]>,
    },
    Not(NodeP),

    Call(Box<Node>, SmallVec<[NodeP; 4]>),
    Return(NodeP),
    TODO(&'static str),
    Phi(SmallVec<[(u16, NodeP); 2]>),

    LoadMem1(Box<Node>),
    LoadMem2(Box<Node>),
    LoadMem4(Box<Node>),
    LoadMem8(Box<Node>),
    StoreMem1(Box<Node>, Box<Node>),
    StoreMem2(Box<Node>, Box<Node>),
    StoreMem4(Box<Node>, Box<Node>),
    StoreMem8(Box<Node>, Box<Node>),

    OverflowOf(Box<Node>),
    CarryOf(Box<Node>),
    SignOf(Box<Node>),
    IsZero(Box<Node>),
    Parity(Box<Node>),
    StackBot,
    Undefined,
    Nop,
}

type NodeP = Box<Node>;
type Seq = SmallVec<[NodeP; 2]>;

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
    Eq,
}

impl Node {
    fn boxed(self) -> Box<Self> {
        Box::new(self)
    }
}

pub fn ssa_to_ast(ssa: &ssa::Program, pat_sel: &PatternSel) -> Ast {
    // the set of blocks in the CFG
    // is transformed
    // into a set of Thunk-s
    //
    // each thunk can result from one or more blocks, merged.

    let mut builder = Builder::init_to_ast(ssa);

    for (bid, pat_ndx) in pat_sel.sel.items() {
        if let Some(pat_ndx) = pat_ndx {
            let pat = &pat_sel.set.pats[*pat_ndx];
            assert_eq!(pat.key_bid, bid);

            match &pat.pat {
                Pat::IfBranch { path } => {
                    mark_path_inline(&mut builder, path);
                }
                Pat::Cycle { path } => {
                    if path.len() == 0 {
                        todo!("not yet implemented: self-recursive blocks")
                    }

                    mark_path_inline(&mut builder, path);
                    let last = path.last().copied().unwrap();
                    builder.mark_edge_loop(last, bid);
                }
            }
        }
    }

    for bid in ssa.cfg().block_ids() {
        if !builder.visited[bid] {
            builder.compile_new_thunk(bid);
        }
    }

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
    edge_flags: HashMap<(BlockID, BlockID), EdgeFlags>,
    blocks_compiling: Vec<BlockID>,
    visited: cfg::BlockMap<bool>,
}

// TODO replace with a proper bitmask
#[derive(Clone, Copy, Default)]
struct EdgeFlags {
    is_loop: bool,
    is_inline: bool,
}

impl<'a> Builder<'a> {
    fn init_to_ast(ssa: &'a ssa::Program) -> Builder<'a> {
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
            edge_flags: HashMap::new(),
            blocks_compiling: Vec::new(),
            visited: cfg::BlockMap::new(false, cfg.block_count()),
        }
    }

    fn finish(self) -> Ast {
        let root_tid = self.thunk_id_of_block[cfg::ENTRY_BID].clone();
        Ast {
            root_thunk: root_tid,
            thunks: self.thunks,
        }
    }

    fn compile_new_thunk(&mut self, start_bid: cfg::BlockID) -> ThunkID {
        let mut seq = Seq::new();

        let params = self.thunk_params_of_block_phis(start_bid);
        self.compile_thunk_body(start_bid, &mut seq);

        let thunk = Thunk {
            params,
            body: Node::Seq(seq),
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

        assert!(!self.visited[bid]);
        self.visited[bid] = true;

        let nor_ndxs = self.ssa.cfg().insns_ndx_range(bid);

        self.compile_seq(&nor_ndxs, out_seq);

        match self.ssa.cfg().block_cont(bid) {
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
                let cond = self.get_node(cond.0).boxed();
                let cons = {
                    let mut seq = Seq::new();
                    self.compile_continue(neg_bid, neg_pred_ndx, &mut seq);
                    Node::Seq(seq).boxed()
                };
                let alt = {
                    let mut seq = Seq::new();
                    self.compile_continue(pos_bid, pos_pred_ndx, &mut seq);
                    Node::Seq(seq).boxed()
                };

                out_seq.push(Node::If { cond, cons, alt }.boxed());
            }
        };

        let check = self.blocks_compiling.pop();
        assert_eq!(check, Some(bid));
    }

    fn compile_seq(&mut self, nor_ndxs: &Range<mil::Index>, out_seq: &mut Seq) {
        out_seq.extend(
            nor_ndxs
                .clone()
                .map(|ndx| (ndx, self.ssa.get(ndx).unwrap()))
                .filter_map(|(ndx, iv)| {
                    let name = self.name_of_value.get(&iv.dest.0);

                    if !iv.insn.has_side_effects() && name.is_none() {
                        return None;
                    }

                    let node = self.compile_node(ndx);
                    if node == Node::Nop {
                        return None;
                    }

                    if let Some(name) = name {
                        return Some(
                            Node::Let {
                                name: name.clone(),
                                value: node.boxed(),
                            }
                            .boxed(),
                        );
                    };

                    Some(node.boxed())
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
                    self.get_node(reg.0).boxed()
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
                            Node::LetMut {
                                name: param,
                                value: arg,
                            }
                            .boxed()
                        }),
                );

                let mut inner_seq = SmallVec::new();
                self.compile_thunk_body(target_bid, &mut inner_seq);
                let inner_seq = Node::Seq(inner_seq).boxed();
                let label = self.thunk_id_of_block[target_bid].clone();
                out_seq.push(Node::Labeled(label, inner_seq).boxed());
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
            out_seq.push(node.boxed());
        }
    }

    fn compile_node(&self, start_ndx: mil::Index) -> Node {
        use mil::Insn;

        let iv = self.ssa.get(start_ndx).unwrap();
        match iv.insn {
            Insn::Call { callee, arg0 } => {
                let callee = self.get_node(callee.0).boxed();
                let mut args = SmallVec::new();
                let mut arg = arg0;
                loop {
                    let arg_insn = self.ssa.get(arg.0).unwrap();
                    match arg_insn.insn {
                        Insn::CArg { value, prev } => {
                            let arg_val = self.get_node(value.0).boxed();
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
            Insn::Ret(arg) => Node::Return(self.get_node(arg.0).boxed()),
            Insn::JmpExt(target) => Node::ContinueToExtern(*target),
            Insn::JmpI(_) | Insn::Jmp(_) | Insn::JmpExtIf { .. } | Insn::JmpIf { .. } => {
                // skip.  the control fllow handling in compile_thunk shall take care of this
                Node::Nop
            }
            Insn::StoreMem1(addr, val) => {
                let addr = self.get_node(addr.0).boxed();
                let val = self.get_node(val.0).boxed();
                Node::StoreMem1(addr, val)
            }
            Insn::StoreMem2(addr, val) => {
                let addr = self.get_node(addr.0).boxed();
                let val = self.get_node(val.0).boxed();
                Node::StoreMem2(addr, val)
            }
            Insn::StoreMem4(addr, val) => {
                let addr = self.get_node(addr.0).boxed();
                let val = self.get_node(val.0).boxed();
                Node::StoreMem4(addr, val)
            }
            Insn::StoreMem8(addr, val) => {
                let addr = self.get_node(addr.0).boxed();
                let val = self.get_node(val.0).boxed();
                Node::StoreMem8(addr, val)
            }

            Insn::Const1(val) => Node::Const1(*val),
            Insn::Const2(val) => Node::Const2(*val),
            Insn::Const4(val) => Node::Const4(*val),
            Insn::Const8(val) => Node::Const8(*val),

            Insn::L1(reg) => Node::L1(self.get_node(reg.0).boxed()),
            Insn::L2(reg) => Node::L2(self.get_node(reg.0).boxed()),
            Insn::L4(reg) => Node::L4(self.get_node(reg.0).boxed()),
            Insn::Get(reg) => self.get_node(reg.0),

            Insn::WithL1(a, b) => {
                Node::WithL1(self.get_node(a.0).boxed(), self.get_node(b.0).boxed())
            }
            Insn::WithL2(a, b) => {
                Node::WithL2(self.get_node(a.0).boxed(), self.get_node(b.0).boxed())
            }
            Insn::WithL4(a, b) => {
                Node::WithL4(self.get_node(a.0).boxed(), self.get_node(b.0).boxed())
            }
            Insn::Add(a, b) => fold_bin(
                BinOp::Add,
                self.get_node(a.0).boxed(),
                self.get_node(b.0).boxed(),
            ),
            Insn::AddK(a, k) => fold_bin(
                BinOp::Add,
                self.get_node(a.0).boxed(),
                Node::Const8(*k as u64).boxed(),
            ),
            Insn::Sub(a, b) => fold_bin(
                BinOp::Sub,
                self.get_node(a.0).boxed(),
                self.get_node(b.0).boxed(),
            ),
            Insn::Mul(a, b) => fold_bin(
                BinOp::Mul,
                self.get_node(a.0).boxed(),
                self.get_node(b.0).boxed(),
            ),
            Insn::MulK32(a, k) => fold_bin(
                BinOp::Mul,
                self.get_node(a.0).boxed(),
                Node::Const8(*k as u64).boxed(),
            ),
            Insn::Shl(a, b) => fold_bin(
                BinOp::Shl,
                self.get_node(a.0).boxed(),
                self.get_node(b.0).boxed(),
            ),
            Insn::BitAnd(a, b) => fold_bin(
                BinOp::BitAnd,
                self.get_node(a.0).boxed(),
                self.get_node(b.0).boxed(),
            ),
            Insn::BitOr(a, b) => fold_bin(
                BinOp::BitOr,
                self.get_node(a.0).boxed(),
                self.get_node(b.0).boxed(),
            ),
            Insn::Eq(a, b) => fold_bin(
                BinOp::Eq,
                self.get_node(a.0).boxed(),
                self.get_node(b.0).boxed(),
            ),
            Insn::Not(x) => Node::Not(self.get_node(x.0).boxed()),
            Insn::TODO(msg) => Node::TODO(msg),
            Insn::LoadMem1(addr_reg) => Node::LoadMem1(self.get_node(addr_reg.0).boxed()),
            Insn::LoadMem2(addr_reg) => Node::LoadMem2(self.get_node(addr_reg.0).boxed()),
            Insn::LoadMem4(addr_reg) => Node::LoadMem4(self.get_node(addr_reg.0).boxed()),
            Insn::LoadMem8(addr_reg) => Node::LoadMem8(self.get_node(addr_reg.0).boxed()),

            Insn::OverflowOf(arg) => Node::OverflowOf(self.get_node(arg.0).boxed()),
            Insn::CarryOf(arg) => Node::CarryOf(self.get_node(arg.0).boxed()),
            Insn::SignOf(arg) => Node::SignOf(self.get_node(arg.0).boxed()),
            Insn::IsZero(arg) => Node::IsZero(self.get_node(arg.0).boxed()),
            Insn::Parity(arg) => Node::Parity(self.get_node(arg.0).boxed()),

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
                    let value_expr = self.get_node(value.0).boxed();
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

    fn get_node(&self, ndx: mil::Index) -> Node {
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
}

fn fold_bin(op: BinOp, a: Box<Node>, b: Box<Node>) -> Node {
    // TODO!
    Node::Bin {
        op,
        args: [a, b].into(),
    }
}

impl Ast {
    pub fn pretty_print<W: std::fmt::Write>(&self, pp: &mut PrettyPrinter<W>) -> std::fmt::Result {
        use std::fmt::Write;
        for (thid, thunk) in self.thunks.iter() {
            write!(pp, "{} :: ", thid.0.as_str())?;
            thunk.pretty_print(pp)?;
            writeln!(pp)?;
        }
        Ok(())
    }
}
impl Thunk {
    pub fn pretty_print<W: std::fmt::Write>(&self, pp: &mut PrettyPrinter<W>) -> std::fmt::Result {
        use std::fmt::Write;

        write!(pp, "thunk (")?;
        for param_name in &self.params {
            write!(pp, "{}, ", param_name.0.as_str())?;
        }
        write!(pp, ") ")?;

        self.body.pretty_print(pp)?;
        Ok(())
    }
}
impl Node {
    pub fn pretty_print<W: std::fmt::Write>(&self, pp: &mut PrettyPrinter<W>) -> std::fmt::Result {
        use std::fmt::Write;
        match self {
            Node::Seq(nodes) => {
                write!(pp, "{{\n    ")?;
                pp.open_box();
                for (ndx, node) in nodes.iter().enumerate() {
                    node.pretty_print(pp)?;
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
                cond.pretty_print(pp)?;
                pp.close_box();

                write!(pp, " ")?;
                cons.pretty_print(pp)?;

                write!(pp, " else ")?;
                alt.pretty_print(pp)
            }

            Node::Let { name, value } => {
                write!(pp, "let {} = ", name.0.as_str())?;
                pp.open_box();
                value.pretty_print(pp)?;
                pp.close_box();
                Ok(())
            }
            Node::LetMut { name, value } => {
                write!(pp, "let mut {} = ", name.0.as_str())?;
                pp.open_box();
                value.pretty_print(pp)?;
                pp.close_box();
                Ok(())
            }

            Node::Ref(ident) => write!(pp, "{}", ident.0.as_str()),

            Node::Labeled(thunk_id, node) => {
                write!(pp, "'{}: ", thunk_id.0)?;
                node.pretty_print(pp)
            }

            Node::ContinueToThunk(thunk_id, args) => {
                write!(pp, "goto {}", thunk_id.0.as_str())?;
                let args_count = args.names.len();
                assert_eq!(args_count, args.values.len());
                match args_count {
                    0 => {}
                    1 => {
                        write!(pp, " ({} = ", args.names[0].0.as_str())?;
                        args.values[0].pretty_print(pp)?;
                        write!(pp, ")")?;
                    }
                    _ => {
                        write!(pp, " with (")?;
                        pp.open_box();
                        for (ndx, (name, arg)) in args.names.iter().zip(&args.values).enumerate() {
                            pp.open_box();
                            write!(pp, "{} = ", name.0.as_str())?;
                            arg.pretty_print(pp)?;
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
                arg.pretty_print(pp)?;
                write!(pp, ".l1")
            }
            Node::L2(arg) => {
                arg.pretty_print(pp)?;
                write!(pp, ".l2")
            }
            Node::L4(arg) => {
                arg.pretty_print(pp)?;
                write!(pp, ".l4")
            }
            Node::WithL1(a, b) => {
                write!(pp, "(")?;
                a.pretty_print(pp)?;
                write!(pp, " with l1 = ")?;
                b.pretty_print(pp)?;
                write!(pp, ")")
            }
            Node::WithL2(a, b) => {
                write!(pp, "(")?;
                a.pretty_print(pp)?;
                write!(pp, " with l2 = ")?;
                b.pretty_print(pp)?;
                write!(pp, ")")
            }
            Node::WithL4(a, b) => {
                write!(pp, "(")?;
                a.pretty_print(pp)?;
                write!(pp, " with l4 = ")?;
                b.pretty_print(pp)?;
                write!(pp, ")")
            }
            Node::Bin { op, args } => {
                let op_s = match op {
                    BinOp::Add => " + ",
                    BinOp::Sub => " - ",
                    BinOp::Mul => " * ",
                    BinOp::Div => " / ",
                    BinOp::Shl => " << ",
                    BinOp::Shr => " >> ",
                    BinOp::BitAnd => " & ",
                    BinOp::BitOr => " | ",
                    BinOp::Eq => " == ",
                };

                for (ndx, arg) in args.iter().enumerate() {
                    let needs_parens = matches!(&**arg, Node::Bin { .. });

                    if ndx > 0 {
                        write!(pp, "{}", op_s)?;
                    }
                    if needs_parens {
                        write!(pp, "(")?;
                    }
                    pp.open_box();
                    arg.pretty_print(pp)?;
                    pp.close_box();
                    if needs_parens {
                        write!(pp, ")")?;
                    }
                }
                Ok(())
            }

            Node::Not(arg) => {
                write!(pp, "not ")?;
                arg.pretty_print(pp)
            }
            Node::Call(callee, args) => {
                callee.pretty_print(pp)?;
                write!(pp, "(\n  ")?;
                pp.open_box();
                for (ndx, arg) in args.iter().enumerate() {
                    if ndx > 0 {
                        writeln!(pp)?;
                    }
                    arg.pretty_print(pp)?;
                    write!(pp, ",")?;
                }
                pp.close_box();
                write!(pp, "\n)")
            }
            Node::Return(arg) => {
                write!(pp, "return ")?;
                arg.pretty_print(pp)
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
                    value.pretty_print(pp)?;
                    write!(pp, ",")?;
                }
                pp.close_box();
                write!(pp, "\n)")
            }

            Node::LoadMem1(arg) => pp_load_mem(pp, 1, arg),
            Node::LoadMem2(arg) => pp_load_mem(pp, 2, arg),
            Node::LoadMem4(arg) => pp_load_mem(pp, 4, arg),
            Node::LoadMem8(arg) => pp_load_mem(pp, 8, arg),

            Node::StoreMem1(dest, val) => pp_store_mem(pp, 1, dest, val),
            Node::StoreMem2(dest, val) => pp_store_mem(pp, 2, dest, val),
            Node::StoreMem4(dest, val) => pp_store_mem(pp, 4, dest, val),
            Node::StoreMem8(dest, val) => pp_store_mem(pp, 8, dest, val),

            Node::OverflowOf(arg) => {
                write!(pp, "overflow (")?;
                arg.pretty_print(pp)?;
                write!(pp, ")")
            }
            Node::CarryOf(arg) => {
                write!(pp, "carry (")?;
                arg.pretty_print(pp)?;
                write!(pp, ")")
            }
            Node::SignOf(arg) => {
                write!(pp, "sign (")?;
                arg.pretty_print(pp)?;
                write!(pp, ")")
            }
            Node::IsZero(arg) => {
                write!(pp, "is0 (")?;
                arg.pretty_print(pp)?;
                write!(pp, ")")
            }
            Node::Parity(arg) => {
                write!(pp, "parity (")?;
                arg.pretty_print(pp)?;
                write!(pp, ")")
            }
            Node::StackBot => write!(pp, "<stackBottom>"),
            Node::Undefined => write!(pp, "<undefined>"),
            Node::Nop => write!(pp, "nop"),
        }
    }
}

fn pp_store_mem<W: std::fmt::Write>(
    pp: &mut PrettyPrinter<W>,
    dest_size: u8,
    dest: &Node,
    val: &Node,
) -> std::fmt::Result {
    use std::fmt::Write;
    write!(pp, "[")?;
    dest.pretty_print(pp)?;
    write!(pp, "]:{} = ", dest_size)?;
    pp.open_box();
    val.pretty_print(pp)?;
    pp.close_box();
    Ok(())
}

fn pp_load_mem<W: std::fmt::Write>(
    pp: &mut PrettyPrinter<W>,
    src_size: u8,
    addr: &Node,
) -> std::fmt::Result {
    use std::fmt::Write;
    write!(pp, "[")?;
    addr.pretty_print(pp)?;
    write!(pp, "]:{}", src_size)
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
