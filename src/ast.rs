#[allow(dead_code)]
#[allow(unused)]
use std::{collections::HashMap, rc::Rc};

use smallvec::SmallVec;

use crate::{cfg, mil, ssa};

#[derive(Debug)]
pub struct Ast {
    root_thunk: Ident,
    thunks: HashMap<Ident, Thunk>,
}

#[derive(Debug)]
struct Thunk {
    body: Node,
    // there is one param per phi node
    params: SmallVec<[Ident; 2]>,
}

#[derive(Debug, PartialEq, Eq)]
struct ThunkArgs {
    // values are in the same length and order as the target Thunk's params
    values: SmallVec<[NodeP; 2]>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Ident(Rc<String>);

#[derive(Debug, PartialEq, Eq)]
enum Node {
    Seq(SmallVec<[NodeP; 2]>),
    If {
        cond: NodeP,
        cons: NodeP,
        alt: NodeP,
    },
    Let {
        name: Ident,
        value: NodeP,
    },
    Ref(Ident),
    ContinueToThunk(Ident, ThunkArgs),
    ContinueToExtern(u64),

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
    StoreMem(Box<Node>, Box<Node>),

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

pub fn ssa_to_ast(ssa: &ssa::Program) -> Ast {
    // the set of blocks in the CFG
    // is transformed
    // into a set of Thunk-s
    //
    // each thunk can result from one or more blocks, merged.

    let mut builder = Builder::new(ssa);

    let root_thunk = builder.compile_thunk(cfg::ENTRY_BID);
    while let Some(bid) = builder.next_unvisited_block() {
        builder.compile_thunk(bid);
    }
    let thunks = builder.finish();

    Ast { root_thunk, thunks }
}

struct Builder<'a> {
    ssa: &'a ssa::Program,
    name_of_value: HashMap<mil::Index, Ident>,
    thunk_id_of_block: cfg::BlockMap<Ident>,
    visited: cfg::BlockMap<bool>,
    thunks: HashMap<Ident, Thunk>,
}

impl<'a> Builder<'a> {
    fn new(ssa: &'a ssa::Program) -> Builder<'a> {
        let name_of_value = (0..ssa.len())
            .filter_map(|ndx| {
                if ssa.readers_count(mil::Reg(ndx)) > 1 {
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
        let thunk_id_of_block = cfg::BlockMap::new_with(ssa.cfg(), |bid| {
            Ident(Rc::new(format!("T{}", bid.as_number())))
        });

        Builder {
            ssa,
            name_of_value,
            thunk_id_of_block,
            visited: cfg::BlockMap::new(false, ssa.cfg().block_count()),
            thunks: HashMap::new(),
        }
    }

    fn next_unvisited_block(&self) -> Option<cfg::BasicBlockID> {
        self.visited
            .items()
            .find(|(_, visited)| !*visited)
            .map(|(bid, _)| bid)
    }

    fn finish(self) -> HashMap<Ident, Thunk> {
        self.thunks
    }

    fn compile_thunk(&mut self, start_bid: cfg::BasicBlockID) -> Ident {
        // create thunk and assign ID immediately, in case the thunk jumps to itself
        let lbl = self.thunk_id_of_block[start_bid].clone();
        self.thunks.insert(
            lbl.clone(),
            Thunk {
                body: Node::Nop,
                params: SmallVec::new(),
            },
        );

        let phis = self.ssa.block_phi(start_bid);
        let phi_count = phis.phi_count();
        self.thunks.get_mut(&lbl).unwrap().params = (0..phi_count)
            .filter(|phi_ndx| self.ssa.is_alive(phis.node_ndx(*phi_ndx)))
            .map(|phi_ndx| Ident(Rc::new(format!("phi{}", phi_ndx))))
            .collect();

        let nor_ndxs = self.ssa.cfg().insns_ndx_range(start_bid);

        let mut seq: SmallVec<_> = nor_ndxs
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
            })
            .collect();

        match self.ssa.cfg().successors(start_bid) {
            cfg::BlockCont::End => {
                // all done!
            }
            cfg::BlockCont::Jmp((pred_ndx, tgt)) => {
                let cont_node = self.compile_continue(*tgt, *pred_ndx).boxed();
                seq.push(cont_node);
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
                    mil::Insn::JmpIfK { cond, target: _ } => cond,
                    _ => panic!("block with BlockCont::Alt continuation must end with a JmpIfK"),
                };
                let cond = self.get_node(cond.0).boxed();
                let cons = self.compile_continue(*neg_bid, *neg_pred_ndx).boxed();
                let alt = self.compile_continue(*pos_bid, *pos_pred_ndx).boxed();

                seq.push(Node::If { cond, cons, alt }.boxed());
            }
        };

        self.thunks.get_mut(&lbl).unwrap().body = Node::Seq(seq);
        self.visited[start_bid] = true;
        lbl
    }

    fn compile_continue(&self, bid: cfg::BasicBlockID, pred_ndx: u8) -> Node {
        let thunk_id = self.thunk_id_of_block[bid].clone();

        let target_phis = self.ssa.block_phi(bid);
        let values = (0..target_phis.phi_count())
            .filter(|phi_ndx| self.ssa.is_alive(target_phis.node_ndx(*phi_ndx)))
            .map(|phi_ndx| {
                let reg = target_phis.arg(self.ssa, phi_ndx, pred_ndx.into());
                self.get_node(reg.0).boxed()
            })
            .collect();
        let args = ThunkArgs { values };

        Node::ContinueToThunk(thunk_id, args)
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
            Insn::JmpK(target) => Node::ContinueToExtern(*target),
            Insn::Jmp(_) | Insn::JmpIfK { .. } => {
                // skip.  the control fllow handling in compile_thunk shall take care of this
                Node::Nop
            }
            Insn::StoreMem(addr, val) => {
                let addr = self.get_node(addr.0).boxed();
                let val = self.get_node(val.0).boxed();
                Node::StoreMem(addr, val)
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
}

fn fold_bin(op: BinOp, a: Box<Node>, b: Box<Node>) -> Node {
    // TODO!
    Node::Bin {
        op,
        args: [a, b].into(),
    }
}

impl Ast {
    pub fn pretty_print<W: std::fmt::Write>(
        &self,
        pp: &mut crate::pp::PrettyPrinter<W>,
    ) -> std::fmt::Result {
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
    pub fn pretty_print<W: std::fmt::Write>(
        &self,
        pp: &mut crate::pp::PrettyPrinter<W>,
    ) -> std::fmt::Result {
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
    pub fn pretty_print<W: std::fmt::Write>(
        &self,
        pp: &mut crate::pp::PrettyPrinter<W>,
    ) -> std::fmt::Result {
        use std::fmt::Write;
        match self {
            Node::Seq(nodes) => {
                writeln!(pp, "{{")?;
                write!(pp, "  ")?;
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

                write!(pp, " {{\n  ")?;
                pp.open_box();
                cons.pretty_print(pp)?;
                pp.close_box();

                write!(pp, "\n}} else {{\n  ")?;
                pp.open_box();
                alt.pretty_print(pp)?;
                pp.close_box();
                write!(pp, "\n}}")
            }

            Node::Let { name, value } => {
                write!(pp, "let {} = ", name.0.as_str())?;
                pp.open_box();
                value.pretty_print(pp)?;
                pp.close_box();
                Ok(())
            }

            Node::Ref(ident) => write!(pp, "{}", ident.0.as_str()),

            Node::ContinueToThunk(thunk_id, args) => {
                write!(pp, "goto {}", thunk_id.0.as_str())?;
                if args.values.len() > 0 {
                    write!(pp, " with ")?;
                    pp.open_box();
                    for (ndx, arg) in args.values.iter().enumerate() {
                        if ndx == 0 {
                            write!(pp, "( ")?;
                        } else {
                            write!(pp, "\n, ")?;
                        }
                        pp.open_box();
                        arg.pretty_print(pp)?;
                        pp.close_box();
                    }
                    write!(pp, "\n)")?;
                    pp.close_box();
                }
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
                a.pretty_print(pp)?;
                write!(pp, " with l1 = ")?;
                b.pretty_print(pp)
            }
            Node::WithL2(a, b) => {
                a.pretty_print(pp)?;
                write!(pp, " with l2 = ")?;
                b.pretty_print(pp)
            }
            Node::WithL4(a, b) => {
                a.pretty_print(pp)?;
                write!(pp, " with l4 = ")?;
                b.pretty_print(pp)
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

            Node::LoadMem1(arg)
            | Node::LoadMem2(arg)
            | Node::LoadMem4(arg)
            | Node::LoadMem8(arg) => {
                write!(pp, "(")?;
                arg.pretty_print(pp)?;
                write!(pp, ").*")
            }
            Node::StoreMem(dest, val) => {
                write!(pp, "(")?;
                dest.pretty_print(pp)?;
                write!(pp, ").* <- ")?;
                pp.open_box();
                val.pretty_print(pp)?;
                pp.close_box();
                Ok(())
            }
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
