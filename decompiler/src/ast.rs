use std::{collections::HashSet, result::Result};

use thiserror::Error;

use crate::{
    cfg,
    mil::{ArithOp, Insn, Reg},
    ssa, RegMap,
};

mod pp;
pub use pp::write_ast;

// TODO intern strings? (check performance gain)
#[derive(serde::Serialize, Debug, Clone)]
#[serde(tag = "stmt", content = "arg")]
pub enum Stmt {
    // TODO add a table to Ast for assigning custom names to blocks (and use
    // them in labels)
    NamedBlock {
        bid: cfg::BlockID,
        body: StmtID,
    },
    Let {
        name: String,
        value: Reg,
        body: StmtID,
    },
    LetPhi {
        name: String,
        body: StmtID,
    },
    Seq {
        first: StmtID,
        then: StmtID,
    },
    Eval(Reg),

    /// An "If" statement.
    ///
    /// `cond` is only `None` if the condition value could not be located in the
    /// input SSA (which is a bug, but one that AST is robust against).
    If {
        cond: Option<Reg>,
        cons: StmtID,
        alt: StmtID,
    },

    Return(Reg),
    JumpUndefined,
    JumpExternal(u64),
    JumpIndirect(Reg),
    Loop(cfg::BlockID),
    Jump(cfg::BlockID),
}

#[derive(Clone, Copy, serde::Serialize, Debug, PartialEq, Eq)]
#[serde(transparent)]
pub struct StmtID(usize);

/// The abstract syntax tree, as finally output by the decompiler.
///
/// [Ast] values are created via [AstBuilder], which allows customizing certain
/// aspects of the AST, such as the block order.
#[derive(Debug)]
pub struct Ast {
    // Actually, this data structure only contains enough data to *pretend* that
    // there is a full AST inside. The public APIs are supposed to create the
    // illusion of operating on a typed AST that is actually a tree.
    /// All statements in the AST, indexed by [StmtID].
    nodes: Vec<Stmt>,
    is_named: RegMap<bool>,
    block_order: Vec<cfg::BlockID>,
}
impl Ast {
    pub fn root(&self) -> StmtID {
        StmtID(self.nodes.len() - 1)
    }

    pub fn stmt_ids(&self) -> impl Iterator<Item = StmtID> {
        (0..self.nodes.len()).map(StmtID)
    }

    pub fn get(&self, sid: StmtID) -> &Stmt {
        &self.nodes[sid.0]
    }

    pub fn is_value_named(&self, reg: Reg) -> bool {
        self.is_named[reg]
    }

    pub fn block_order(&self) -> &[cfg::BlockID] {
        &self.block_order
    }
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("block order contains duplicates")]
    BlockOrderHasDuplicates,
}

pub struct AstBuilder<'a> {
    // inputs
    ssa: &'a ssa::Program,
    block_order_rev: Vec<cfg::BlockID>,

    // input, state & output
    is_named: ssa::RegMap<bool>,

    // state variables
    let_emitted: HashSet<Reg>,
    block_printed: cfg::BlockMap<bool>,

    // output
    nodes: Vec<Stmt>,
}

impl<'a> AstBuilder<'a> {
    pub fn new(ssa: &'a ssa::Program) -> Self {
        let rdr_count = ssa::count_readers(ssa);
        let is_named = rdr_count.map(|reg, count| {
            let Some(insn) = ssa.get(reg) else {
                return false;
            };
            // ancestral are as good as r# refs, so never 'name' them / always print inline
            matches!(insn, Insn::Phi)
                || (*count > 1
                    && !matches!(insn, Insn::StoreMem { .. })
                    && !matches!(insn, Insn::Ancestral { .. })
                    && !matches!(insn, Insn::Int { .. }))
        });

        let mut builder = AstBuilder {
            ssa,
            is_named,
            block_order_rev: Vec::new(),
            let_emitted: HashSet::new(),
            block_printed: cfg::BlockMap::new(ssa.cfg(), false),
            nodes: Vec::new(),
        };

        // default block order: reverse postorder
        // TODO replace with dominator tree walked in reverse postorder
        builder
            .set_block_order(ssa.cfg().block_ids_rpo().collect())
            .unwrap();

        builder
    }

    pub fn set_block_order(&mut self, mut block_order: Vec<cfg::BlockID>) -> Result<(), Error> {
        let mut count_of_bid = cfg::BlockMap::new(self.ssa.cfg(), 0);

        for &bid in &block_order {
            count_of_bid[bid] += 1;
        }

        for (_, &count) in count_of_bid.items() {
            if count > 1 {
                return Err(Error::BlockOrderHasDuplicates);
            }
        }

        // NOTE: reversed so that we can "peek" into it by inspecting the last
        // element and/or doing .pop() to advance
        block_order.reverse();
        self.block_order_rev = block_order;
        Ok(())
    }

    fn is_named(&self, reg: Reg) -> bool {
        self.is_named[reg]
    }

    pub fn build(mut self) -> Ast {
        for (_, value) in self.block_printed.items_mut() {
            *value = false;
        }

        let block_order = {
            let mut ord = self.block_order_rev.clone();
            ord.reverse();
            ord
        };

        let mut block_heads = Vec::with_capacity(self.block_order_rev.len());
        while let Some(bid) = self.block_order_rev.pop() {
            block_heads.push(self.build_block(bid));
        }

        block_heads
            .into_iter()
            .fold(None, |sid, block_sid| match sid {
                Some(sid) => Some(self.push_stmt(Stmt::Seq {
                    first: sid,
                    then: block_sid,
                })),
                None => Some(block_sid),
            });

        Ast {
            nodes: self.nodes,
            is_named: self.is_named,
            block_order,
        }
    }

    fn build_block(&mut self, bid: cfg::BlockID) -> StmtID {
        let cfg = self.ssa.cfg();

        let end_sid: StmtID = match cfg.block_cont(bid) {
            cfg::BlockCont::Always(tgt) => self.build_continuation(bid, tgt),
            cfg::BlockCont::Conditional { pos, neg } => {
                let cond = self.ssa.find_last_matching(bid, |insn| match insn {
                    Insn::SetJumpCondition(value) => Some(value),
                    _ => None,
                });
                let cons = self.build_continuation(bid, pos);
                let alt = self.build_continuation(bid, neg);

                self.push_stmt(Stmt::If { cond, cons, alt })
            }
        };

        let mut sid = end_sid;
        for reg in self.ssa.block_regs(bid).rev() {
            if self.is_named(reg) || self.ssa.get(reg).unwrap().has_side_effects() {
                sid = self.build_stmt(reg, sid);
            }
        }

        // if cfg.block_preds(bid).len() == 1 {
        self.push_stmt(Stmt::NamedBlock { bid, body: sid })
        // } else {
        //     sid
        // }
    }

    fn build_continuation(&mut self, src_bid: cfg::BlockID, tgt: cfg::Dest) -> StmtID {
        match tgt {
            cfg::Dest::Ext(addr) => self.push_stmt(Stmt::JumpExternal(addr)),
            cfg::Dest::Block(tgt_bid) => {
                if self.block_order_rev.last().copied() == Some(tgt_bid) &&
                    // tgt has only one predecessor (then it must be us)
                    self.ssa.cfg().block_preds(tgt_bid).len() == 1
                {
                    debug_assert_eq!(self.ssa.cfg().block_preds(tgt_bid), &[src_bid]);
                    self.block_order_rev.pop();
                    self.build_block(tgt_bid)
                } else {
                    // TODO emit Stmt::Loop instead if we know we're looping back to an open block
                    self.push_stmt(Stmt::Jump(tgt_bid))
                }
            }
            cfg::Dest::Indirect => {
                let target = self.ssa.find_last_matching(src_bid, |insn| match insn {
                    Insn::SetJumpTarget(value) => Some(value),
                    _ => None,
                });

                match target {
                    Some(target) => self.push_stmt(Stmt::JumpIndirect(target)),
                    None => {
                        // TODO: report a bug: "internal bug: unspecified jump target"
                        self.push_stmt(Stmt::JumpUndefined)
                    }
                }
            }
            cfg::Dest::Return => {
                let ret_val = self.ssa.find_last_matching(src_bid, |insn| match insn {
                    Insn::SetReturnValue(value) => Some(value),
                    _ => None,
                });

                match ret_val {
                    Some(ret_val) => self.push_stmt(Stmt::Return(ret_val)),
                    None => {
                        // TODO: report a bug: "actually unspecified in source program!"
                        self.push_stmt(Stmt::JumpUndefined)
                    }
                }
            }
            cfg::Dest::Undefined => {
                // TODO: report a bug: "warning: due to decompiler bug or limitation"
                self.push_stmt(Stmt::JumpUndefined)
            }
        }
    }

    /// Prints a single "statement line" in a block.
    ///
    /// Named instructions are wrapped in a `let x = ...;`.
    fn build_stmt(&mut self, reg: Reg, sid: StmtID) -> StmtID {
        // NOTE unlike build_def, this function is called only when printing the
        // "toplevel" definition of a named or effectful instruction;
        //
        // This is called by pp_labeled_inputs

        // Check: named values are converted to a single let stmt, never twice
        assert!(!self.let_emitted.contains(&reg));
        self.let_emitted.insert(reg);

        match self.ssa.get(reg).unwrap() {
            // These are managed in the handling of if/return/etc.
            Insn::SetJumpCondition(_) | Insn::SetJumpTarget(_) | Insn::SetReturnValue(_) => sid,
            Insn::Phi => {
                let name = format!("r{}", reg.reg_index());
                self.push_stmt(Stmt::LetPhi { name, body: sid })
            }
            _ if self.is_named(reg) => {
                let name = format!("r{}", reg.reg_index());
                self.push_stmt(Stmt::Let {
                    name,
                    value: reg,
                    body: sid,
                })
            }
            _ => {
                let eval = self.push_stmt(Stmt::Eval(reg));
                self.push_stmt(Stmt::Seq {
                    first: eval,
                    then: sid,
                })
            }
        }
    }

    fn next_stmt_id(&mut self) -> StmtID {
        StmtID(self.nodes.len())
    }

    fn push_stmt(&mut self, stmt: Stmt) -> StmtID {
        let sid = self.next_stmt_id();
        self.nodes.push(stmt);
        sid
    }
}

pub type PrecedenceLevel = u8;

pub fn precedence(insn: &Insn) -> PrecedenceLevel {
    // higher value == higher precedence == evaluated first unless parenthesized
    match insn {
        // actually, Insn::Get is supposed to already be "resolved" to its argument
        // prior to calling this function, but better to return something "safe"
        Insn::Get(_)
        | Insn::Void
        | Insn::True
        | Insn::False
        | Insn::Int { .. }
        | Insn::Bytes(..)
        | Insn::Global(_)
        | Insn::UndefinedBool
        | Insn::UndefinedBytes { .. }
        | Insn::Ancestral { .. }
        | Insn::FuncArgument { .. }
        | Insn::Phi
        | Insn::StructGetMember { .. }
        | Insn::ArrayGetElement { .. }
        | Insn::LoadMem { .. } => 255,

        Insn::Part { .. } => 254,
        Insn::Concat { .. } => 253,
        Insn::Struct { .. } | Insn::StructMember { .. } => 251,
        Insn::Call { .. } => 250,
        Insn::CArg { .. } => 250,
        Insn::Widen { .. } => 248,

        Insn::Arith(op, _, _) | Insn::ArithK(op, _, _) => match op {
            ArithOp::Shl | ArithOp::Shr | ArithOp::BitXor | ArithOp::BitAnd | ArithOp::BitOr => 212,
            ArithOp::Add | ArithOp::Sub => 210,
            ArithOp::Mul => 211,
        },
        Insn::OverflowOf(_) => 200,
        Insn::CarryOf(_) => 200,
        Insn::SignOf(_) => 200,
        Insn::IsZero(_) => 200,
        Insn::Parity(_) => 200,

        Insn::Bool(_, _, _) => 199,
        Insn::Not(_) => 199,
        Insn::Cmp(_, _, _) => 197,

        // effectful instructions are basically *always* done last due to their
        // position in the printed syntax
        Insn::SetReturnValue(_)
        | Insn::SetJumpTarget(_)
        | Insn::SetJumpCondition(_)
        | Insn::Control(_)
        | Insn::NotYetImplemented(_)
        | Insn::Upsilon { .. }
        | Insn::StoreMem { .. } => 0,
        // Insn::CArg { .. } => panic!("CArg undefined precedence"),
    }
}
