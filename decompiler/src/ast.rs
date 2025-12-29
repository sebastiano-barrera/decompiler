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
#[derive(serde::Serialize, Debug, Clone, PartialEq, Eq)]
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

    /// Represents a "no op", doing nothing.
    ///
    /// Useful as placeholder (e.g., as an empty body a "then" or "else"
    /// clause). It is used as a temporary placeholder during the `cleanup`
    /// algorithm.
    Pass,
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
    ///
    /// Note that each Stmt can only be "pointed to" by one other statement
    /// (unless it's *the* root Stmt). For this reason, it's appropriate to say
    /// that each Stmt is "contained" by exactly one other Stmt (unless it's the
    /// root). In other words, a StmtID can only appear in one other Stmt whose
    /// StmtID is higher (unless it's the root stataement, which has the highest
    /// StmtID).
    ///
    /// This enables "moving" Stmts in the tree by copying over Stmt nodes.
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

    pub fn assert_invariants(&self) {
        self.assert_graph_is_tree();
    }

    fn assert_graph_is_tree(&self) {
        let mut has_parent = vec![false; self.nodes.len()];
        let mut work = vec![self.root()];
        while let Some(sid) = work.pop() {
            let had_parent = std::mem::replace(&mut has_parent[sid.0], true);
            assert!(!had_parent);

            match self.get(sid) {
                Stmt::NamedBlock { bid: _, body }
                | Stmt::Let {
                    name: _,
                    value: _,
                    body,
                }
                | Stmt::LetPhi { name: _, body } => work.push(*body),
                Stmt::Seq { first, then } => {
                    work.push(*first);
                    work.push(*then);
                }
                Stmt::If { cond: _, cons, alt } => {
                    work.push(*cons);
                    work.push(*alt);
                }
                Stmt::Eval(_)
                | Stmt::Return(_)
                | Stmt::JumpUndefined
                | Stmt::JumpExternal(_)
                | Stmt::JumpIndirect(_)
                | Stmt::Loop(_)
                | Stmt::Jump(_)
                | Stmt::Pass => {}
            }
        }
    }
}

impl Ast {
    // private mutation API
    // used by AstBuilder
    fn push_stmt(&mut self, stmt: Stmt) -> StmtID {
        self.nodes.push(stmt);
        self.root()
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

    // state variables
    let_emitted: HashSet<Reg>,
    block_printed: cfg::BlockMap<bool>,

    // state & output
    ast: Ast,
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
            block_order_rev: Vec::new(),
            let_emitted: HashSet::new(),
            block_printed: cfg::BlockMap::new(ssa.cfg(), false),
            ast: Ast {
                nodes: Vec::new(),
                is_named,
                block_order: Vec::new(),
            },
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
        self.ast.is_named[reg]
    }

    pub fn build(mut self) -> Ast {
        for (_, value) in self.block_printed.items_mut() {
            *value = false;
        }

        self.ast.block_order = {
            let mut ord = Vec::with_capacity(self.block_order_rev.len());
            ord.extend(self.block_order_rev.iter().rev().cloned());
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

        #[cfg(debug_assertions)]
        self.ast.assert_invariants();

        cleanup::cleanup(&mut self.ast);

        #[cfg(debug_assertions)]
        self.ast.assert_invariants();

        self.ast
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

    fn push_stmt(&mut self, stmt: Stmt) -> StmtID {
        self.ast.push_stmt(stmt)
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

/// AST cleanup algorithm
mod cleanup {
    use crate::BlockID;

    use super::*;

    /// Process the given AST to make it easier to read by a human being.
    ///
    /// The processing done by this function is only cosmetic, and not
    /// particularly sophisticated.
    ///
    /// # Implementation notes
    ///
    /// More articulate transformations should be done at the SSA stage
    /// (including CFG analysis, if appropriate) instead of here. This is only
    /// for surface-level changes.
    pub(super) fn cleanup(ast: &mut Ast) {
        let root_sid_pre = ast.root();
        cleanup_stmt(ast, root_sid_pre, Cont::End);

        // little hack: `cleanup_stmt` may add new nodes via push_stmt, which
        // violates the invariant that the root node MUST be the last in the
        // `nodes` Vec. this restores it (at the cost of maybe one extra copy of
        // the node)
        if ast.root() != root_sid_pre {
            ast.push_stmt(ast.get(root_sid_pre).clone());
        }

        remove_unused_labels(ast);
    }

    /// How a Stmt is expected to "continue", i.e. where control flow is led
    /// after the Stmt is "run".
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Cont {
        /// There is one statement coming naturally after this one (e.g. in a
        /// Seq), which is not labeled (it's not, for example, a block).
        Fallthrough,

        /// Jump "out" of the function.
        ///
        /// Control flow exits the function entirely, never to return to any
        /// other Stmt within it.
        ///
        /// This includes both the natural end of the function, or any form of
        /// return or external jump.
        End,

        /// Jump to a specific block within the function
        ToBlock(BlockID),
    }

    /// Run the cleanup algorithm on a specific statement.
    ///
    /// Recurs on all child statements.
    ///
    /// Parameter [`nat_cont`] represents the statement's "natural
    /// continuation": where control flow would normally go after this statement
    /// (assuming that its *does* continue; i.e., that it's not a return or
    /// external jump.)
    ///
    /// Returns the statement's actual continuation (including the effect of
    /// any change).
    fn cleanup_stmt(ast: &mut Ast, sid: StmtID, nat_cont: Cont) -> Cont {
        match ast.nodes[sid.0].clone() {
            Stmt::NamedBlock { bid: _, body }
            | Stmt::Let {
                name: _,
                value: _,
                body,
            }
            | Stmt::LetPhi { name: _, body } => cleanup_stmt(ast, body, nat_cont),

            Stmt::Seq { first, then } => {
                let cont_from_then = cleanup_stmt(ast, then, nat_cont);
                if ast.nodes[then.0] == Stmt::Pass {
                    move_node_tofrom(ast, sid, first);
                    return cleanup_stmt(ast, sid, nat_cont);
                }

                let cont_to_then = cont_to(ast, then);
                cleanup_stmt(ast, first, cont_to_then);
                if ast.nodes[first.0] == Stmt::Pass {
                    move_node_tofrom(ast, sid, then);
                    return cleanup_stmt(ast, sid, nat_cont);
                }

                cont_from_then
            }

            Stmt::If { cond, cons, alt } => {
                let cons_cont = cleanup_stmt(ast, cons, nat_cont);
                let _alt_cont = cleanup_stmt(ast, alt, nat_cont);

                if cons_cont != Cont::Fallthrough {
                    // hoist else
                    let empty_alt = ast.push_stmt(Stmt::Pass);
                    let new_if = ast.push_stmt(Stmt::If {
                        cond,
                        cons,
                        alt: empty_alt,
                    });
                    ast.nodes[sid.0] = Stmt::Seq {
                        first: new_if,
                        then: alt,
                    };
                }

                Cont::Fallthrough
            }

            Stmt::Loop(bid) | Stmt::Jump(bid) => {
                if let Cont::ToBlock(cont_bid) = nat_cont {
                    if cont_bid == bid {
                        ast.nodes[sid.0] = Stmt::Pass;
                        return Cont::Fallthrough;
                    }
                }

                Cont::ToBlock(bid)
            }

            Stmt::Eval(_) | Stmt::Pass => Cont::Fallthrough,

            Stmt::Return(_)
            | Stmt::JumpUndefined
            | Stmt::JumpExternal(_)
            | Stmt::JumpIndirect(_) => Cont::End,
        }
    }

    fn cont_to(ast: &mut Ast, sid: StmtID) -> Cont {
        match ast.nodes[sid.0] {
            Stmt::NamedBlock { bid, body: _ } => Cont::ToBlock(bid),
            _ => Cont::Fallthrough,
        }
    }
    fn move_node_tofrom(ast: &mut Ast, to: StmtID, from: StmtID) {
        ast.nodes[to.0] = std::mem::replace(&mut ast.nodes[from.0], Stmt::Pass);
    }

    fn remove_unused_labels(ast: &mut Ast) {
        let mut is_used = HashSet::new();

        for stmt in ast.nodes.iter() {
            if let Stmt::Jump(bid) | Stmt::Loop(bid) = stmt {
                is_used.insert(*bid);
            }
        }

        for stmt_ndx in 0..ast.nodes.len() {
            let sid = StmtID(stmt_ndx);
            loop {
                if let Stmt::NamedBlock { bid, body } = ast.nodes[stmt_ndx] {
                    if !is_used.contains(&bid) {
                        move_node_tofrom(ast, sid, body);
                        continue;
                    }
                }

                break;
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::R;

        use super::*;

        // rules to check:
        //
        // - remove empty then/else branch
        //
        // - remove unused block labels

        #[test]
        fn remove_pass_from_seq() {
            let mut ast = mk_simple_ast(vec![
                Stmt::Eval(R(0)),
                Stmt::Pass,
                Stmt::Eval(R(2)),
                Stmt::Seq {
                    first: StmtID(0),
                    then: StmtID(1),
                },
                Stmt::Seq {
                    first: StmtID(3),
                    then: StmtID(2),
                },
            ]);

            cleanup(&mut ast);

            let &Stmt::Seq { first, then } = ast.get(ast.root()) else {
                panic!()
            };
            assert_eq!(ast.get(first), &Stmt::Eval(R(0)));
            assert_eq!(ast.get(then), &Stmt::Eval(R(2)));
        }

        fn mk_simple_ast(nodes: Vec<Stmt>) -> Ast {
            let ast = Ast {
                nodes,
                is_named: RegMap::empty(),
                // wrong, but the cleanup algorithm doesn't care about this
                block_order: Vec::new(),
            };
            ast.assert_invariants();
            ast
        }

        #[test]
        fn fallthrough_to_block() {
            let bid = BlockID::from_number(3);
            let mut ast = mk_simple_ast(vec![
                Stmt::Eval(R(0)),
                Stmt::Jump(bid),
                Stmt::Eval(R(2)),
                // 3
                Stmt::Seq {
                    first: StmtID(0),
                    then: StmtID(1),
                },
                Stmt::NamedBlock {
                    bid,
                    body: StmtID(2),
                },
                Stmt::Seq {
                    first: StmtID(3),
                    then: StmtID(4),
                },
            ]);

            cleanup(&mut ast);

            let &Stmt::Seq {
                first: block_jumper,
                then: _,
            } = ast.get(ast.root())
            else {
                panic!();
            };
            assert_eq!(ast.get(block_jumper), &Stmt::Eval(R(0)));
        }

        #[test]
        fn remove_unused_block_labels() {
            let bid = BlockID::from_number(3);
            let other_bid = BlockID::from_number(4);
            let mut ast = mk_simple_ast(vec![
                Stmt::Eval(R(0)),
                Stmt::Jump(other_bid),
                Stmt::Eval(R(2)),
                // 3
                Stmt::Seq {
                    first: StmtID(0),
                    then: StmtID(1),
                },
                Stmt::NamedBlock {
                    bid,
                    body: StmtID(2),
                },
                Stmt::Seq {
                    first: StmtID(3),
                    then: StmtID(4),
                },
            ]);

            cleanup(&mut ast);

            let &Stmt::Seq {
                first: _,
                then: block_label,
            } = ast.get(ast.root())
            else {
                panic!();
            };
            assert_eq!(ast.get(block_label), &Stmt::Eval(R(2)));
        }

        #[test]
        fn remove_unused_block_labels_neg() {
            let bid = BlockID::from_number(3);
            let mut ast = mk_simple_ast(vec![
                Stmt::Eval(R(0)),
                Stmt::Jump(bid),
                Stmt::Eval(R(1)),
                Stmt::Eval(R(2)),
                // 4
                Stmt::Seq {
                    first: StmtID(0),
                    then: StmtID(1),
                },
                Stmt::Seq {
                    first: StmtID(4),
                    then: StmtID(2),
                },
                Stmt::NamedBlock {
                    bid,
                    body: StmtID(3),
                },
                Stmt::Seq {
                    first: StmtID(5),
                    then: StmtID(6),
                },
            ]);

            cleanup(&mut ast);

            let &Stmt::Seq {
                first: _,
                then: block_label,
            } = ast.get(ast.root())
            else {
                panic!();
            };
            assert_eq!(
                ast.get(block_label),
                &Stmt::NamedBlock {
                    bid,
                    body: StmtID(3),
                }
            );
        }

        #[test]
        #[should_panic]
        fn cyclic_detected() {
            mk_simple_ast(vec![
                Stmt::Eval(R(0)),
                Stmt::Jump(BlockID::from_number(3)),
                Stmt::Eval(R(1)),
                // 3
                Stmt::Seq {
                    first: StmtID(0),
                    then: StmtID(1),
                },
                Stmt::If {
                    cond: Some(R(2)),
                    cons: StmtID(4),
                    alt: StmtID(2),
                },
            ]);
        }

        /// In an if-then-else clause, if the consequent (`then`) block does not
        /// fallthrough to the next statement, the alternate (`else`) block can
        /// be hoised out of the if (with an appropriate wrapping Seq).
        #[test]
        fn if_then_else_hoist_else() {
            let mut ast = mk_simple_ast(vec![
                Stmt::Eval(R(0)),
                Stmt::Jump(BlockID::from_number(3)),
                Stmt::Eval(R(1)),
                // 3
                Stmt::Seq {
                    first: StmtID(0),
                    then: StmtID(1),
                },
                Stmt::If {
                    cond: Some(R(2)),
                    cons: StmtID(3),
                    alt: StmtID(2),
                },
            ]);

            cleanup(&mut ast);

            let root = ast.get(ast.root());
            let &Stmt::Seq { first, then } = root else {
                panic!();
            };

            {
                let &Stmt::If { cond: _, cons, alt } = ast.get(first) else {
                    panic!()
                };
                assert_eq!(cons, StmtID(3));
                assert_eq!(ast.get(alt), &Stmt::Pass);
            }

            assert_eq!(then, StmtID(2));
        }

        #[test]
        fn if_then_else_hoist_else_neg() {
            let mut ast = mk_simple_ast(vec![
                Stmt::Eval(R(0)),
                Stmt::Eval(R(1)),
                Stmt::If {
                    cond: Some(R(2)),
                    cons: StmtID(0),
                    alt: StmtID(1),
                },
            ]);

            cleanup(&mut ast);

            assert_eq!(
                ast.get(ast.root()),
                &Stmt::If {
                    cond: Some(R(2)),
                    cons: StmtID(0),
                    alt: StmtID(1),
                },
            );
        }
    }
}
