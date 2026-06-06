use super::*;

use crate::mil::Reg;

use std::collections::HashSet;

pub fn build(builder: AstBuilder) -> Ast {
    State::start(builder).build()
}

struct State<'a> {
    ssa: &'a ssa::Program,
    block_order_rev: Vec<cfg::BlockID>,
    let_emitted: HashSet<Reg>,
    block_printed: cfg::BlockMap<bool>,
    ast: Ast,
    is_all_blocks_named: bool,
}

impl<'a> State<'a> {
    fn start(builder: AstBuilder<'a>) -> Self {
        let rdr_count = ssa::count_readers(builder.ssa);
        let is_named = rdr_count.map(|reg, count| {
            let Some(insn) = builder.ssa.get(reg) else {
                return false;
            };
            // ancestral are as good as r# refs, so never 'name' them / always
            // print inline
            matches!(insn, Insn::Phi)
                || (*count > 1
                    && !matches!(insn, Insn::StoreMem { .. })
                    && !matches!(insn, Insn::Ancestral { .. })
                    && !matches!(insn, Insn::Int { .. }))
        });

        let block_order_rev = builder.block_order.into_iter().rev().collect();
        State {
            ssa: builder.ssa,
            block_order_rev,
            let_emitted: HashSet::new(),
            block_printed: cfg::BlockMap::new(builder.ssa.cfg(), false),
            ast: Ast {
                nodes: Vec::new(),
                is_named,
                block_order: Vec::new(),
            },
            is_all_blocks_named: builder.is_all_blocks_named,
        }
    }

    fn build(mut self) -> Ast {
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
        if !self.is_all_blocks_named {
            cleanup::remove_unused_labels(&mut self.ast);
        }

        #[cfg(debug_assertions)]
        self.ast.assert_invariants();

        self.ast
    }

    fn is_named(&self, reg: Reg) -> bool {
        self.ast.is_named[reg]
    }

    fn build_block(&mut self, bid: cfg::BlockID) -> StmtID {
        let cfg = self.ssa.cfg();

        let end_sid: StmtID = match cfg.block_cont(bid) {
            cfg::BlockCont::Always(tgt) => self.build_continuation(bid, tgt),
            cfg::BlockCont::Conditional { pos, neg } => {
                let cond = self.ssa.find_last_matching(bid, |insn| match insn {
                    Insn::SetJumpCondition(value) => Some(*value),
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
        // } else { sid }
    }

    fn build_continuation(&mut self, src_bid: cfg::BlockID, tgt: cfg::Dest) -> StmtID {
        match tgt {
            cfg::Dest::Ext(addr) => self.push_stmt(Stmt::JumpExternal(addr)),
            cfg::Dest::Block(tgt_bid) => {
                if self.block_order_rev.last() == Some(&tgt_bid)
                // tgt has only one predecessor (then it must be us)
                && self.ssa.cfg().block_preds(tgt_bid).len() == 1
                {
                    debug_assert_eq!(self.ssa.cfg().block_preds(tgt_bid), &[src_bid]);
                    self.block_order_rev.pop();
                    self.build_block(tgt_bid)
                } else {
                    // TODO emit Stmt::Loop instead if we know we're looping
                    // back to an open block
                    self.push_stmt(Stmt::Jump(tgt_bid))
                }
            }
            cfg::Dest::Indirect => {
                let target = self.ssa.find_last_matching(src_bid, |insn| match insn {
                    Insn::SetJumpTarget(value) => Some(*value),
                    _ => None,
                });

                match target {
                    Some(target) => self.push_stmt(Stmt::JumpIndirect(target)),
                    None => {
                        // TODO: report a bug: "internal bug: unspecified jump
                        // target"
                        self.push_stmt(Stmt::JumpUndefined)
                    }
                }
            }
            cfg::Dest::Return => {
                let ret_val = self.ssa.find_last_matching(src_bid, |insn| match insn {
                    Insn::SetReturnValue(value) => Some(*value),
                    _ => None,
                });

                match ret_val {
                    Some(ret_val) => self.push_stmt(Stmt::Return(ret_val)),
                    None => {
                        // TODO: report a bug: "actually unspecified in source
                        // program!"
                        self.push_stmt(Stmt::JumpUndefined)
                    }
                }
            }
            cfg::Dest::Undefined => {
                // TODO: report a bug: "warning: due to decompiler bug or
                // limitation"
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
