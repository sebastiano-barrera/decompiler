use std::{collections::HashSet, result::Result};

use thiserror::Error;

use crate::{
    cfg,
    mil::{ArithOp, Insn, Reg},
    ssa, RegMap,
};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::define_ancestral_name;

    #[ignore = "I don't remember what this test was for"]
    #[test]
    fn test() {
        let ssa = sample_program();
        let ast = AstBuilder::new(&ssa).build();
        println!("ast = {:#?}", ast);
        panic!();
    }

    // fakes of the ones in x86_to_mil, to avoid having a dependency
    define_ancestral_name!(ANC_PF, "PF");
    define_ancestral_name!(ANC_AF, "AF");
    define_ancestral_name!(ANC_ZF, "ZF");
    define_ancestral_name!(ANC_SF, "SF");
    define_ancestral_name!(ANC_TF, "TF");
    define_ancestral_name!(ANC_IF, "IF");
    define_ancestral_name!(ANC_DF, "DF");
    define_ancestral_name!(ANC_OF, "OF");
    define_ancestral_name!(ANC_RBP, "RBP");
    define_ancestral_name!(ANC_RSP, "RSP");
    define_ancestral_name!(ANC_RIP, "RIP");
    define_ancestral_name!(ANC_RDI, "RDI");
    define_ancestral_name!(ANC_RSI, "RSI");
    define_ancestral_name!(ANC_RAX, "RAX");
    define_ancestral_name!(ANC_RBX, "RBX");
    define_ancestral_name!(ANC_RCX, "RCX");
    define_ancestral_name!(ANC_RDX, "RDX");
    define_ancestral_name!(ANC_R8, "R8");
    define_ancestral_name!(ANC_R9, "R9");
    define_ancestral_name!(ANC_R10, "R10");
    define_ancestral_name!(ANC_R11, "R11");
    define_ancestral_name!(ANC_R12, "R12");
    define_ancestral_name!(ANC_R13, "R13");
    define_ancestral_name!(ANC_R14, "R14");
    define_ancestral_name!(ANC_R15, "R15");

    #[rustfmt::skip]
    fn sample_program() -> crate::SSAProgram {
        // Extracted from an actual program
        //
        // Chosen because the control flow (and the set of instructions "called" here)
        // is varied enough to test for many things

        use crate::mil::{self, Insn, ArithOp, BoolOp,  Reg};

        let mut prog = mil::Program::new(Reg(0));

        // 0x0: Ancestral registers
        prog.push(Reg(3), Insn::Ancestral { anc_name: ANC_PF, reg_type: mil::RegType::Bool });
        prog.push(Reg(4), Insn::Ancestral { anc_name: ANC_AF, reg_type: mil::RegType::Bool });
        prog.push(Reg(5), Insn::Ancestral { anc_name: ANC_ZF, reg_type: mil::RegType::Bool });
        prog.push(Reg(6), Insn::Ancestral { anc_name: ANC_SF, reg_type: mil::RegType::Bool });
        prog.push(Reg(7), Insn::Ancestral { anc_name: ANC_TF, reg_type: mil::RegType::Bool });
        prog.push(Reg(8), Insn::Ancestral { anc_name: ANC_IF, reg_type: mil::RegType::Bool });
        prog.push(Reg(9), Insn::Ancestral { anc_name: ANC_DF, reg_type: mil::RegType::Bool });
        prog.push(Reg(10), Insn::Ancestral { anc_name: ANC_OF, reg_type: mil::RegType::Bool });
        prog.push(Reg(11), Insn::Ancestral { anc_name: ANC_RBP, reg_type: mil::RegType::Bytes(8) });
        prog.push(Reg(12), Insn::Ancestral { anc_name: ANC_RSP, reg_type: mil::RegType::Bytes(8) });
        prog.push(Reg(13), Insn::Ancestral { anc_name: ANC_RIP, reg_type: mil::RegType::Bytes(8) });
        prog.push(Reg(14), Insn::Ancestral { anc_name: ANC_RDI, reg_type: mil::RegType::Bytes(8) });
        prog.push(Reg(15), Insn::Ancestral { anc_name: ANC_RSI, reg_type: mil::RegType::Bytes(8) });
        prog.push(Reg(16), Insn::Ancestral { anc_name: ANC_RAX, reg_type: mil::RegType::Bytes(8) });
        prog.push(Reg(17), Insn::Ancestral { anc_name: ANC_RBX, reg_type: mil::RegType::Bytes(8) });
        prog.push(Reg(18), Insn::Ancestral { anc_name: ANC_RCX, reg_type: mil::RegType::Bytes(8) });
        prog.push(Reg(19), Insn::Ancestral { anc_name: ANC_RDX, reg_type: mil::RegType::Bytes(8) });
        prog.push(Reg(20), Insn::Ancestral { anc_name: ANC_R8, reg_type: mil::RegType::Bytes(8) });
        prog.push(Reg(21), Insn::Ancestral { anc_name: ANC_R9, reg_type: mil::RegType::Bytes(8) });
        prog.push(Reg(22), Insn::Ancestral { anc_name: ANC_R10, reg_type: mil::RegType::Bytes(8) });
        prog.push(Reg(23), Insn::Ancestral { anc_name: ANC_R11, reg_type: mil::RegType::Bytes(8) });
        prog.push(Reg(24), Insn::Ancestral { anc_name: ANC_R12, reg_type: mil::RegType::Bytes(8) });
        prog.push(Reg(25), Insn::Ancestral { anc_name: ANC_R13, reg_type: mil::RegType::Bytes(8) });
        prog.push(Reg(26), Insn::Ancestral { anc_name: ANC_R14, reg_type: mil::RegType::Bytes(8) });
        prog.push(Reg(27), Insn::Ancestral { anc_name: ANC_R15, reg_type: mil::RegType::Bytes(8) });

        // 0x486540
        prog.push(Reg(12), Insn::ArithK(ArithOp::Add, Reg(12), -8));
        prog.push(Reg(46), Insn::StoreMem { addr: Reg(12), value: Reg(11) });

        // 0x486541
        prog.push(Reg(11), Insn::Get(Reg(12)));

        // 0x486544
        prog.push(Reg(12), Insn::ArithK(ArithOp::Add, Reg(12), -8));
        prog.push(Reg(46), Insn::StoreMem { addr: Reg(12), value: Reg(25) });

        // 0x486546
        prog.push(Reg(12), Insn::ArithK(ArithOp::Add, Reg(12), -8));
        prog.push(Reg(46), Insn::StoreMem { addr: Reg(12), value: Reg(24) });

        // 0x486548
        prog.push(Reg(12), Insn::ArithK(ArithOp::Add, Reg(12), -8));
        prog.push(Reg(46), Insn::StoreMem { addr: Reg(12), value: Reg(17) });

        // 0x486549
        prog.push(Reg(17), Insn::Get(Reg(14)));

        // 0x48654c
        prog.push(Reg(14), Insn::Get(Reg(15)));

        // 0x48654f
        prog.push(Reg(46), Insn::Int { value: 8, size: 8 });
        prog.push(Reg(12), Insn::Arith(ArithOp::Sub, Reg(12), Reg(46)));
        prog.push(Reg(12), Insn::Get(Reg(12)));
        prog.push(Reg(10), Insn::OverflowOf(Reg(12)));
        prog.push(Reg(2), Insn::CarryOf(Reg(12)));
        prog.push(Reg(6), Insn::SignOf(Reg(12)));
        prog.push(Reg(5), Insn::IsZero(Reg(12)));
        prog.push(Reg(47), Insn::Part { src: Reg(12), offset: 0, size: 1 });
        prog.push(Reg(3), Insn::Parity(Reg(47)));

        // 0x486553
        prog.push(Reg(46), Insn::Int { value: 0, size: 8 });
        prog.push(Reg(46), Insn::Arith(ArithOp::Add, Reg(46), Reg(17)));
        prog.push(Reg(45), Insn::LoadMem { addr: Reg(46), size: 8 });
        prog.push(Reg(16), Insn::Get(Reg(45)));

        // 0x486556
        prog.push(Reg(46), Insn::Int { value: 0, size: 8 });
        prog.push(Reg(46), Insn::Arith(ArithOp::Add, Reg(46), Reg(16)));
        prog.push(Reg(45), Insn::LoadMem { addr: Reg(46), size: 8 });
        prog.push(Reg(24), Insn::Get(Reg(45)));

        // 0x486559
        prog.push(Reg(45), Insn::Int { value: 4575568, size: 8 });
        prog.push(Reg(47), Insn::Void);
        prog.push(Reg(47), Insn::Get(Reg(14)));
        prog.push(Reg(46), Insn::CArg { value: Reg(47), next_arg: None });
        prog.push(Reg(46), Insn::Call { callee: Reg(45), first_arg: Some(Reg(46)) });
        prog.push(Reg(2), Insn::UndefinedBool);
        prog.push(Reg(3), Insn::UndefinedBool);
        prog.push(Reg(4), Insn::UndefinedBool);
        prog.push(Reg(5), Insn::UndefinedBool);
        prog.push(Reg(6), Insn::UndefinedBool);
        prog.push(Reg(7), Insn::UndefinedBool);
        prog.push(Reg(8), Insn::UndefinedBool);
        prog.push(Reg(9), Insn::UndefinedBool);
        prog.push(Reg(10), Insn::UndefinedBool);
        prog.push(Reg(16), Insn::Part { src: Reg(46), offset: 0, size: 8 });

        // 0x48655e
        prog.push(Reg(46), Insn::Int { value: 8, size: 8 });
        prog.push(Reg(46), Insn::Arith(ArithOp::Add, Reg(46), Reg(16)));
        prog.push(Reg(45), Insn::LoadMem { addr: Reg(46), size: 8 });
        prog.push(Reg(15), Insn::Get(Reg(45)));

        // 0x486562
        prog.push(Reg(25), Insn::Get(Reg(16)));

        // 0x486565
        prog.push(Reg(46), Insn::Int { value: -1, size: 8 });
        prog.push(Reg(46), Insn::Arith(ArithOp::Add, Reg(46), Reg(15)));
        prog.push(Reg(45), Insn::LoadMem { addr: Reg(46), size: 1 });
        prog.push(Reg(45), Insn::Widen { reg: Reg(45), target_size: 4, sign: false });
        prog.push(Reg(47), Insn::Part { src: Reg(18), offset: 1, size: 7 });
        prog.push(Reg(18), Insn::Concat { lo: Reg(45), hi: Reg(47) });

        // 0x486569
        prog.push(Reg(46), Insn::Part { src: Reg(18), offset: 0, size: 4 });
        prog.push(Reg(46), Insn::Widen { reg: Reg(46), target_size: 8, sign: false });
        prog.push(Reg(16), Insn::Get(Reg(46)));

        // 0x48656b
        prog.push(Reg(46), Insn::Part { src: Reg(16), offset: 0, size: 4 });
        prog.push(Reg(47), Insn::Int { value: 7, size: 4 });
        prog.push(Reg(46), Insn::Arith(ArithOp::BitAnd, Reg(46), Reg(47)));
        prog.push(Reg(46), Insn::Widen { reg: Reg(46), target_size: 8, sign: false });
        prog.push(Reg(16), Insn::Get(Reg(46)));
        prog.push(Reg(10), Insn::False);
        prog.push(Reg(2), Insn::False);
        prog.push(Reg(6), Insn::SignOf(Reg(46)));
        prog.push(Reg(5), Insn::IsZero(Reg(46)));
        prog.push(Reg(48), Insn::Part { src: Reg(46), offset: 0, size: 1 });
        prog.push(Reg(3), Insn::Parity(Reg(48)));

        // 0x48656e
        prog.push(Reg(46), Insn::Part { src: Reg(16), offset: 0, size: 1 });
        prog.push(Reg(47), Insn::Int { value: 4, size: 1 });
        prog.push(Reg(46), Insn::Arith(ArithOp::Sub, Reg(46), Reg(47)));
        prog.push(Reg(10), Insn::OverflowOf(Reg(46)));
        prog.push(Reg(2), Insn::CarryOf(Reg(46)));
        prog.push(Reg(6), Insn::SignOf(Reg(46)));
        prog.push(Reg(5), Insn::IsZero(Reg(46)));
        prog.push(Reg(48), Insn::Part { src: Reg(46), offset: 0, size: 1 });
        prog.push(Reg(3), Insn::Parity(Reg(48)));

        // 0x486570
        prog.push(Reg(45), Insn::Not(Reg(6)));
        prog.push(Reg(46), Insn::Not(Reg(5)));
        prog.push(Reg(45), Insn::Bool(BoolOp::And, Reg(45), Reg(46)));
        prog.push(Reg(47), Insn::SetJumpCondition(Reg(45)));
        prog.push(Reg(47), Insn::Control(mil::Control::JmpExtIf(4203618)));

        // 0x486576
        prog.push(Reg(46), Insn::Part { src: Reg(16), offset: 0, size: 1 });
        prog.push(Reg(46), Insn::Widen { reg: Reg(46), target_size: 4, sign: false });
        prog.push(Reg(47), Insn::Part { src: Reg(16), offset: 1, size: 7 });
        prog.push(Reg(16), Insn::Concat { lo: Reg(46), hi: Reg(47) });

        // 0x486579
        prog.push(Reg(46), Insn::Int { value: 6851792, size: 8 });
        prog.push(Reg(47), Insn::ArithK(ArithOp::Mul, Reg(16), 8));
        prog.push(Reg(46), Insn::Arith(ArithOp::Add, Reg(46), Reg(47)));
        prog.push(Reg(46), Insn::SetJumpTarget(Reg(46)));
        prog.push(Reg(45), Insn::Control(mil::Control::JmpIndirect));

        // 0x486580
        prog.push(Reg(46), Insn::Int { value: -9, size: 8 });
        prog.push(Reg(46), Insn::Arith(ArithOp::Add, Reg(46), Reg(15)));
        prog.push(Reg(45), Insn::LoadMem { addr: Reg(46), size: 4 });
        prog.push(Reg(45), Insn::Widen { reg: Reg(45), target_size: 8, sign: false });
        prog.push(Reg(18), Insn::Get(Reg(45)));

        // 0x486583
        prog.push(Reg(46), Insn::Int { value: 0, size: 8 });
        prog.push(Reg(46), Insn::Arith(ArithOp::Add, Reg(46), Reg(17)));
        prog.push(Reg(45), Insn::LoadMem { addr: Reg(46), size: 8 });
        prog.push(Reg(19), Insn::Get(Reg(45)));

        // 0x486586
        prog.push(Reg(46), Insn::Int { value: 8, size: 8 });
        prog.push(Reg(46), Insn::Arith(ArithOp::Add, Reg(46), Reg(19)));
        prog.push(Reg(45), Insn::LoadMem { addr: Reg(46), size: 1 });
        prog.push(Reg(45), Insn::Widen { reg: Reg(45), target_size: 4, sign: false });
        prog.push(Reg(47), Insn::Part { src: Reg(16), offset: 1, size: 7 });
        prog.push(Reg(16), Insn::Concat { lo: Reg(45), hi: Reg(47) });

        // 0x48658a
        prog.push(Reg(46), Insn::Part { src: Reg(16), offset: 0, size: 1 });
        prog.push(Reg(47), Insn::Int { value: 9, size: 1 });
        prog.push(Reg(46), Insn::Arith(ArithOp::Sub, Reg(46), Reg(47)));
        prog.push(Reg(10), Insn::OverflowOf(Reg(46)));
        prog.push(Reg(2), Insn::CarryOf(Reg(46)));
        prog.push(Reg(6), Insn::SignOf(Reg(46)));
        prog.push(Reg(5), Insn::IsZero(Reg(46)));
        prog.push(Reg(48), Insn::Part { src: Reg(46), offset: 0, size: 1 });
        prog.push(Reg(3), Insn::Parity(Reg(48)));

        // 0x48658c
        prog.push(Reg(45), Insn::SetJumpCondition(Reg(5)));
        prog.push(Reg(45), Insn::Control(mil::Control::JmpExtIf(4744672)));

        // 0x48658e
        prog.push(Reg(46), Insn::Part { src: Reg(16), offset: 0, size: 1 });
        prog.push(Reg(47), Insn::Int { value: 11, size: 1 });
        prog.push(Reg(46), Insn::Arith(ArithOp::Sub, Reg(46), Reg(47)));
        prog.push(Reg(10), Insn::OverflowOf(Reg(46)));
        prog.push(Reg(2), Insn::CarryOf(Reg(46)));
        prog.push(Reg(6), Insn::SignOf(Reg(46)));
        prog.push(Reg(5), Insn::IsZero(Reg(46)));
        prog.push(Reg(48), Insn::Part { src: Reg(46), offset: 0, size: 1 });
        prog.push(Reg(3), Insn::Parity(Reg(48)));

        // 0x486590
        prog.push(Reg(45), Insn::Not(Reg(5)));
        prog.push(Reg(46), Insn::SetJumpCondition(Reg(45)));
        prog.push(Reg(46), Insn::Control(mil::Control::JmpExtIf(4744731)));

        // 0x486596
        prog.push(Reg(46), Insn::Int { value: 8, size: 8 });
        prog.push(Reg(12), Insn::Arith(ArithOp::Sub, Reg(12), Reg(46)));
        prog.push(Reg(12), Insn::Get(Reg(12)));
        prog.push(Reg(10), Insn::OverflowOf(Reg(12)));
        prog.push(Reg(2), Insn::CarryOf(Reg(12)));
        prog.push(Reg(6), Insn::SignOf(Reg(12)));
        prog.push(Reg(5), Insn::IsZero(Reg(12)));
        prog.push(Reg(47), Insn::Part { src: Reg(12), offset: 0, size: 1 });
        prog.push(Reg(3), Insn::Parity(Reg(47)));

        // 0x48659a
        prog.push(Reg(45), Insn::Int { value: 8, size: 8 });
        prog.push(Reg(45), Insn::Arith(ArithOp::Add, Reg(45), Reg(17)));
        prog.push(Reg(16), Insn::Get(Reg(45)));

        // 0x48659e
        prog.push(Reg(46), Insn::Int { value: 8, size: 8 });
        prog.push(Reg(46), Insn::Arith(ArithOp::Add, Reg(46), Reg(24)));
        prog.push(Reg(45), Insn::LoadMem { addr: Reg(46), size: 8 });
        prog.push(Reg(14), Insn::Get(Reg(45)));

        // 0x4865a3
        prog.push(Reg(46), Insn::Part { src: Reg(19), offset: 0, size: 4 });
        prog.push(Reg(48), Insn::Part { src: Reg(19), offset: 0, size: 4 });
        prog.push(Reg(46), Insn::Arith(ArithOp::BitXor, Reg(46), Reg(48)));
        prog.push(Reg(46), Insn::Widen { reg: Reg(46), target_size: 8, sign: false });
        prog.push(Reg(19), Insn::Get(Reg(46)));
        prog.push(Reg(10), Insn::False);
        prog.push(Reg(2), Insn::False);
        prog.push(Reg(6), Insn::SignOf(Reg(46)));
        prog.push(Reg(5), Insn::IsZero(Reg(46)));
        prog.push(Reg(49), Insn::Part { src: Reg(46), offset: 0, size: 1 });
        prog.push(Reg(3), Insn::Parity(Reg(49)));

        // 0x4865a5
        prog.push(Reg(12), Insn::ArithK(ArithOp::Add, Reg(12), -8));
        prog.push(Reg(46), Insn::StoreMem { addr: Reg(12), value: Reg(16) });

        // 0x4865a6
        prog.push(Reg(46), Insn::Int { value: 8, size: 8 });
        prog.push(Reg(46), Insn::Arith(ArithOp::Add, Reg(46), Reg(17)));
        prog.push(Reg(45), Insn::LoadMem { addr: Reg(46), size: 8 });
        prog.push(Reg(20), Insn::Get(Reg(45)));

        // 0x4865aa
        prog.push(Reg(45), Insn::Int { value: 2, size: 4 });
        prog.push(Reg(45), Insn::Widen { reg: Reg(45), target_size: 8, sign: false });
        prog.push(Reg(21), Insn::Get(Reg(45)));

        // 0x4865b0
        prog.push(Reg(45), Insn::Int { value: 5624160, size: 8 });
        prog.push(Reg(47), Insn::Void);
        prog.push(Reg(47), Insn::Get(Reg(14)));
        prog.push(Reg(48), Insn::Void);
        prog.push(Reg(48), Insn::Get(Reg(15)));
        prog.push(Reg(49), Insn::Void);
        prog.push(Reg(49), Insn::Get(Reg(19)));
        prog.push(Reg(50), Insn::Void);
        prog.push(Reg(50), Insn::Get(Reg(18)));
        prog.push(Reg(50), Insn::Part { src: Reg(50), offset: 0, size: 4 });
        prog.push(Reg(51), Insn::Void);
        prog.push(Reg(51), Insn::Get(Reg(20)));
        prog.push(Reg(52), Insn::Void);
        prog.push(Reg(52), Insn::Get(Reg(21)));
        prog.push(Reg(52), Insn::Part { src: Reg(52), offset: 0, size: 4 });
        prog.push(Reg(53), Insn::Void);
        prog.push(Reg(54), Insn::ArithK(ArithOp::Add, Reg(12), 8));
        prog.push(Reg(53), Insn::LoadMem { addr: Reg(54), size: 8 });
        prog.push(Reg(46), Insn::CArg { value: Reg(53), next_arg: None });
        prog.push(Reg(46), Insn::CArg { value: Reg(52), next_arg: Some(Reg(46)) });
        prog.push(Reg(46), Insn::CArg { value: Reg(51), next_arg: Some(Reg(46)) });
        prog.push(Reg(46), Insn::CArg { value: Reg(50), next_arg: Some(Reg(46)) });
        prog.push(Reg(46), Insn::CArg { value: Reg(49), next_arg: Some(Reg(46)) });
        prog.push(Reg(46), Insn::CArg { value: Reg(48), next_arg: Some(Reg(46)) });
        prog.push(Reg(46), Insn::CArg { value: Reg(47), next_arg: Some(Reg(46)) });
        prog.push(Reg(46), Insn::Call { callee: Reg(45), first_arg: Some(Reg(46)) });
        prog.push(Reg(2), Insn::UndefinedBool);
        prog.push(Reg(3), Insn::UndefinedBool);
        prog.push(Reg(4), Insn::UndefinedBool);
        prog.push(Reg(5), Insn::UndefinedBool);
        prog.push(Reg(6), Insn::UndefinedBool);
        prog.push(Reg(7), Insn::UndefinedBool);
        prog.push(Reg(8), Insn::UndefinedBool);
        prog.push(Reg(9), Insn::UndefinedBool);
        prog.push(Reg(10), Insn::UndefinedBool);
        prog.push(Reg(16), Insn::Part { src: Reg(46), offset: 0, size: 8 });

        // 0x4865b5
        prog.push(Reg(46), Insn::Int { value: 8, size: 8 });
        prog.push(Reg(46), Insn::Arith(ArithOp::Add, Reg(46), Reg(24)));
        prog.push(Reg(46), Insn::StoreMem { addr: Reg(46), value: Reg(16) });

        // 0x4865ba
        prog.push(Reg(45), Insn::LoadMem { addr: Reg(12), size: 8 });
        prog.push(Reg(16), Insn::Get(Reg(45)));
        prog.push(Reg(12), Insn::ArithK(ArithOp::Add, Reg(12), 8));

        // 0x4865bb
        prog.push(Reg(45), Insn::LoadMem { addr: Reg(12), size: 8 });
        prog.push(Reg(19), Insn::Get(Reg(45)));
        prog.push(Reg(12), Insn::ArithK(ArithOp::Add, Reg(12), 8));

        // 0x4865bc
        prog.push(Reg(45), Insn::Int { value: -24, size: 8 });
        prog.push(Reg(45), Insn::Arith(ArithOp::Add, Reg(45), Reg(11)));
        prog.push(Reg(12), Insn::Get(Reg(45)));

        // 0x4865c0
        prog.push(Reg(14), Insn::Get(Reg(25)));

        // 0x4865c3
        prog.push(Reg(45), Insn::LoadMem { addr: Reg(12), size: 8 });
        prog.push(Reg(17), Insn::Get(Reg(45)));
        prog.push(Reg(12), Insn::ArithK(ArithOp::Add, Reg(12), 8));

        // 0x4865c4
        prog.push(Reg(45), Insn::LoadMem { addr: Reg(12), size: 8 });
        prog.push(Reg(24), Insn::Get(Reg(45)));
        prog.push(Reg(12), Insn::ArithK(ArithOp::Add, Reg(12), 8));

        // 0x4865c6
        prog.push(Reg(45), Insn::LoadMem { addr: Reg(12), size: 8 });
        prog.push(Reg(25), Insn::Get(Reg(45)));
        prog.push(Reg(12), Insn::ArithK(ArithOp::Add, Reg(12), 8));

        // 0x4865c8
        prog.push(Reg(45), Insn::LoadMem { addr: Reg(12), size: 8 });
        prog.push(Reg(11), Insn::Get(Reg(45)));
        prog.push(Reg(12), Insn::ArithK(ArithOp::Add, Reg(12), 8));

        // 0x4865c9
        prog.push(Reg(45), Insn::Control(mil::Control::JmpExt(4577184)));

        // 0x4865d0
        prog.push(Reg(46), Insn::Int { value: 0, size: 8 });
        prog.push(Reg(46), Insn::Arith(ArithOp::Add, Reg(46), Reg(17)));
        prog.push(Reg(45), Insn::LoadMem { addr: Reg(46), size: 8 });
        prog.push(Reg(19), Insn::Get(Reg(45)));

        // 0x4865d3
        prog.push(Reg(46), Insn::Int { value: -17, size: 8 });
        prog.push(Reg(46), Insn::Arith(ArithOp::Add, Reg(46), Reg(15)));
        prog.push(Reg(45), Insn::LoadMem { addr: Reg(46), size: 8 });
        prog.push(Reg(18), Insn::Get(Reg(45)));

        // 0x4865d7
        prog.push(Reg(46), Insn::Int { value: 8, size: 8 });
        prog.push(Reg(46), Insn::Arith(ArithOp::Add, Reg(46), Reg(19)));
        prog.push(Reg(45), Insn::LoadMem { addr: Reg(46), size: 1 });
        prog.push(Reg(45), Insn::Widen { reg: Reg(45), target_size: 4, sign: false });
        prog.push(Reg(47), Insn::Part { src: Reg(16), offset: 1, size: 7 });
        prog.push(Reg(16), Insn::Concat { lo: Reg(45), hi: Reg(47) });

        // 0x4865db
        prog.push(Reg(46), Insn::Part { src: Reg(16), offset: 0, size: 1 });
        prog.push(Reg(47), Insn::Int { value: 9, size: 1 });
        prog.push(Reg(46), Insn::Arith(ArithOp::Sub, Reg(46), Reg(47)));
        prog.push(Reg(10), Insn::OverflowOf(Reg(46)));
        prog.push(Reg(2), Insn::CarryOf(Reg(46)));
        prog.push(Reg(6), Insn::SignOf(Reg(46)));
        prog.push(Reg(5), Insn::IsZero(Reg(46)));
        prog.push(Reg(48), Insn::Part { src: Reg(46), offset: 0, size: 1 });
        prog.push(Reg(3), Insn::Parity(Reg(48)));

        // 0x4865dd
        prog.push(Reg(45), Insn::Not(Reg(5)));
        prog.push(Reg(46), Insn::SetJumpCondition(Reg(45)));
        prog.push(Reg(46), Insn::Control(mil::Control::JmpExtIf(4744590)));

        // 0x4865e0
        prog.push(Reg(46), Insn::Int { value: 24, size: 8 });
        prog.push(Reg(46), Insn::Arith(ArithOp::Add, Reg(46), Reg(19)));
        prog.push(Reg(45), Insn::LoadMem { addr: Reg(46), size: 8 });
        prog.push(Reg(14), Insn::Get(Reg(45)));

        // 0x4865e4
        prog.push(Reg(46), Insn::Int { value: 16, size: 8 });
        prog.push(Reg(17), Insn::Arith(ArithOp::Add, Reg(17), Reg(46)));
        prog.push(Reg(17), Insn::Get(Reg(17)));
        prog.push(Reg(10), Insn::OverflowOf(Reg(17)));
        prog.push(Reg(2), Insn::CarryOf(Reg(17)));
        prog.push(Reg(6), Insn::SignOf(Reg(17)));
        prog.push(Reg(5), Insn::IsZero(Reg(17)));
        prog.push(Reg(47), Insn::Part { src: Reg(17), offset: 0, size: 1 });
        prog.push(Reg(3), Insn::Parity(Reg(47)));

        // 0x4865e8
        prog.push(Reg(19), Insn::Get(Reg(15)));

        // 0x4865eb
        prog.push(Reg(15), Insn::Get(Reg(17)));

        // 0x4865ee
        prog.push(Reg(45), Insn::Int { value: 4299888, size: 8 });
        prog.push(Reg(47), Insn::Void);
        prog.push(Reg(47), Insn::Get(Reg(14)));
        prog.push(Reg(48), Insn::Void);
        prog.push(Reg(48), Insn::Get(Reg(15)));
        prog.push(Reg(49), Insn::Void);
        prog.push(Reg(49), Insn::Get(Reg(19)));
        prog.push(Reg(50), Insn::Void);
        prog.push(Reg(50), Insn::Get(Reg(18)));
        prog.push(Reg(46), Insn::CArg { value: Reg(50), next_arg: None });
        prog.push(Reg(46), Insn::CArg { value: Reg(49), next_arg: Some(Reg(46)) });
        prog.push(Reg(46), Insn::CArg { value: Reg(48), next_arg: Some(Reg(46)) });
        prog.push(Reg(46), Insn::CArg { value: Reg(47), next_arg: Some(Reg(46)) });
        prog.push(Reg(46), Insn::Call { callee: Reg(45), first_arg: Some(Reg(46)) });
        prog.push(Reg(2), Insn::UndefinedBool);
        prog.push(Reg(3), Insn::UndefinedBool);
        prog.push(Reg(4), Insn::UndefinedBool);
        prog.push(Reg(5), Insn::UndefinedBool);
        prog.push(Reg(6), Insn::UndefinedBool);
        prog.push(Reg(7), Insn::UndefinedBool);
        prog.push(Reg(8), Insn::UndefinedBool);
        prog.push(Reg(9), Insn::UndefinedBool);
        prog.push(Reg(10), Insn::UndefinedBool);

        // 0x4865f3
        prog.push(Reg(45), Insn::Control(mil::Control::JmpExt(4744636)));

        // 0x4865f8
        prog.push(Reg(46), Insn::Int { value: -3, size: 8 });
        prog.push(Reg(46), Insn::Arith(ArithOp::Add, Reg(46), Reg(15)));
        prog.push(Reg(45), Insn::LoadMem { addr: Reg(46), size: 1 });
        prog.push(Reg(45), Insn::Widen { reg: Reg(45), target_size: 4, sign: false });
        prog.push(Reg(47), Insn::Part { src: Reg(18), offset: 1, size: 7 });
        prog.push(Reg(18), Insn::Concat { lo: Reg(45), hi: Reg(47) });

        // 0x4865fc
        prog.push(Reg(45), Insn::Control(mil::Control::JmpExt(4744579)));

        // 0x486600
        prog.push(Reg(46), Insn::Int { value: -5, size: 8 });
        prog.push(Reg(46), Insn::Arith(ArithOp::Add, Reg(46), Reg(15)));
        prog.push(Reg(45), Insn::LoadMem { addr: Reg(46), size: 2 });
        prog.push(Reg(45), Insn::Widen { reg: Reg(45), target_size: 4, sign: false });
        prog.push(Reg(47), Insn::Part { src: Reg(18), offset: 2, size: 6 });
        prog.push(Reg(18), Insn::Concat { lo: Reg(45), hi: Reg(47) });

        // 0x486604
        prog.push(Reg(45), Insn::Control(mil::Control::JmpExt(4744579)));

        // 0x486610
        prog.push(Reg(46), Insn::Part { src: Reg(18), offset: 0, size: 1 });
        prog.push(Reg(47), Insn::Int { value: 3, size: 1 });
        prog.push(Reg(46), Insn::Arith(ArithOp::Shr, Reg(46), Reg(47)));
        prog.push(Reg(48), Insn::Part { src: Reg(18), offset: 1, size: 7 });
        prog.push(Reg(18), Insn::Concat { lo: Reg(46), hi: Reg(48) });
        prog.push(Reg(6), Insn::SignOf(Reg(46)));
        prog.push(Reg(5), Insn::IsZero(Reg(46)));
        prog.push(Reg(49), Insn::Part { src: Reg(46), offset: 0, size: 1 });
        prog.push(Reg(3), Insn::Parity(Reg(49)));

        // 0x486613
        prog.push(Reg(46), Insn::Part { src: Reg(18), offset: 0, size: 1 });
        prog.push(Reg(46), Insn::Widen { reg: Reg(46), target_size: 4, sign: false });
        prog.push(Reg(47), Insn::Part { src: Reg(18), offset: 1, size: 7 });
        prog.push(Reg(18), Insn::Concat { lo: Reg(46), hi: Reg(47) });

        // 0x486616
        prog.push(Reg(45), Insn::Control(mil::Control::JmpExt(4744579)));

        // 0x48661b
        prog.push(Reg(45), Insn::Int { value: 6979771, size: 4 });
        prog.push(Reg(45), Insn::Widen { reg: Reg(45), target_size: 8, sign: false });
        prog.push(Reg(19), Insn::Get(Reg(45)));

        // 0x486620
        prog.push(Reg(45), Insn::Int { value: 352, size: 4 });
        prog.push(Reg(45), Insn::Widen { reg: Reg(45), target_size: 8, sign: false });
        prog.push(Reg(15), Insn::Get(Reg(45)));

        // 0x486625
        prog.push(Reg(45), Insn::Int { value: 6982459, size: 4 });
        prog.push(Reg(45), Insn::Widen { reg: Reg(45), target_size: 8, sign: false });
        prog.push(Reg(14), Insn::Get(Reg(45)));

        // 0x48662a
        prog.push(Reg(46), Insn::Part { src: Reg(16), offset: 0, size: 4 });
        prog.push(Reg(48), Insn::Part { src: Reg(16), offset: 0, size: 4 });
        prog.push(Reg(46), Insn::Arith(ArithOp::BitXor, Reg(46), Reg(48)));
        prog.push(Reg(46), Insn::Widen { reg: Reg(46), target_size: 8, sign: false });
        prog.push(Reg(16), Insn::Get(Reg(46)));
        prog.push(Reg(10), Insn::False);
        prog.push(Reg(2), Insn::False);
        prog.push(Reg(6), Insn::SignOf(Reg(46)));
        prog.push(Reg(5), Insn::IsZero(Reg(46)));
        prog.push(Reg(49), Insn::Part { src: Reg(46), offset: 0, size: 1 });
        prog.push(Reg(3), Insn::Parity(Reg(49)));

        // 0x48662c
        prog.push(Reg(45), Insn::Int { value: 4990320, size: 8 });
        prog.push(Reg(47), Insn::Void);
        prog.push(Reg(47), Insn::Get(Reg(14)));
        prog.push(Reg(48), Insn::Void);
        prog.push(Reg(48), Insn::Get(Reg(15)));
        prog.push(Reg(48), Insn::Part { src: Reg(48), offset: 0, size: 4 });
        prog.push(Reg(49), Insn::Void);
        prog.push(Reg(49), Insn::Get(Reg(19)));
        prog.push(Reg(46), Insn::CArg { value: Reg(49), next_arg: None });
        prog.push(Reg(46), Insn::CArg { value: Reg(48), next_arg: Some(Reg(46)) });
        prog.push(Reg(46), Insn::CArg { value: Reg(47), next_arg: Some(Reg(46)) });
        prog.push(Reg(46), Insn::Call { callee: Reg(45), first_arg: Some(Reg(46)) });
        prog.push(Reg(2), Insn::UndefinedBool);
        prog.push(Reg(3), Insn::UndefinedBool);
        prog.push(Reg(4), Insn::UndefinedBool);
        prog.push(Reg(5), Insn::UndefinedBool);
        prog.push(Reg(6), Insn::UndefinedBool);
        prog.push(Reg(7), Insn::UndefinedBool);
        prog.push(Reg(8), Insn::UndefinedBool);
        prog.push(Reg(9), Insn::UndefinedBool);
        prog.push(Reg(10), Insn::UndefinedBool);

        crate::ssa::Program::from_mil(prog)
    }
}
