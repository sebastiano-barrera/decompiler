use std::{borrow::Cow, collections::HashSet};

use crate::{
    cfg,
    mil::{ArithOp, BoolOp, CmpOp, Insn, Reg},
    pp::PP,
    ssa,
};

#[derive(Debug)]
pub struct Ast<'a> {
    ssa: &'a ssa::Program,
    let_printed: HashSet<Reg>,
    rdr_count: ssa::RegMap<usize>,
}

impl<'a> Ast<'a> {
    pub fn new(ssa: &'a ssa::Program) -> Self {
        let rdr_count = ssa::count_readers(&ssa);
        Ast {
            ssa,
            let_printed: HashSet::new(),
            rdr_count,
        }
    }

    pub fn pretty_print<W: PP + ?Sized>(&mut self, pp: &mut W) -> std::io::Result<()> {
        let cfg = self.ssa.cfg();
        let entry_bid = cfg.entry_block_id();
        self.pp_block_labeled(pp, entry_bid)
    }

    fn pp_block_labeled<W: PP + ?Sized>(
        &mut self,
        pp: &mut W,
        bid: cfg::BlockID,
    ) -> std::io::Result<()> {
        write!(pp, "T{}: {{\n  ", bid.as_number())?;
        pp.open_box();

        self.pp_block_inner(pp, bid)?;

        pp.close_box();
        writeln!(pp, "\n}}")
    }

    fn pp_block_inner<W: PP + ?Sized>(
        &mut self,
        pp: &mut W,
        bid: cfg::BlockID,
    ) -> std::io::Result<()> {
        let cfg = self.ssa.cfg();
        let block_cont = cfg.block_cont(bid);

        let block_effects = self.ssa.block_effects(bid);
        for (ndx, reg) in block_effects.into_iter().enumerate() {
            if ndx > 0 {
                writeln!(pp)?;
            }
            self.pp_labeled_inputs(pp, reg)?;
            self.pp_def(pp, reg)?;
        }

        match block_cont {
            cfg::BlockCont::End => {
                // all done!
            }
            cfg::BlockCont::Jmp((pred_ndx, tgt)) => {
                writeln!(pp)?;
                self.pp_continuation(pp, bid, tgt, pred_ndx as u16)?;
            }
            cfg::BlockCont::Alt {
                straight: (neg_pred_ndx, neg_bid),
                side: (pos_pred_ndx, pos_bid),
            } => {
                let last_insn_reg = self.ssa.block_effects(bid).last().unwrap();
                let last_insn = self.ssa.get(last_insn_reg).unwrap().insn.get();
                if !matches!(last_insn, Insn::JmpIf { .. }) {
                    panic!("block with BlockCont::Alt continuation must end with a JmpIf");
                }

                write!(pp, " {{\n  ")?;

                pp.open_box();
                self.pp_continuation(pp, bid, pos_bid, pos_pred_ndx as u16)?;
                pp.close_box();

                write!(pp, "\n}}\n")?;

                self.pp_continuation(pp, bid, neg_bid, neg_pred_ndx as u16)?;
            }
        }

        for &child in cfg.dom_tree().children_of(bid) {
            if cfg.block_preds(child).len() > 1 {
                writeln!(pp)?;
                self.pp_block_labeled(pp, child)?;
            }
        }

        Ok(())
    }

    fn pp_continuation<W: PP + ?Sized>(
        &mut self,
        pp: &mut W,
        src_bid: cfg::BlockID,
        tgt_bid: cfg::BlockID,
        _pred_ndx: u16,
    ) -> std::io::Result<()> {
        let cfg = self.ssa.cfg();
        let looping_back = cfg
            .dom_tree()
            .imm_doms(src_bid)
            .find(|i| *i == tgt_bid)
            .is_some();
        let keyword = if looping_back { "loop" } else { "goto" };

        let pred_count = cfg.block_preds(tgt_bid).len();
        if pred_count == 1 {
            self.pp_block_inner(pp, tgt_bid)
        } else {
            write!(pp, "{keyword} T{}", tgt_bid.as_number())
        }
    }

    fn is_named(&self, reg: Reg) -> bool {
        self.rdr_count[reg] > 1
    }

    /// Results in a series of `let x = ...` expressions being printed.
    ///
    /// This function ensures that all named inputs to the given instruction
    /// have been printed exactly once. Values that require `let` expressions
    /// that were already printed in the past are not printed again.
    ///
    /// This function does NOT print reg or the instruction defining reg.
    fn pp_labeled_inputs<W: PP + ?Sized>(&mut self, pp: &mut W, reg: Reg) -> std::io::Result<()> {
        let insn = self.ssa.get(reg).unwrap().insn.get();
        for input in insn.input_regs_iter() {
            self.pp_labeled_inputs(pp, input)?;
            if self.is_named(input) && !self.let_printed.contains(&input) {
                self.let_printed.insert(input);
                write!(pp, "let r{} = ", input.reg_index())?;
                self.pp_def(pp, input)?;
                writeln!(pp, ";")?;
            }
        }
        Ok(())
    }

    /// Prints the instruction defining the given register.
    ///
    /// Inline (unnamed) inputs to the instruction will be printed inline. For
    /// named inputs, register references (`r##`) will be used (the function
    /// panics if the corresponding `let` has not been printed yet).
    fn pp_def<W: PP + ?Sized>(&mut self, pp: &mut W, reg: Reg) -> std::io::Result<()> {
        // NOTE this function is called in both cases:
        //  - printing the "toplevel" definition of a named or effectful instruction;
        //  - printing an instruction definition inline as part of the 1 dependent instruction
        // (For this reason, we can't pp_labeled_inputs here)
        let iv = self.ssa.get(reg).unwrap();
        let insn = iv.insn.get();

        let op_s: Cow<str> = match insn {
            Insn::True => "True".into(),
            Insn::False => "False".into(),
            Insn::Const { value, size } => {
                return write!(pp, "{}i{} /* 0x{:x} */", value, size * 8, value)
            }
            Insn::Get(x) => return self.pp_def(pp, x),
            Insn::StructGet8 {
                struct_value: _,
                offset,
            } => format!("StructGet8[{offset}]").into(),
            Insn::Part { src, offset, size } => {
                self.pp_ref(pp, src)?;
                write!(pp, "[{} .. {}]", offset, offset + size)?;
                return Ok(());
            }
            Insn::Concat { lo, hi } => {
                self.pp_ref(pp, hi)?;
                write!(pp, "⧺")?;
                self.pp_ref(pp, lo)?;
                return Ok(());
            }
            Insn::Widen {
                reg: _,
                target_size,
            } => format!("WidenTo{}", target_size).into(),
            Insn::Arith(arith_op, a, b) => {
                self.pp_ref(pp, a)?;
                write!(pp, " {} ", arith_op_str(arith_op))?;
                self.pp_ref(pp, b)?;
                return Ok(());
            }
            Insn::ArithK(arith_op, a, k) => {
                self.pp_ref(pp, a)?;
                write!(pp, " {} {}", arith_op_str(arith_op), k)?;
                return Ok(());
            }
            Insn::Cmp(cmp_op, _, _) => match cmp_op {
                CmpOp::EQ => "EQ",
                CmpOp::LT => "LT",
            }
            .into(),
            Insn::Bool(bool_op, _, _) => match bool_op {
                BoolOp::Or => "OR",
                BoolOp::And => "AND",
            }
            .into(),
            Insn::Not(_) => "!".into(),
            Insn::Call(callee) => {
                self.pp_ref(pp, callee)?;
                write!(pp, "(")?;
                pp.open_box();
                for (ndx, arg) in self.ssa.get_call_args(reg).enumerate() {
                    if ndx > 0 {
                        writeln!(pp, ",")?;
                    }
                    self.pp_ref(pp, arg)?;
                }
                pp.close_box();
                write!(pp, ")")?;
                return Ok(());
            }
            Insn::CArg(_) => panic!("CArg not handled via this path!"),
            Insn::Ret(_) => "Ret".into(),
            Insn::TODO(msg) => return write!(pp, "TODO /* {} */", msg),
            Insn::LoadMem { reg, size } => return self.pp_load_mem(pp, reg, size),
            Insn::StoreMem(addr, value) => {
                write!(pp, "[")?;
                pp.open_box();
                self.pp_ref(pp, addr)?;
                write!(pp, "] = ")?;
                pp.open_box();
                self.pp_ref(pp, value)?;
                pp.close_box();
                pp.close_box();
                return Ok(());
            }
            Insn::OverflowOf(_) => "OverflowOf".into(),
            Insn::CarryOf(_) => "CarryOf".into(),
            Insn::SignOf(_) => "SignOf".into(),
            Insn::IsZero(_) => "IsZero".into(),
            Insn::Parity(_) => "Parity".into(),
            Insn::Undefined => "Undefined".into(),
            Insn::Ancestral(anc_name) => return write!(pp, "pre:{}", anc_name.name()),

            Insn::Phi | Insn::PhiBool => return write!(pp, "r{}", iv.dest.get().reg_index()),
            Insn::Jmp(_) => {
                // hidden, handled by pp_block
                return Ok(());
            }
            Insn::JmpIf { .. } => "if".into(),

            Insn::JmpInd(_) => "JmpInd".into(),
            Insn::JmpExt(addr) => return write!(pp, "JmpExt(0x{:x})", addr),
            Insn::JmpExtIf { cond, addr } => {
                write!(pp, "if ")?;
                self.pp_ref(pp, cond)?;
                write!(pp, "{{\n  goto 0x{:0x}\n}}", addr)?;
                return Ok(());
            }
            Insn::Upsilon { value, phi_ref } => {
                write!(pp, "r{} := ", phi_ref.reg_index())?;
                pp.open_box();
                self.pp_def(pp, value)?;
                pp.close_box();
                write!(pp, ";")?;
                return Ok(());
            }
        };

        write!(pp, "{}(", op_s)?;

        for (arg_ndx, arg) in insn.input_regs_iter().enumerate() {
            if arg_ndx > 0 {
                write!(pp, ", ")?;
            }

            self.pp_ref(pp, arg)?;
        }

        write!(pp, ")")?;
        Ok(())
    }

    fn pp_load_mem<W: PP + ?Sized>(
        &mut self,
        pp: &mut W,
        addr: Reg,
        sz: i32,
    ) -> std::io::Result<()> {
        write!(pp, "[")?;
        pp.open_box();
        self.pp_ref(pp, addr)?;
        pp.close_box();
        write!(pp, "]:{}", sz)
    }

    fn pp_ref<W: PP + ?Sized>(&mut self, pp: &mut W, arg: Reg) -> std::io::Result<()> {
        if self.is_named(arg) {
            assert!(
                self.let_printed.contains(&arg),
                "arg needs let but not yet printed: {:?}",
                arg
            );
            write!(pp, "r{}", arg.reg_index())
        } else {
            self.pp_def(pp, arg)
        }
    }
}

fn arith_op_str(arith_op: ArithOp) -> &'static str {
    match arith_op {
        ArithOp::Add => "+",
        ArithOp::Sub => "-",
        ArithOp::Mul => "*",
        ArithOp::Shl => "/",
        ArithOp::BitXor => "^",
        ArithOp::BitAnd => "&",
        ArithOp::BitOr => "|",
    }
}
