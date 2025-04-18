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
    is_named: ssa::RegMap<bool>,

    block_printed: cfg::BlockMap<bool>,
}

impl<'a> Ast<'a> {
    pub fn new(ssa: &'a ssa::Program) -> Self {
        let rdr_count = ssa::count_readers(&ssa);

        let is_named = rdr_count.map(|reg, count| {
            let insn = ssa.get(reg).unwrap().insn.get();
            // ancestral are as good as r# refs, so never 'name' them / always print inline
            matches!(insn, Insn::Phi)
                || (*count > 1
                    && !matches!(insn, Insn::Ancestral(_))
                    && !matches!(insn, Insn::Const { .. }))
        });

        Ast {
            ssa,
            let_printed: HashSet::new(),
            is_named,
            block_printed: cfg::BlockMap::new(ssa.cfg(), false),
        }
    }

    fn is_named(&self, reg: Reg) -> bool {
        self.is_named[reg]
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
        write!(pp, "\nT{}: {{\n  ", bid.as_number())?;
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
        assert!(!self.block_printed[bid]);
        self.block_printed[bid] = true;

        let cfg = self.ssa.cfg();
        let block_cont = cfg.block_cont(bid);

        let block_effects = self.ssa.block_effects(bid);
        for (ndx, reg) in block_effects.into_iter().enumerate() {
            if ndx > 0 {
                writeln!(pp)?;
            }
            self.pp_labeled_inputs(pp, *reg)?;
            self.pp_def(pp, *reg, 0)?;
        }

        match block_cont {
            cfg::BlockCont::End => {
                write!(pp, "end");
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
                write!(pp, " then {{\n  ")?;

                pp.open_box();
                self.pp_continuation(pp, bid, pos_bid, pos_pred_ndx as u16)?;
                pp.close_box();

                write!(pp, "\n}} else:\n")?;

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

        if cfg.block_preds(tgt_bid).len() == 1 {
            self.pp_block_inner(pp, tgt_bid)
        } else {
            let looping_back = cfg
                .dom_tree()
                .imm_doms(src_bid)
                .find(|i| *i == tgt_bid)
                .is_some();
            let keyword = if looping_back { "loop" } else { "goto" };
            write!(pp, "{keyword} T{}", tgt_bid.as_number())
        }
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

                if let Insn::Phi = self.ssa.get(input).unwrap().insn.get() {
                    writeln!(pp, "let mut r{};", input.reg_index())?;
                } else {
                    write!(pp, "let r{} = ", input.reg_index())?;
                    self.pp_def(pp, input, 0)?;
                    writeln!(pp, ";")?;
                }
            }
        }
        Ok(())
    }

    /// Prints the instruction defining the given register.
    ///
    /// Inline (unnamed) inputs to the instruction will be printed inline. For
    /// named inputs, register references (`r##`) will be used (the function
    /// panics if the corresponding `let` has not been printed yet).
    fn pp_def<W: PP + ?Sized>(
        &mut self,
        pp: &mut W,
        reg: Reg,
        parent_prec: u8,
    ) -> std::io::Result<()> {
        // NOTE this function is called in both cases:
        //  - printing the "toplevel" definition of a named or effectful instruction;
        //  - printing an instruction definition inline as part of the 1 dependent instruction
        // (For this reason, we can't pp_labeled_inputs here)
        let iv = self.ssa.get(reg).unwrap();
        let insn = iv.insn.get();

        if let Insn::Get(x) = insn {
            return self.pp_def(pp, x, parent_prec);
        }
        if let Insn::Jmp(_) = insn {
            // hidden, handled by pp_block
            return Ok(());
        }

        let self_prec = precedence(&insn);

        if self_prec < parent_prec {
            write!(pp, "(")?;
        }

        match insn {
            Insn::Void => write!(pp, "void")?,
            Insn::True => self.pp_def_default(pp, "True".into(), insn.input_regs(), self_prec)?,
            Insn::False => self.pp_def_default(pp, "False".into(), insn.input_regs(), self_prec)?,
            Insn::Const { value, size } => {
                write!(pp, "{}_i{}", value, size)?;
            }
            Insn::Get(_) | Insn::Jmp(_) => unreachable!(),

            Insn::StructGetMember {
                struct_value,
                name,
                size: _,
            } => {
                self.pp_ref(pp, struct_value, self_prec)?;
                write!(pp, ".{}", name)?;
            }
            Insn::Part { src, offset, size } => {
                self.pp_ref(pp, src, self_prec)?;
                // syntax is [end..start] because it's more intuitive with concatenation, e.g.:
                //   r13[8 .. 4] ++ r12[4 .. 0]
                write!(pp, "[{} .. {}]", offset + size, offset,)?;
            }
            Insn::Concat { lo, hi } => {
                self.pp_ref(pp, hi, self_prec)?;
                write!(pp, " ++ ")?;
                self.pp_ref(pp, lo, self_prec)?;
            }
            Insn::Widen {
                reg: arg,
                target_size,
                sign,
            } => {
                self.pp_ref(pp, arg, self_prec)?;
                let ch = if sign { 'i' } else { 'u' };
                write!(pp, "_{}{}", ch, 8 * target_size)?;
            }
            Insn::Arith(arith_op, a, b) => {
                self.pp_ref(pp, a, self_prec)?;
                write!(pp, " {} ", arith_op_str(arith_op))?;
                self.pp_ref(pp, b, self_prec)?;
            }
            Insn::ArithK(arith_op, a, k) => {
                // trick: it's convenient to prefer ArithOp::Add to ArithOp::Sub
                // in SSA, even with a negative constant (because + is
                // associative, which makes constant folding easier), but it's
                // unsightly to see a bunch of `[r11 + -42]:8` in AST. so we
                // make a replacement just here, just for this purpose.
                self.pp_ref(pp, a, self_prec)?;
                if arith_op == ArithOp::Add && k < 0 {
                    write!(pp, " - {}", -k)?;
                } else {
                    write!(pp, " {} {}", arith_op_str(arith_op), k)?;
                }
            }
            Insn::Cmp(cmp_op, _, _) => {
                let op_s = match cmp_op {
                    CmpOp::EQ => "EQ",
                    CmpOp::LT => "LT",
                };
                self.pp_def_default(pp, op_s.into(), insn.input_regs(), self_prec)?;
            }
            Insn::Bool(bool_op, _, _) => {
                let op_s = match bool_op {
                    BoolOp::Or => "OR",
                    BoolOp::And => "AND",
                };
                self.pp_def_default(pp, op_s.into(), insn.input_regs(), self_prec)?;
            }
            Insn::Not(_) => {
                self.pp_def_default(pp, "!".into(), insn.input_regs(), self_prec)?;
            }
            Insn::Call { callee, first_arg } => {
                self.pp_ref(pp, callee, self_prec)?;
                write!(pp, "(")?;
                pp.open_box();
                for (ndx, arg) in self.ssa.get_call_args(first_arg).enumerate() {
                    if ndx > 0 {
                        writeln!(pp, ",")?;
                    }
                    self.pp_ref(pp, arg, self_prec)?;
                }
                pp.close_box();
                write!(pp, ")")?;
            }
            Insn::Ret(_) => {
                self.pp_def_default(pp, "Ret".into(), insn.input_regs(), self_prec)?;
            }
            Insn::NotYetImplemented(msg) => {
                write!(pp, "TODO /* {} */", msg)?;
            }
            Insn::LoadMem { mem: _, addr, size } => {
                let sz = size.try_into().unwrap();
                self.pp_load_mem(pp, addr, sz)?;
            }
            Insn::StoreMem {
                mem: _,
                addr,
                value,
            } => {
                write!(pp, "[")?;
                pp.open_box();
                self.pp_ref(pp, addr, self_prec)?;
                write!(pp, "] = ")?;
                pp.open_box();
                self.pp_ref(pp, value, self_prec)?;
                pp.close_box();
                pp.close_box();
            }
            Insn::OverflowOf(_) => {
                self.pp_def_default(pp, "OverflowOf".into(), insn.input_regs(), self_prec)?;
            }
            Insn::CarryOf(_) => {
                self.pp_def_default(pp, "CarryOf".into(), insn.input_regs(), self_prec)?;
            }
            Insn::SignOf(_) => {
                self.pp_def_default(pp, "SignOf".into(), insn.input_regs(), self_prec)?;
            }
            Insn::IsZero(_) => {
                self.pp_def_default(pp, "IsZero".into(), insn.input_regs(), self_prec)?;
            }
            Insn::Parity(_) => {
                self.pp_def_default(pp, "Parity".into(), insn.input_regs(), self_prec)?;
            }
            Insn::Undefined => {
                self.pp_def_default(pp, "Undefined".into(), insn.input_regs(), self_prec)?;
            }
            Insn::Ancestral(anc_name) => {
                write!(pp, "pre:{}", anc_name.name())?;
            }

            Insn::Phi => {
                self.pp_def_default(pp, "phi".into(), insn.input_regs(), self_prec)?;
            }
            Insn::JmpIf { .. } => {
                self.pp_def_default(pp, "if".into(), insn.input_regs(), self_prec)?;
            }
            Insn::JmpInd(_) => {
                self.pp_def_default(pp, "JmpInd".into(), insn.input_regs(), self_prec)?;
            }
            Insn::JmpExt(addr) => {
                write!(pp, "JmpExt(0x{:x})", addr)?;
            }
            Insn::JmpExtIf { cond, addr } => {
                write!(pp, "if ")?;
                self.pp_ref(pp, cond, self_prec)?;
                write!(pp, "{{\n  goto 0x{:0x}\n}}", addr)?;
            }
            Insn::Upsilon { value, phi_ref } => {
                write!(pp, "r{} := ", phi_ref.reg_index())?;
                pp.open_box();
                self.pp_def(pp, value, 0)?;
                pp.close_box();
                write!(pp, ";")?;
            }
            Insn::CArg { .. } => {
                unreachable!("CArg should be handled via the Call it belongs to!")
            }
        };

        if self_prec < parent_prec {
            write!(pp, ")")?;
        }
        Ok(())
    }

    fn pp_def_default<W: PP + ?Sized>(
        &mut self,
        pp: &mut W,
        op_s: Cow<'_, str>,
        args: [Option<&Reg>; 3],
        self_prec: u8,
    ) -> Result<(), std::io::Error> {
        write!(pp, "{} (", op_s)?;
        for (arg_ndx, arg) in args.into_iter().flatten().enumerate() {
            if arg_ndx > 0 {
                write!(pp, ", ")?;
            }
            self.pp_ref(pp, *arg, self_prec)?;
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

        // we're writing parens in this function, so no need to print more
        // parens
        let parent_prec = 0;
        self.pp_ref(pp, addr, parent_prec)?;

        pp.close_box();
        write!(pp, "]:{}", sz)
    }

    fn pp_ref<W: PP + ?Sized>(
        &mut self,
        pp: &mut W,
        arg: Reg,
        parent_prec: u8,
    ) -> std::io::Result<()> {
        if self.is_named(arg) {
            assert!(
                self.let_printed.contains(&arg),
                "arg needs let but not yet printed: {:?}",
                arg
            );
            write!(pp, "r{}", arg.reg_index())
        } else {
            self.pp_def(pp, arg, parent_prec)
        }
    }
}

fn arith_op_str(arith_op: ArithOp) -> &'static str {
    match arith_op {
        ArithOp::Add => "+",
        ArithOp::Sub => "-",
        ArithOp::Mul => "*",
        ArithOp::Shl => "<<",
        ArithOp::Shr => ">>",
        ArithOp::BitXor => "^",
        ArithOp::BitAnd => "&",
        ArithOp::BitOr => "|",
    }
}

fn precedence(insn: &Insn) -> u8 {
    // higher value == higher precedence == evaluated first unless parenthesized
    match insn {
        Insn::Get(_) => panic!("Get must be resolved prior to calling precedence"),

        Insn::Void
        | Insn::True
        | Insn::False
        | Insn::Const { .. }
        | Insn::Undefined
        | Insn::Ancestral(_)
        | Insn::Phi
        | Insn::StructGetMember { .. }
        | Insn::LoadMem { .. } => 255,

        Insn::Part { .. } => 254,
        Insn::Concat { .. } => 253,

        Insn::Call { .. } => 251,
        Insn::Not(_) => 250,
        Insn::Widen { .. } => 249,

        Insn::Arith(_, _, _) => 200,
        Insn::ArithK(_, _, _) => 200,
        Insn::OverflowOf(_) => 200,
        Insn::CarryOf(_) => 200,
        Insn::SignOf(_) => 200,
        Insn::IsZero(_) => 200,
        Insn::Parity(_) => 200,

        Insn::Bool(_, _, _) => 199,
        Insn::Cmp(_, _, _) => 197,

        // effectful instructions are basically *always* done last due to their
        // position in the printed syntax
        Insn::Ret(_)
        | Insn::JmpInd(_)
        | Insn::Jmp(_)
        | Insn::JmpIf { .. }
        | Insn::JmpExt(_)
        | Insn::JmpExtIf { .. }
        | Insn::NotYetImplemented(_)
        | Insn::Upsilon { .. }
        | Insn::StoreMem { .. } => 0,

        Insn::CArg { .. } => panic!("CArg undefined precedence"),
    }
}
