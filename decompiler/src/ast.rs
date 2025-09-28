use std::{borrow::Cow, collections::HashSet};

use crate::{
    cfg,
    mil::{self, ArithOp, BoolOp, CmpOp, Insn, Reg},
    pp::PP,
    ssa, ty,
};

#[derive(Debug)]
pub struct Ast<'a> {
    ssa: &'a ssa::Program,
    types: &'a ty::TypeSet,
    printed: HashSet<Reg>,
    is_named: ssa::RegMap<bool>,

    block_printed: cfg::BlockMap<bool>,
}

impl<'a> Ast<'a> {
    pub fn new(ssa: &'a ssa::Program, types: &'a ty::TypeSet) -> Self {
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

        Ast {
            ssa,
            types,
            printed: HashSet::new(),
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

        for reg in self.ssa.block_regs(bid) {
            if self.is_named(reg) || self.ssa.get(reg).unwrap().has_side_effects() {
                self.pp_stmt(pp, reg)?;
            }
        }

        match self.ssa.cfg().block_cont(bid) {
            cfg::BlockCont::Always(tgt) => {
                self.pp_continuation(pp, bid, tgt)?;
            }
            cfg::BlockCont::Conditional { pos, neg } => {
                let cond = self.ssa.find_last_matching(bid, |insn| match insn {
                    Insn::SetJumpCondition(value) => Some(value),
                    _ => None,
                });

                match cond {
                    Some(cond) => {
                        write!(pp, "if ")?;
                        self.pp_ref(pp, cond, 0)?;
                    }
                    None => {
                        write!(pp, "if ???")?;
                    }
                }

                write!(pp, " {{\n  ")?;
                pp.open_box();
                self.pp_continuation(pp, bid, pos)?;
                pp.close_box();
                write!(pp, "\n}}\n")?;

                self.pp_continuation(pp, bid, neg)?;
            }
        }

        for &child in self.ssa.cfg().dom_tree().children_of(bid) {
            if self.ssa.cfg().block_preds(child).len() > 1 {
                self.pp_block_labeled(pp, child)?;
            }
        }

        Ok(())
    }

    fn pp_continuation<W: PP + ?Sized>(
        &mut self,
        pp: &mut W,
        src_bid: cfg::BlockID,
        tgt: cfg::Dest,
    ) -> std::io::Result<()> {
        let cfg = self.ssa.cfg();

        match tgt {
            cfg::Dest::Ext(addr) => {
                write!(pp, "goto ext 0x{:x}", addr)?;
            }
            cfg::Dest::Block(tgt_bid) => {
                if cfg.block_preds(tgt_bid).len() == 1 {
                    self.pp_block_inner(pp, tgt_bid)?;
                } else {
                    let looping_back = cfg.dom_tree().imm_doms(src_bid).any(|i| i == tgt_bid);
                    let keyword = if looping_back { "loop" } else { "goto" };
                    write!(pp, "{keyword} T{}", tgt_bid.as_number())?;
                }
            }
            cfg::Dest::Indirect => {
                let target = self.ssa.find_last_matching(src_bid, |insn| match insn {
                    Insn::SetJumpTarget(value) => Some(value),
                    _ => None,
                });

                match target {
                    Some(target) => {
                        write!(pp, "goto (")?;
                        self.pp_ref(pp, target, 0)?;
                        write!(pp, ").*")?;
                    }
                    None => {
                        write!(
                            pp,
                            "goto (???).* /* internal bug: unspecified jump target */"
                        )?;
                    }
                }
            }
            cfg::Dest::Return => {
                let ret_val = self.ssa.find_last_matching(src_bid, |insn| match insn {
                    Insn::SetReturnValue(value) => Some(value),
                    _ => None,
                });

                match ret_val {
                    Some(ret_val) => {
                        write!(pp, "return ")?;
                        self.pp_ref(pp, ret_val, 0)?;
                    }
                    None => {
                        write!(
                            pp,
                            "return undefined /* actually unspecified in source program! */"
                        )?;
                    }
                }
            }
            cfg::Dest::Undefined => {
                write!(
                    pp,
                    "goto undefined /* warning: due to decompiler bug or limitation */"
                )?;
            }
        }

        Ok(())
    }

    /// Prints a single "statement line" in a block.
    ///
    /// Named instructions are wrapped in a `let x = ...;`.
    fn pp_stmt<W: PP + ?Sized>(&mut self, pp: &mut W, reg: Reg) -> std::io::Result<()> {
        // NOTE unlike pp_def, this function is called only when printing the
        // "toplevel" definition of a named or effectful instruction;
        //
        // This is called by pp_labeled_inputs

        if self.printed.contains(&reg) {
            return Ok(());
        }
        self.printed.insert(reg);

        let reg_type = self.ssa.reg_type(reg);
        let should_print_semicolon = if let Insn::Phi = self.ssa.get(reg).unwrap() {
            write!(pp, "let mut r{}: {:?}", reg.reg_index(), reg_type)?;
            true
        } else if self.is_named(reg) {
            write!(pp, "let r{}: {:?} = ", reg.reg_index(), reg_type)?;
            self.pp_def(pp, reg, 0)?
        } else {
            self.pp_def(pp, reg, 0)?
        };

        if should_print_semicolon {
            writeln!(pp, ";")?;
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
    ) -> std::io::Result<bool> {
        // NOTE this function is called in both cases:
        //  - printing the "toplevel" definition of a named or effectful instruction;
        //  - printing an instruction definition inline as part of the 1 dependent instruction
        // (For this reason, we can't pp_labeled_inputs here)
        let mut insn = self.ssa.get(reg).unwrap();

        if let Insn::Get(x) = insn {
            return self.pp_def(pp, x, parent_prec);
        }

        let self_prec = precedence(&insn);

        if self_prec < parent_prec {
            write!(pp, "(")?;
        }

        match insn {
            Insn::Void => write!(pp, "void")?,
            Insn::True => write!(pp, "true")?,
            Insn::False => write!(pp, "false")?,
            Insn::Int { value, .. } => {
                write!(pp, "{}", value)?;
            }
            Insn::Float32(fb) => {
                write!(pp, "{}", fb.value())?;
            }
            Insn::Float64(fb) => {
                write!(pp, "{}", fb.value())?;
            }
            Insn::ReinterpretFloat32(src) => {
                write!(pp, "<f32>(")?;
                self.pp_ref(pp, src, self_prec)?;
                write!(pp, ")")?;
            }
            Insn::ReinterpretFloat64(src) => {
                write!(pp, "<f64>(")?;
                self.pp_ref(pp, src, self_prec)?;
                write!(pp, ")")?;
            }

            Insn::Bytes(bytes) => {
                write!(pp, "0x<")?;
                for (ndx, byte) in bytes.as_slice().iter().enumerate() {
                    if ndx > 0 {
                        write!(pp, " ")?;
                    }
                    write!(pp, "{:02x}", byte)?;
                }
                write!(pp, ">")?;
            }

            Insn::Get(arg) => {
                write!(pp, "/* warning: residual Get */ ")?;
                self.pp_def(pp, arg, parent_prec)?;
            }

            Insn::Control(ctl_insn) => {
                // any effect on control-flow should be encoded only in the CFG.
                // by the time we get here, it should have been patched out and
                // *replaced* by the CFG's BlockConts. Still, we can cope by
                // treating Control as a normal instruction and only trust the
                // CFG to understand control flow structure.
                write!(pp, "/* warning: unexpected Control */ {:?}", ctl_insn)?;
            }

            // nothing to do: handled in this block's BlockCont (see pp_block_inner)
            Insn::SetJumpCondition(_) | Insn::SetJumpTarget(_) | Insn::SetReturnValue(_) => {}

            Insn::StructGetMember {
                struct_value,
                name,
                size: _,
            } => {
                self.pp_ref(pp, struct_value, self_prec)?;
                write!(pp, ".{}", name)?;
            }
            Insn::Struct {
                type_name,
                first_member,
                ..
            } => {
                if let Some(first_member) = first_member {
                    pp.open_box();
                    write!(pp, "struct {} {{", type_name)?;
                    for (name, reg) in self.ssa.get_struct_members(first_member) {
                        write!(pp, "\n  .{} = ", name)?;
                        pp.open_box();
                        self.pp_ref(pp, reg, self_prec)?;
                        write!(pp, ",")?;
                        pp.close_box();
                    }
                    write!(pp, "\n}}")?;
                    pp.close_box();
                } else {
                    write!(pp, "struct {} {{}}", type_name)?;
                }
            }
            Insn::StructMember { .. } => {
                unreachable!("CArg should be handled via the Call it belongs to!")
            }
            Insn::ArrayGetElement {
                array,
                index,
                size: _,
            } => {
                self.pp_ref(pp, array, self_prec)?;
                write!(pp, "[{}]", index)?;
            }
            Insn::Part { src, offset, size } => {
                self.pp_ref(pp, src, self_prec)?;
                // syntax is [end..start] because it's more intuitive with concatenation, e.g.:
                //   r13[8 .. 4] ++ r12[4 .. 0]
                write!(pp, "[{} .. {}]", offset + size, offset,)?;
            }
            Insn::Concat { lo, hi } => {
                self.pp_bin_op(pp, hi, lo, "++", self_prec)?;
            }
            Insn::Widen {
                reg: arg,
                target_size,
                sign,
            } => {
                self.pp_ref(pp, arg, self_prec)?;
                let ch = if sign { 'i' } else { 'u' };
                write!(pp, " as {}{}", ch, 8 * target_size)?;
            }
            Insn::Arith(arith_op, a, b) => {
                self.pp_bin_op(pp, a, b, arith_op_str(arith_op), self_prec)?;
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
            Insn::Cmp(cmp_op, a, b) => {
                let op_s = match cmp_op {
                    CmpOp::EQ => "EQ",
                    CmpOp::LT => "LT",
                };
                self.pp_bin_op(pp, a, b, op_s, self_prec)?;
            }
            Insn::Bool(bool_op, a, b) => {
                let op_s = match bool_op {
                    BoolOp::Or => "OR",
                    BoolOp::And => "AND",
                };
                self.pp_bin_op(pp, a, b, op_s, self_prec)?;
            }
            Insn::Not(operand) => {
                write!(pp, "! ")?;
                self.pp_ref(pp, operand, self_prec)?;
            }
            Insn::Call { callee, first_arg } => {
                if let Some(tyid) = self.ssa.value_type(callee) {
                    // Not quite correct (why would we print the type name?) but
                    // happens to be always correct for well formed programs
                    if let Some(name) = self.types.name(tyid) {
                        write!(pp, "{}", name)?;
                    } else {
                        write!(pp, "<?>")?;
                        self.pp_ref(pp, callee, self_prec)?;
                    }
                } else {
                    self.pp_ref(pp, callee, self_prec)?;
                }

                write!(pp, "(")?;
                pp.open_box();
                if let Some(first_arg) = first_arg {
                    for (ndx, arg) in self.ssa.get_call_args(first_arg).enumerate() {
                        if ndx > 0 {
                            writeln!(pp, ",")?;
                        }
                        self.pp_ref(pp, arg, self_prec)?;
                    }
                }
                pp.close_box();
                write!(pp, ")")?;
            }
            Insn::NotYetImplemented(msg) => {
                write!(pp, "TODO /* {} */", msg)?;
            }
            Insn::LoadMem { addr, size } => {
                let sz = size.try_into().unwrap();
                self.pp_load_mem(pp, addr, sz)?;
            }
            Insn::StoreMem { addr, value } => {
                write!(pp, "[")?;
                pp.open_box();
                self.pp_ref(pp, addr, self_prec)?;
                write!(pp, "]:* := ")?;
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
            Insn::UndefinedBool => {
                write!(pp, "undefined_bool")?;
            }
            Insn::UndefinedBytes { size } => {
                write!(pp, "undefined_bytes({})", size)?;
            }
            Insn::Ancestral { anc_name, .. } => {
                write!(pp, "pre:{}", anc_name.name())?;
            }

            Insn::Phi => {
                self.pp_def_default(pp, "phi".into(), insn.input_regs(), self_prec)?;
            }
            Insn::Upsilon { value, phi_ref } => match self.ssa.reg_type(value) {
                // `value` doesn't need to be printed here, as it's already been
                // printed (has side effect) and the target phi register's value
                // is not significant
                mil::RegType::Effect => {}
                _ => {
                    write!(pp, "r{} := ", phi_ref.reg_index())?;
                    pp.open_box();
                    self.pp_def(pp, value, 0)?;
                    pp.close_box();
                }
            },
            Insn::CArg { .. } => {
                unreachable!("CArg should be handled via the Call it belongs to!")
            }
            Insn::FuncArgument { index, .. } => {
                write!(pp, "$arg{}", index)?;
            }
        };

        if self_prec < parent_prec {
            write!(pp, ")")?;
        }
        Ok(true)
    }

    fn pp_bin_op<W: PP + ?Sized>(
        &mut self,
        pp: &mut W,
        a: Reg,
        b: Reg,
        op_str: &str,
        self_prec: u8,
    ) -> std::io::Result<()> {
        let (asz, bsz) = self.operands_sizes(a, b);

        self.pp_ref(pp, a, self_prec)?;
        if let Some(asz) = asz {
            write!(pp, " as i{}", asz)?;
        }

        write!(pp, " {} ", op_str)?;

        self.pp_ref(pp, b, self_prec)?;
        if let Some(bsz) = bsz {
            write!(pp, " as i{}", bsz)?;
        }

        Ok(())
    }

    fn operands_sizes(&self, a: Reg, b: Reg) -> (Option<usize>, Option<usize>) {
        let at = self.ssa.reg_type(a);
        let bt = self.ssa.reg_type(b);
        if let (Some(asz), Some(bsz)) = (at.bytes_size(), bt.bytes_size()) {
            if asz != bsz {
                return (Some(asz), Some(bsz));
            }
        }

        (None, None)
    }

    fn pp_def_default<W: PP + ?Sized>(
        &mut self,
        pp: &mut W,
        op_s: Cow<'_, str>,
        args: mil::ArgsMut,
        self_prec: u8,
    ) -> Result<(), std::io::Error> {
        write!(pp, "{} (", op_s)?;
        for (arg_ndx, arg) in args.into_iter().enumerate() {
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
                self.printed.contains(&arg),
                "arg needs let but not yet printed: {:?}",
                arg
            );
            write!(pp, "r{}", arg.reg_index())
        } else {
            let anything_printed = self.pp_def(pp, arg, parent_prec)?;
            assert!(anything_printed);
            Ok(())
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
        | Insn::Float32(_)
        | Insn::Float64(_)
        | Insn::Bytes(..)
        | Insn::ReinterpretFloat32(_)
        | Insn::ReinterpretFloat64(_)
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
            ArithOp::Shl | ArithOp::Shr | ArithOp::BitXor | ArithOp::BitAnd | ArithOp::BitOr => 202,
            ArithOp::Add | ArithOp::Sub => 200,
            ArithOp::Mul => 201,
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
