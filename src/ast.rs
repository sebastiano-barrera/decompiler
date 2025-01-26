use std::borrow::Cow;

use crate::{
    cfg,
    mil::{ArithOp, BoolOp, CmpOp, Insn, Reg},
    pp::PP,
    ssa,
};

#[derive(Debug)]
pub struct Ast<'a> {
    ssa: &'a ssa::Program,
    is_named: Vec<bool>,
}

impl<'a> Ast<'a> {
    pub fn new(ssa: &'a ssa::Program) -> Self {
        let mut is_named = vec![false; ssa.reg_count() as usize];
        for iv in ssa.insns_unordered() {
            let dest = iv.dest.get();
            let insn = iv.insn.get();

            is_named[dest.reg_index() as usize] = match insn {
                    Insn::Const1(_)
                    | Insn::Const2(_)
                    | Insn::Const4(_)
                    | Insn::Const8(_)
                    | Insn::LoadMem1(_)
                    | Insn::LoadMem2(_)
                    | Insn::LoadMem4(_)
                    | Insn::LoadMem8(_)
                    // ancestral values are akin to (named) consts
                    | Insn::Ancestral(_) => false,

                    Insn::Phi1
                    | Insn::Phi2
                    | Insn::Phi4
                    | Insn::Phi8 => true,

                    Insn::Call { .. } => true,
                    _ => ssa.readers_count(dest) > 1,
                };
        }

        Ast { ssa, is_named }
    }

    pub fn pretty_print<W: PP + ?Sized>(&self, pp: &mut W) -> std::fmt::Result {
        let cfg = self.ssa.cfg();
        let entry_bid = cfg.entry_block_id();
        self.pp_block(pp, entry_bid)
    }

    fn pp_block<W: PP + ?Sized>(&self, pp: &mut W, bid: cfg::BlockID) -> std::fmt::Result {
        let phis = self.ssa.block_phi(bid);

        write!(pp, "T{}(", bid.as_number())?;
        for (ndx, phi_reg) in phis.phi_regs().enumerate() {
            if ndx > 0 {
                write!(pp, ", ")?;
            }
            write!(pp, "{:?}", phi_reg)?;
        }
        write!(pp, "): {{\n  ")?;
        pp.open_box();

        self.pp_block_inner(pp, bid)?;

        pp.close_box();
        writeln!(pp, "}}")
    }

    fn pp_block_inner<W: PP + ?Sized>(
        &self,
        pp: &mut W,
        bid: cfg::BlockID,
    ) -> Result<(), std::fmt::Error> {
        let insn_slice = self.ssa.block_normal_insns(bid).unwrap();
        for (dest, insn) in insn_slice.iter_copied() {
            let is_named = self.is_named(dest);
            if is_named
                || (insn.has_side_effects()
                    && !matches!(insn, Insn::CArg(_) | Insn::Jmp(_) | Insn::JmpIf { .. }))
            {
                if is_named {
                    write!(pp, "let r{} = ", dest.reg_index())?;
                }
                pp.open_box();
                self.pp_insn(pp, dest)?;
                writeln!(pp, ";")?;
                pp.close_box();
            }
        }

        let cfg = self.ssa.cfg();

        match cfg.block_cont(bid) {
            cfg::BlockCont::End => {
                // all done!
            }
            cfg::BlockCont::Jmp((pred_ndx, tgt)) => {
                self.pp_continuation(pp, bid, tgt, pred_ndx as u16)?;
            }
            cfg::BlockCont::Alt {
                straight: (neg_pred_ndx, neg_bid),
                side: (pos_pred_ndx, pos_bid),
            } => {
                let last_insn = insn_slice.insns.last().unwrap().get();
                let cond = match last_insn {
                    Insn::JmpIf { cond, target: _ } => cond,
                    _ => panic!("block with BlockCont::Alt continuation must end with a JmpIf"),
                };

                write!(pp, "if ")?;
                self.pp_arg(pp, cond)?;
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
                self.pp_block(pp, child)?;
            }
        }

        Ok(())
    }

    fn pp_continuation<W: PP + ?Sized>(
        &self,
        pp: &mut W,
        src_bid: cfg::BlockID,
        tgt_bid: cfg::BlockID,
        pred_ndx: u16,
    ) -> std::fmt::Result {
        let phi = self.ssa.block_phi(tgt_bid);

        let cfg = self.ssa.cfg();
        let looping_back = cfg
            .dom_tree()
            .imm_doms(src_bid)
            .find(|i| *i == tgt_bid)
            .is_some();
        let keyword = if looping_back { "loop" } else { "goto" };

        let pred_count = cfg.block_preds(tgt_bid).len();
        if pred_count == 1 {
            assert_eq!(phi.phi_count(), 0);
            return self.pp_block_inner(pp, tgt_bid);
        }

        if phi.phi_count() == 0 {
            write!(pp, "{keyword} T{}", tgt_bid.as_number())?;
        } else {
            write!(pp, "{keyword} T{} (", tgt_bid.as_number())?;
            for phi_ndx in 0..phi.phi_count() {
                let phi_reg = phi.phi_reg(phi_ndx);
                write!(pp, "\n  r{} = ", phi_reg.0)?;

                let arg = phi.arg(&(&self).ssa, phi_ndx, pred_ndx);
                pp.open_box();
                self.pp_arg(pp, arg)?;
                pp.close_box();
            }

            write!(pp, "\n)")?;
        }

        writeln!(pp)
    }

    fn is_named(&self, reg: Reg) -> bool {
        self.is_named[reg.reg_index() as usize]
    }

    fn pp_insn<W: PP + ?Sized>(&self, pp: &mut W, reg: Reg) -> std::fmt::Result {
        let iv = self.ssa.get(reg).unwrap();
        let insn = iv.insn.get();

        let op_s: Cow<str> = match insn {
            Insn::True => "True".into(),
            Insn::False => "False".into(),
            Insn::Const1(k) => return write!(pp, "0x{:x} /* {} */", k, k),
            Insn::Const2(k) => return write!(pp, "0x{:x} /* {} */", k, k),
            Insn::Const4(k) => return write!(pp, "0x{:x} /* {} */", k, k),
            Insn::Const8(k) => return write!(pp, "0x{:x} /* {} */", k, k),
            Insn::L1(_) => "L1".into(),
            Insn::L2(_) => "L2".into(),
            Insn::L4(_) => "L4".into(),
            Insn::Get8(_) => "Get8".into(),
            Insn::StructGet8 {
                struct_value: _,
                offset,
            } => format!("StructGet8[{offset}]").into(),
            Insn::V8WithL1(_, _) => "V8WithL1".into(),
            Insn::V8WithL2(_, _) => "V8WithL2".into(),
            Insn::V8WithL4(_, _) => "V8WithL4".into(),
            Insn::Widen1_2(_) => "Widen1_2".into(),
            Insn::Widen1_4(_) => "Widen1_4".into(),
            Insn::Widen1_8(_) => "Widen1_8".into(),
            Insn::Widen2_4(_) => "Widen2_4".into(),
            Insn::Widen2_8(_) => "Widen2_8".into(),
            Insn::Widen4_8(_) => "Widen4_8".into(),
            Insn::Arith1(arith_op, a, b)
            | Insn::Arith2(arith_op, a, b)
            | Insn::Arith4(arith_op, a, b)
            | Insn::Arith8(arith_op, a, b) => {
                self.pp_arg(pp, a)?;

                let op_s = match arith_op {
                    ArithOp::Add => " + ",
                    ArithOp::Sub => " - ",
                    ArithOp::Mul => " * ",
                    ArithOp::Shl => " / ",
                    ArithOp::BitXor => " ^ ",
                    ArithOp::BitAnd => " & ",
                    ArithOp::BitOr => " | ",
                };
                write!(pp, "{}", op_s)?;

                self.pp_arg(pp, b)?;
                return Ok(());
            }
            Insn::ArithK1(arith_op, reg, k)
            | Insn::ArithK2(arith_op, reg, k)
            | Insn::ArithK4(arith_op, reg, k)
            | Insn::ArithK8(arith_op, reg, k) => {
                self.pp_arg(pp, reg)?;
                let op_s = match arith_op {
                    ArithOp::Add => " + ",
                    ArithOp::Sub => " - ",
                    ArithOp::Mul => " * ",
                    ArithOp::Shl => " / ",
                    ArithOp::BitXor => " ^ ",
                    ArithOp::BitAnd => " & ",
                    ArithOp::BitOr => " | ",
                };
                write!(pp, "{}{}", op_s, k)?;
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
                self.pp_arg(pp, callee)?;
                write!(pp, "(")?;
                pp.open_box();
                for (ndx, arg) in self.ssa.get_call_args(reg).enumerate() {
                    if ndx > 0 {
                        writeln!(pp, ",")?;
                    }
                    self.pp_arg(pp, arg)?;
                }
                pp.close_box();
                write!(pp, ")")?;
                return Ok(());
            }
            Insn::CArg(_) => panic!("CArg not handled via this path!"),
            Insn::Ret(_) => "Ret".into(),
            Insn::TODO(msg) => return write!(pp, "TODO /* {} */", msg),
            Insn::LoadMem1(addr) => return self.pp_load_mem(pp, addr, 1),
            Insn::LoadMem2(addr) => return self.pp_load_mem(pp, addr, 2),
            Insn::LoadMem4(addr) => return self.pp_load_mem(pp, addr, 4),
            Insn::LoadMem8(addr) => return self.pp_load_mem(pp, addr, 8),
            Insn::StoreMem(addr, value) => {
                write!(pp, "[")?;
                pp.open_box();
                self.pp_arg(pp, addr)?;
                write!(pp, "] = ")?;
                pp.open_box();
                self.pp_arg(pp, value)?;
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

            Insn::Phi1 | Insn::Phi2 | Insn::Phi4 | Insn::Phi8 | Insn::PhiBool | Insn::PhiArg(_) => {
                panic!("phi insns should not be reachable here")
            }
            // handled by pp_block
            Insn::Jmp(_) | Insn::JmpIf { .. } => {
                panic!("jump insns should not be reachable here")
            }

            Insn::JmpInd(_) => "JmpInd".into(),
            Insn::JmpExt(addr) => return write!(pp, "JmpExt(0x{:x})", addr),
            Insn::JmpExtIf { cond, addr } => {
                write!(pp, "if ")?;
                self.pp_arg(pp, cond)?;
                write!(pp, "{{\n  goto 0x{:0x}\n}}", addr)?;
                return Ok(());
            }

            Insn::StructGetMember { struct_value, name } => {
                self.pp_arg(pp, struct_value)?;
                write!(pp, ".\"{}\"", name);
                return Ok(());
            }
        };

        write!(pp, "{}(", op_s)?;

        for (arg_ndx, arg) in insn.input_regs_iter().enumerate() {
            if arg_ndx > 0 {
                write!(pp, ", ")?;
            }

            self.pp_arg(pp, arg)?;
        }

        write!(pp, ")")?;
        Ok(())
    }

    fn pp_load_mem<W: PP + ?Sized>(
        &self,
        pp: &mut W,
        addr: Reg,
        sz: i32,
    ) -> Result<(), std::fmt::Error> {
        write!(pp, "[")?;
        pp.open_box();
        self.pp_arg(pp, addr)?;
        pp.close_box();
        write!(pp, "]:{}", sz)
    }

    fn pp_arg<W: PP + ?Sized>(&self, pp: &mut W, arg: Reg) -> Result<(), std::fmt::Error> {
        Ok(if self.is_named(arg) {
            write!(pp, "r{}", arg.reg_index())?;
        } else {
            self.pp_insn(pp, arg)?;
        })
    }
}
