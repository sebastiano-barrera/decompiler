use crate::{cfg, mil, ssa};

pub fn fold_constants(prog: &mut ssa::Program) {
    use mil::{Insn, Reg};

    /// Associativity status of an instruction.
    ///
    /// It's a "projection" of an instruction: only represents just the bit of info
    /// that the algorithm cares about. Mostly for simplifying the algo and reducing branching.
    enum Assoc {
        Opaque,
        Const(i64),
        Add(Reg, i64),
        Mul(Reg, i64),
    }

    fn assoc_of(prog: &ssa::Program, mut reg: mil::Reg) -> Assoc {
        loop {
            match prog.get(reg).unwrap().insn {
                Insn::Const1(k) => return Assoc::Const(*k as i64),
                Insn::Const2(k) => return Assoc::Const(*k as i64),
                Insn::Const4(k) => return Assoc::Const(*k as i64),
                Insn::Const8(k) => return Assoc::Const(*k as i64),
                Insn::AddK(r, k) => return Assoc::Add(*r, *k),
                Insn::MulK(r, k) => return Assoc::Mul(*r, *k as i64),
                Insn::Get(r) => {
                    reg = *r;
                }
                _ => return Assoc::Opaque,
            };
        }
    }

    let mut prog = prog.edit();
    let order = cfg::traverse_reverse_postorder(prog.cfg());

    for &bid in order.block_ids() {
        let insns = prog.block_normal_insns(bid).unwrap();
        for (reg, insn) in insns.iter() {
            let (a, b) = match insn {
                Insn::Add(a, b) | Insn::Mul(a, b) => (a, b),
                _ => continue,
            };

            let aa = assoc_of(&*prog, a);
            let ba = assoc_of(&*prog, b);

            let repl_insn = match insn {
                Insn::Add(_, _) => match (aa, ba) {
                    (Assoc::Const(0), _) => Some(Insn::Get(b)),
                    (_, Assoc::Const(0)) => Some(Insn::Get(a)),
                    (Assoc::Const(ak), Assoc::Const(bk)) => Some(Insn::Const8((ak + bk) as u64)),
                    (Assoc::Const(ak), Assoc::Add(r, bk)) => Some(Insn::AddK(r, ak + bk)),
                    (Assoc::Add(r, ak), Assoc::Const(bk)) => Some(Insn::AddK(r, ak + bk)),
                    (Assoc::Const(ak), _) => Some(Insn::AddK(b, ak)),
                    (_, Assoc::Const(bk)) => Some(Insn::AddK(a, bk)),
                    (_, _) => None,
                },

                Insn::Mul(_, _) => match (aa, ba) {
                    (Assoc::Const(1), _) => Some(Insn::Get(b)),
                    (_, Assoc::Const(1)) => Some(Insn::Get(a)),
                    (Assoc::Const(ak), Assoc::Const(bk)) => Some(Insn::Const8((ak * bk) as u64)),
                    (Assoc::Const(ak), Assoc::Mul(r, bk)) => Some(Insn::MulK(r, ak * bk)),
                    (Assoc::Mul(r, ak), Assoc::Const(bk)) => Some(Insn::MulK(r, ak * bk)),
                    (Assoc::Const(ak), _) => Some(Insn::MulK(b, ak)),
                    (_, Assoc::Const(bk)) => Some(Insn::MulK(a, bk)),
                    (_, _) => None,
                },

                _ => None,
            };

            // reborrow here, so that the match above runs with prog borrowed immut.
            if let Some(repl_insn) = repl_insn {
                *prog.get_mut(reg).unwrap().insn = repl_insn;
            }
        }
    }
}
