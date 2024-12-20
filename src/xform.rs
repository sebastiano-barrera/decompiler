#[cfg(test)]
use crate::{mil, ssa};

#[cfg(test)]
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

    fn assoc_of(prog: &ssa::Program, mut ndx: mil::Index) -> Assoc {
        loop {
            match prog.get(ndx).unwrap().insn {
                Insn::Const1(k) => return Assoc::Const(*k as i64),
                Insn::Const2(k) => return Assoc::Const(*k as i64),
                Insn::Const4(k) => return Assoc::Const(*k as i64),
                Insn::Const8(k) => return Assoc::Const(*k as i64),
                Insn::AddK(r, k) => return Assoc::Add(*r, *k),
                Insn::MulK(r, k) => return Assoc::Mul(*r, *k as i64),
                Insn::Get(r) => {
                    ndx = r.reg_index();
                }
                _ => return Assoc::Opaque,
            };
        }
    }

    let mut prog = prog.edit();
    for ndx in 0..prog.len() {
        let insn = *prog.get_mut(ndx).unwrap().insn;
        let (a, b) = match insn {
            Insn::Add(a, b) | Insn::Mul(a, b) => (a, b),
            _ => continue,
        };

        let a = a.reg_index();
        let b = b.reg_index();
        assert!(a < ndx);
        assert!(b < ndx);
        let aa = assoc_of(&*prog, a);
        let ba = assoc_of(&*prog, b);

        let repl_insn = match insn {
            Insn::Add(_, _) => match (aa, ba) {
                (Assoc::Const(0), _) => Some(Insn::Get(Reg(b))),
                (_, Assoc::Const(0)) => Some(Insn::Get(Reg(a))),
                (Assoc::Const(ak), Assoc::Const(bk)) => Some(Insn::Const8((ak + bk) as u64)),
                (Assoc::Const(ak), Assoc::Add(r, bk)) => Some(Insn::AddK(r, ak + bk)),
                (Assoc::Add(r, ak), Assoc::Const(bk)) => Some(Insn::AddK(r, ak + bk)),
                (Assoc::Const(ak), _) => Some(Insn::AddK(Reg(b), ak)),
                (_, Assoc::Const(bk)) => Some(Insn::AddK(Reg(a), bk)),
                (_, _) => None,
            },

            Insn::Mul(_, _) => match (aa, ba) {
                (Assoc::Const(1), _) => Some(Insn::Get(Reg(b))),
                (_, Assoc::Const(1)) => Some(Insn::Get(Reg(a))),
                (Assoc::Const(ak), Assoc::Const(bk)) => Some(Insn::Const8((ak * bk) as u64)),
                (Assoc::Const(ak), Assoc::Mul(r, bk)) => Some(Insn::MulK(r, ak * bk)),
                (Assoc::Mul(r, ak), Assoc::Const(bk)) => Some(Insn::MulK(r, ak * bk)),
                (Assoc::Const(ak), _) => Some(Insn::MulK(Reg(b), ak)),
                (_, Assoc::Const(bk)) => Some(Insn::MulK(Reg(a), bk)),
                (_, _) => None,
            },

            _ => None,
        };

        // reborrow here, so that the match above runs with prog borrowed immut.
        if let Some(repl_insn) = repl_insn {
            *prog.get_mut(ndx).unwrap().insn = repl_insn;
        }
    }
}
