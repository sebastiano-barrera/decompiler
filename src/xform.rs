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
        None,
        Const(i64),
        Add(Reg, i64),
        Mul(Reg, i64),
    }
    impl Assoc {
        fn of(insn: &Insn) -> Self {
            match insn {
                Insn::Const1(k) => Assoc::Const(*k as i64),
                Insn::Const2(k) => Assoc::Const(*k as i64),
                Insn::Const4(k) => Assoc::Const(*k as i64),
                Insn::Const8(k) => Assoc::Const(*k as i64),
                Insn::AddK(r, k) => Assoc::Add(*r, *k),
                Insn::MulK(r, k) => Assoc::Mul(*r, *k as i64),
                _ => Assoc::None,
            }
        }
    }

    let mut prog = prog.edit();
    for ndx in 0..prog.len() {
        let insn = prog.get_mut(ndx).unwrap();
        let repl_insn = match insn.insn {
            Insn::Add(a, b) => {
                let a = a.reg_index();
                let b = b.reg_index();
                assert!(a < ndx);
                assert!(b < ndx);
                let aa = Assoc::of(prog.get(a).unwrap().insn);
                let ba = Assoc::of(prog.get(b).unwrap().insn);

                match (aa, ba) {
                    (Assoc::Const(ak), Assoc::Const(bk)) => Some(Insn::Const8((ak + bk) as u64)),
                    (Assoc::Const(ak), Assoc::Add(r, bk)) => Some(Insn::AddK(r, ak + bk)),
                    (Assoc::Add(r, ak), Assoc::Const(bk)) => Some(Insn::AddK(r, ak + bk)),
                    (Assoc::Const(ak), _) => Some(Insn::AddK(Reg(b), ak)),
                    (_, Assoc::Const(bk)) => Some(Insn::AddK(Reg(a), bk)),
                    (_, _) => None,
                }
            }
            Insn::Mul(a, b) => {
                let a = a.reg_index();
                let b = b.reg_index();
                let aa = Assoc::of(prog.get(a).unwrap().insn);
                let ba = Assoc::of(prog.get(b).unwrap().insn);

                match (aa, ba) {
                    (Assoc::Const(ak), Assoc::Const(bk)) => Some(Insn::Const8((ak * bk) as u64)),
                    (Assoc::Const(ak), Assoc::Mul(r, bk)) => Some(Insn::MulK(r, ak * bk)),
                    (Assoc::Mul(r, ak), Assoc::Const(bk)) => Some(Insn::MulK(r, ak * bk)),
                    (Assoc::Const(ak), _) => Some(Insn::MulK(Reg(b), ak)),
                    (_, Assoc::Const(bk)) => Some(Insn::MulK(Reg(a), bk)),
                    (_, _) => None,
                }
            }
            _ => continue,
        };

        // reborrow here, so that the match above runs with prog borrowed immut.
        if let Some(repl_insn) = repl_insn {
            *prog.get_mut(ndx).unwrap().insn = repl_insn;
        }
    }
}
