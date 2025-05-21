mod ast;
mod cfg;
mod elf;
#[macro_use]
mod mil;
mod ssa;
mod tests;
mod ty;
mod x86_to_mil;

// temporarily disabled
#[cfg(any())]
mod xform;
mod xform {
    use crate::ssa;
    pub fn canonical(prog: &mut ssa::Program) {
        // do nothing!
    }

    pub fn eliminate_dead_code(prog: &mut ssa::Program) {
        let mut is_read = ssa::RegMap::for_program(prog, false);

        for bid in prog.cfg().block_ids_postorder() {
            for reg in prog.block_regs(bid).rev() {
                let mut insn = prog[reg].get();

                if insn.has_side_effects() {
                    is_read[reg] = true;
                }

                if is_read[reg] {
                    for &mut input in insn.input_regs() {
                        is_read[input] = true;
                    }
                }
            }
        }

        for (reg, &is_read) in is_read.items() {
            use crate::mil;
            if !is_read {
                prog[reg].set(mil::Insn::Void);
            }
        }
    }
}

mod util;

pub mod api;
pub mod pp;

pub use api::*;
