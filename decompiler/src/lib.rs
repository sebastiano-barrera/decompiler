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
}

mod util;

pub mod api;
pub mod pp;

pub use api::*;
