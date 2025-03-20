mod cfg;
#[macro_use]
mod mil;
pub mod pp;
mod ssa;
mod tests;
mod ty;
mod x86_to_mil;

// TODO rebuild and re-enable for SoN repr

mod ast;

#[cfg(any())]
mod xform;
mod xform {
    pub fn canonical(prog: &mut super::ssa::Program) {}
}

pub mod test_tool;
