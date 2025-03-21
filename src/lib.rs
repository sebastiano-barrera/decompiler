mod ast;
mod cfg;
#[macro_use]
mod mil;
pub mod pp;
mod ssa;
mod tests;
mod ty;
mod x86_to_mil;
#[cfg(any())]
mod xform;

mod xform {
    pub fn canonical(_: &mut super::ssa::Program) {
        //
    }
}

pub mod test_tool;
