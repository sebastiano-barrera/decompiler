use super::*;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;
#[derive(Error, Debug)]
pub enum Error {
    #[error("expected an if statement")]
    ExpectedIf,

    #[error("while mutating SSA: {0}")]
    SSAMutation(#[from] ssa::MutationError),
}

pub fn invert_if(ast: &mut Ast, ssa: &mut ssa::Program, sid: StmtID) -> Result<()> {
    let &Stmt::If {
        cond: Some(cond),
        cons,
        alt,
    } = ast.get(sid)
    else {
        return Err(Error::ExpectedIf);
    };

    ssa.mutate(|mut prog| {
        let new_cond: Reg = prog.invert_bool(cond)?;
        ast.nodes[sid.0] = Stmt::If {
            cond: Some(new_cond),
            cons: alt,
            alt: cons,
        };
        Ok(())
    })
}
