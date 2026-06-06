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

/// Some additional "cleanup" operations that require access to the dataflow graph (the ssa::Program)
pub fn cleanup_with_ssa(ast: &mut Ast, ssa: &mut ssa::Program) {
    for sid in ast.stmt_ids() {
        let _ = invert_biased_if(ast, ssa, sid);
    }
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

    invert_if_inner(ast, ssa, sid, cond, cons, alt)
}

pub fn invert_biased_if(ast: &mut Ast, ssa: &mut ssa::Program, sid: StmtID) -> Result<()> {
    let &Stmt::If {
        cond: Some(cond),
        cons,
        alt,
    } = ast.get(sid)
    else {
        return Ok(());
    };

    if matches!(ast.get(cons), Stmt::Pass) {
        return invert_if_inner(ast, ssa, sid, cond, cons, alt);
    }
    Ok(())
}

fn invert_if_inner(
    ast: &mut Ast,
    ssa: &mut ssa::Program,
    sid: StmtID,
    cond: Reg,
    cons: StmtID,
    alt: StmtID,
) -> Result<()> {
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
