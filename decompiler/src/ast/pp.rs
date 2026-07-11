use super::*;
use crate::{mil, pp, ssa, ty};
use pp::PP as _;

pub fn write_ast<W: std::io::Write>(
    wrt: &mut pp::PrettyPrinter<W>,
    ast: &Ast,
    ssa: &ssa::Program,
    types: &ty::TypeSet,
) -> std::io::Result<()> {
    use std::io::Write;

    write!(wrt, "# block order: ")?;
    for (ndx, bid) in ast.block_order().iter().enumerate() {
        if ndx > 0 {
            write!(wrt, ", ")?;
        }
        write!(wrt, "B{}", bid.as_number())?;
    }
    writeln!(wrt)?;

    write_ast_node(wrt, ast, ssa, types, ast.root())?;
    writeln!(wrt)
}

pub fn write_ast_node<W: std::io::Write>(
    wrt: &mut pp::PrettyPrinter<W>,
    ast: &Ast,
    ssa: &ssa::Program,
    types: &ty::TypeSet,
    mut sid: StmtID,
) -> std::io::Result<()> {
    use std::io::Write;

    loop {
        let node = ast.get(sid);

        let sid_init = sid;

        match node {
            Stmt::NamedBlock { bid, body } => {
                write!(wrt, ".B{}: {{\n  ", bid.as_number())?;
                wrt.open_box();
                write_ast_node(wrt, ast, ssa, types, *body)?;
                wrt.close_box();
                write!(wrt, "\n}}")?;
                return Ok(());
            }
            Stmt::Let { name, value, body } => {
                write!(wrt, "let r{} = ", name.reg_index())?;
                wrt.open_box();
                write_ast_expr(
                    wrt,
                    ast,
                    ssa,
                    *value,
                    ExprFlags {
                        always_expand_root: true,
                    },
                )?;
                wrt.close_box();
                writeln!(wrt, ";")?;
                sid = *body;
            }
            Stmt::LetPhi { name, body } => {
                writeln!(wrt, "r{}: phi;", name.reg_index())?;
                sid = *body;
            }
            Stmt::Seq { first, then } => {
                write_ast_node(wrt, ast, ssa, types, *first)?;
                writeln!(wrt, ";")?;
                sid = *then;
            }
            Stmt::If { cond, cons, alt } => {
                if let Some(cond) = cond {
                    write!(wrt, "if ")?;
                    write_ast_expr(wrt, ast, ssa, *cond, ExprFlags::default())?;
                } else {
                    write!(wrt, "if ???")?;
                }
                write!(wrt, "\n  then  ")?;
                wrt.open_box();
                write_ast_node(wrt, ast, ssa, types, *cons)?;
                wrt.close_box();

                if ast.get(*alt) != &Stmt::Pass {
                    write!(wrt, "\n  else  ")?;
                    wrt.open_box();
                    write_ast_node(wrt, ast, ssa, types, *alt)?;
                    wrt.close_box();
                }

                return Ok(());
            }

            Stmt::Eval(reg) => {
                return write_ast_expr(
                    wrt,
                    ast,
                    ssa,
                    *reg,
                    ExprFlags {
                        always_expand_root: true,
                    },
                );
            }
            Stmt::Return(reg) => {
                write!(wrt, "return ")?;
                return write_ast_expr(wrt, ast, ssa, *reg, ExprFlags::default());
            }
            Stmt::JumpUndefined => {
                write!(wrt, "jump_undefined")?;
                return Ok(());
            }
            Stmt::JumpExternal(target) => {
                write!(wrt, "jump_external {:?}", target)?;
                return Ok(());
            }
            Stmt::JumpIndirect(reg) => {
                write!(wrt, "jump (")?;
                write_ast_expr(wrt, ast, ssa, *reg, ExprFlags::default())?;
                write!(wrt, ").*")?;
                return Ok(());
            }
            Stmt::Loop(bid) => {
                write!(wrt, "loop .B{}", bid.as_number())?;
                return Ok(());
            }
            Stmt::Jump(bid) => {
                write!(wrt, "jump .B{}", bid.as_number())?;
                return Ok(());
            }
            Stmt::Pass => {
                write!(wrt, "pass")?;
                return Ok(());
            }
        }

        // TODO extend to check we never visit a node twice
        assert_ne!(sid, sid_init);
    }
}

#[derive(Default)]
pub(crate) struct ExprFlags {
    pub(crate) always_expand_root: bool,
}

pub(crate) fn write_ast_expr<W: std::io::Write>(
    wrt: &mut W,
    ast: &Ast,
    ssa: &ssa::Program,
    reg: Reg,
    flags: ExprFlags,
) -> std::io::Result<()> {
    if !flags.always_expand_root && ast.is_value_named(reg) {
        write!(wrt, "r{}", reg.0)?;
        return Ok(());
    }

    let insn = ssa.get(reg).unwrap();

    // `Select` is rendered with C-like ternary syntax instead of the generic
    // `Select cond then else` opcode form, so the output reads naturally as
    // `(cond ? pos : neg)`.  The sub-expressions are rendered with the same
    // `write_ast_reg` path (which parenthesizes unnamed composite values).
    if let mil::Insn::Select {
        cond,
        then_val,
        else_val,
    } = insn
    {
        write!(wrt, "(")?;
        write_ast_reg(wrt, ast, ssa, *cond, false)?;
        write!(wrt, " ? ")?;
        write_ast_reg(wrt, ast, ssa, *then_val, false)?;
        write!(wrt, " : ")?;
        write_ast_reg(wrt, ast, ssa, *else_val, false)?;
        write!(wrt, ")")?;
        return Ok(());
    }

    let xpinsn = mil::to_expanded(insn);

    write!(wrt, "{} ", xpinsn.opcode)?;
    for (ndx, (_key, arg)) in xpinsn.fields.into_iter().enumerate() {
        if ndx > 0 {
            write!(wrt, " ")?;
        }
        // write!(wrt, "{}:", key)?;
        match arg {
            mil::ExpandedValue::Reg(reg) => {
                write_ast_reg(wrt, ast, ssa, reg, true)?;
            }
            mil::ExpandedValue::Generic(repr) => {
                write!(wrt, "{}", repr)?;
            }
            mil::ExpandedValue::RegList(regs) => {
                write!(wrt, "[")?;
                for (ndx, reg) in regs.into_iter().enumerate() {
                    if ndx > 0 {
                        write!(wrt, ", ")?;
                    }
                    write_ast_reg(wrt, ast, ssa, reg, false)?;
                }
                write!(wrt, "]")?;
            }
        }
    }

    Ok(())
}

fn write_ast_reg<W: std::io::Write>(
    wrt: &mut W,
    ast: &Ast,
    ssa: &ssa::Program,
    reg: Reg,
    parens_required: bool,
) -> Result<(), std::io::Error> {
    let _: () = if ast.is_value_named(reg) {
        write!(wrt, "r{}", reg.reg_index())?;
    } else {
        if parens_required {
            write!(wrt, "(")?;
        }
        write_ast_expr(wrt, ast, ssa, reg, ExprFlags::default())?;
        if parens_required {
            write!(wrt, ")")?;
        }
    };
    Ok(())
}
