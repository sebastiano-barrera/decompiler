use decompiler::{api::proto, pp::PP};
use std::{fs::File, io::Read};

struct CliOptions {
    exe: String,
    object: Object,
    output_format: OutputFormat,
}
enum Object {
    Function { name: String },
    FunctionAll,
    Types,
}
#[allow(non_camel_case_types)]
enum OutputFormat {
    JSON,
    SSA_Text,
    AST_DOT,
    AST_Text,
}

fn main() {
    tracing_subscriber::fmt::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let mut args = std::env::args();
    let program_name = args.next().unwrap();
    let opts = match parse_cli(args) {
        Ok(opts) => opts,
        Err(err) => {
            eprintln!("error: {}", err);
            eprintln!(
                "usage: {} --exe EXE [--func FUNCTION | --types]",
                program_name
            );
            eprintln!("");
            eprintln!("       EXE = path to the executable (only ELF is supported)");
            eprintln!("");
            eprintln!("  FUNCTION = name of the function to analyze (e.g. 'main')");
            eprintln!("             (If absent, all functions are decompiled in parallel)");
            eprintln!("");
            eprintln!("    FORMAT = format for the output ");
            eprintln!("             (supported: json, ast/text, ast/json, ast/dot)");
            return;
        }
    };

    // TODO Replace with memory mapping? (but it requires locking, see memmap2's docs)
    // https://docs.rs/memmap2/0.9.5/memmap2/struct.Mmap.html#safety
    let contents = {
        let mut contents = Vec::new();
        let mut elf = File::open(&opts.exe).expect("could not open executable");
        elf.read_to_end(&mut contents).expect("read error");
        contents
    };

    let exe = match decompiler::Executable::parse(&contents) {
        Err(err) => {
            eprintln!("error: {}", err);
            return;
        }
        Ok(exe) => exe,
    };

    match opts.object {
        Object::Function { name } => {
            dump_functions(&exe, &[&name], opts.output_format).unwrap();
        }
        Object::FunctionAll => {
            let mut function_names: Vec<_> = exe.function_names().collect();
            // a stable ordering is often useful when re-running this program over and
            // over while tracking some issue
            function_names.sort();
            dump_functions(&exe, &function_names, opts.output_format).unwrap();
        }
        Object::Types => todo!(),
    };
}

fn dump_functions(
    exe: &decompiler::Executable<'_>,
    function_names: &[&str],
    output_format: OutputFormat,
) -> anyhow::Result<()> {
    for function_name in function_names {
        let df = exe
            .decompile_function(&function_name)
            .expect("decompiling function");

        let stdout = std::io::stdout().lock();

        match output_format {
            OutputFormat::JSON => {
                let doc = proto::Function::from(&df);
                serde_json::to_writer_pretty(stdout, &doc)?;
            }
            OutputFormat::SSA_Text => {
                let doc = proto::Function::from(&df);
                write_text(&doc, stdout)?;
            }
            OutputFormat::AST_DOT => {
                let Some(ssa) = df.ssa() else {
                    anyhow::bail!("could not produce ssa/cfg");
                };

                write_cfg_dot(ssa.cfg(), stdout)?;
            }
            OutputFormat::AST_Text => {
                let Some(ssa) = df.ssa() else {
                    anyhow::bail!("could not produce ssa/cfg");
                };
                let Some(ast) = df.ast() else {
                    anyhow::bail!("could not produce ast");
                };
                let mut pp = decompiler::pp::PrettyPrinter::start(stdout);
                write_ast(&mut pp, ast, ssa, exe.types())?;
            }
        }
    }
    Ok(())
}

fn write_cfg_dot<W: std::io::Write>(cfg: &decompiler::Graph, mut wrt: W) -> anyhow::Result<()> {
    writeln!(wrt, "digraph {{")?;

    for bid in cfg.block_ids() {
        for dest in cfg.block_cont(bid).block_dests() {
            writeln!(wrt, "  {} -> {};", bid.as_number(), dest.as_number())?;
        }
    }

    writeln!(wrt, "}}")?;

    Ok(())
}

fn write_text<W: std::io::Write>(doc: &proto::Function, mut wrt: W) -> std::io::Result<()> {
    let mut last_addr = None;
    match doc.mil.as_ref() {
        Some(mil) => {
            writeln!(wrt, "# mil")?;
            for insn in &mil.body {
                write_text_insn(&mut wrt, &mut last_addr, insn)?;
            }
        }
        None => {
            writeln!(wrt, "# mil: no")?;
        }
    }

    writeln!(wrt)?;

    let mut last_addr = None;
    match doc.ssa.as_ref() {
        Some(ssa) => {
            writeln!(wrt, "# ssa")?;
            for (bid, block) in ssa.blocks.iter() {
                writeln!(wrt, "B{}:", bid.as_number())?;
                for insn in &block.body {
                    write_text_insn(&mut wrt, &mut last_addr, insn)?;
                }
                writeln!(wrt, "    -> {:?}", block.cont)?;
            }
        }
        None => {
            writeln!(wrt, "# ssa: no")?;
        }
    }

    Ok(())
}

fn write_text_insn<W: std::io::Write>(
    wrt: &mut W,
    last_addr: &mut Option<u64>,
    insn: &proto::Insn,
) -> std::io::Result<()> {
    if *last_addr != insn.addr {
        if let Some(addr) = insn.addr {
            writeln!(wrt, "0x{:x}:", addr)?;
        }
        *last_addr = insn.addr;
    }
    write!(wrt, "    r{:<4}", insn.dest)?;
    match insn.reg_type {
        Some(decompiler::RegType::Bytes(n)) => write!(wrt, ": {:6} ", n)?,
        Some(decompiler::RegType::Bool) => write!(wrt, ":   bool ")?,
        Some(decompiler::RegType::Effect) => write!(wrt, ": effect ")?,
        Some(decompiler::RegType::Error) => write!(wrt, ":    !!! ")?,
        None => write!(wrt, ":      ? ")?,
    }
    write!(wrt, " <- {:10} ", insn.insn.opcode)?;
    for (name, arg) in insn.insn.fields.iter() {
        match arg {
            decompiler::ExpandedValue::Reg(reg) => write!(wrt, "{}:r{}  ", name, reg.0)?,
            decompiler::ExpandedValue::Generic(s) => write!(wrt, "{}:{}  ", name, s)?,
        }
    }
    if let Some(tyid) = insn.tyid {
        write!(wrt, "  ;; {:?}", tyid)?;
    }
    writeln!(wrt)?;
    Ok(())
}

pub fn write_ast<W: std::io::Write>(
    wrt: &mut decompiler::pp::PrettyPrinter<W>,
    ast: &decompiler::Ast,
    ssa: &decompiler::SSAProgram,
    types: &decompiler::ty::TypeSet,
) -> std::io::Result<()> {
    use std::io::Write;

    write!(wrt, "# block order: ")?;
    for (ndx, bid) in ast.block_order().into_iter().enumerate() {
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
    wrt: &mut decompiler::pp::PrettyPrinter<W>,
    ast: &decompiler::Ast,
    ssa: &decompiler::SSAProgram,
    types: &decompiler::ty::TypeSet,
    mut sid: decompiler::ast::StmtID,
) -> std::io::Result<()> {
    use decompiler::ast::Stmt;
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
                write!(wrt, "let {} = ", name)?;
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
                write!(wrt, ";\n")?;
                sid = *body;
            }
            Stmt::LetPhi { name, body } => {
                writeln!(wrt, "{}: phi;", name)?;
                sid = *body;
            }
            Stmt::Seq { first, then } => {
                write_ast_node(wrt, ast, ssa, types, *first)?;
                write!(wrt, ";\n")?;
                sid = *then;
            }
            Stmt::If { cond, cons, alt } => {
                if let Some(cond) = cond {
                    write!(wrt, "if ")?;
                    write_ast_expr(wrt, ast, ssa, *cond, ExprFlags::default())?;
                } else {
                    write!(wrt, "if ???")?;
                }
                write!(wrt, "\nthen  ")?;
                wrt.open_box();
                write_ast_node(wrt, ast, ssa, types, *cons)?;
                wrt.close_box();
                write!(wrt, "\nelse  ")?;
                wrt.open_box();
                write_ast_node(wrt, ast, ssa, types, *alt)?;
                wrt.close_box();
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
        }

        // TODO extend to check we never visit a node twice
        assert_ne!(sid, sid_init);
    }
}

#[derive(Default)]
struct ExprFlags {
    always_expand_root: bool,
}

fn write_ast_expr<W: std::io::Write>(
    wrt: &mut W,
    ast: &decompiler::Ast,
    ssa: &decompiler::SSAProgram,
    reg: decompiler::Reg,
    flags: ExprFlags,
) -> std::io::Result<()> {
    if !flags.always_expand_root && ast.is_value_named(reg) {
        write!(wrt, "r{}", reg.0)?;
        return Ok(());
    }

    let insn = ssa.get(reg).unwrap();
    let xpinsn = decompiler::to_expanded(&insn);

    write!(wrt, "{} ", xpinsn.opcode)?;
    for (ndx, (_key, arg)) in xpinsn.fields.into_iter().enumerate() {
        if ndx > 0 {
            write!(wrt, " ")?;
        }
        // write!(wrt, "{}:", key)?;
        match arg {
            decompiler::ExpandedValue::Reg(reg) => {
                if ast.is_value_named(reg) {
                    write!(wrt, "r{}", reg.reg_index())?;
                } else {
                    write!(wrt, "(")?;
                    write_ast_expr(wrt, ast, ssa, reg, ExprFlags::default())?;
                    write!(wrt, ")")?;
                }
            }
            decompiler::ExpandedValue::Generic(repr) => {
                write!(wrt, "{}", repr)?;
            }
        }
    }

    Ok(())
}

fn parse_cli<S: AsRef<str>>(mut opts: impl Iterator<Item = S>) -> anyhow::Result<CliOptions> {
    let mut exe = None;
    let mut object = Object::FunctionAll;
    let mut output_format = OutputFormat::JSON;

    // Use a while loop to manually advance the iterator when consuming values.
    while let Some(arg) = opts.next() {
        match arg.as_ref() {
            "--exe" => {
                let val = opts
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("Missing value for --exe"))?;
                exe = Some(val.as_ref().to_string());
            }
            "--func" => {
                let val = opts
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("Missing value for --func"))?;
                object = Object::Function {
                    name: val.as_ref().to_string(),
                };
            }
            "--types" => {
                object = Object::Types;
            }
            "--format" => {
                let val = opts
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("Missing value for --format"))?;
                output_format = match val.as_ref() {
                    "json" => OutputFormat::JSON,
                    "ssa/text" => OutputFormat::SSA_Text,
                    "ast/text" => OutputFormat::AST_Text,
                    "ast/dot" => OutputFormat::AST_DOT,
                    other => {
                        anyhow::bail!("invalid format name: '{}' (supported: json, text)", other);
                    }
                };
            }

            unknown => anyhow::bail!("Unknown argument: {}", unknown),
        }
    }

    // After parsing all arguments, ensure the mandatory --exe was provided.
    let exe =
        exe.ok_or_else(|| anyhow::anyhow!("Missing required argument --exe. Usage: --exe <PATH>"))?;

    Ok(CliOptions {
        exe,
        object,
        output_format,
    })
}
