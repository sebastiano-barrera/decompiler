use decompiler::api::proto;
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
enum OutputFormat {
    JSON,
    Text,
    DOT,
}

fn main() {
    tracing_subscriber::fmt::fmt()
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
            eprintln!("    FORMAT = format for the output (supported: json (default), csv, dot)");
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
                let doc = proto::Document::from(&df);
                serde_json::to_writer_pretty(stdout, &doc)?;
            }
            OutputFormat::Text => {
                let doc = {
                    let df: &decompiler::DecompiledFunction = &df;
                    df.into()
                };
                write_text(&doc, stdout)?;
            }
            OutputFormat::DOT => {
                match df.ssa() {
                    Some(ssa) => {
                        write_cfg_dot(ssa.cfg(), stdout)?;
                    }
                    None => {
                        anyhow::bail!("could not produce ssa/cfg");
                    }
                }
                // write_cfg_dot(doc.ssa.),
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

fn write_text<W: std::io::Write>(doc: &proto::Document, mut wrt: W) -> std::io::Result<()> {
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
                    "text" => OutputFormat::Text,
                    "dot" => OutputFormat::DOT,
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
