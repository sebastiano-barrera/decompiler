use std::{collections::HashMap, fs::File, io::Read};

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
                let doc = df_to_doc(&df);
                serde_json::to_writer_pretty(stdout, &doc)?;
            }
            OutputFormat::Text => {
                let doc = df_to_doc(&df);
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

#[derive(serde::Serialize)]
struct Document {
    mil: Option<MIL>,
    ssa: Option<SSA>,
}

#[derive(serde::Serialize)]
struct MIL {
    body: Vec<Insn>,
}

#[derive(serde::Serialize)]
struct SSA {
    blocks: HashMap<decompiler::BlockID, SSABlock>,
}

#[derive(serde::Serialize)]
struct SSABlock {
    body: Vec<Insn>,
    cont: decompiler::BlockCont,
}

#[derive(serde::Serialize)]
struct Insn {
    addr: Option<u64>,
    dest: u16,
    insn: decompiler::Insn,
    tyid: Option<decompiler::ty::TypeID>,
    reg_type: Option<decompiler::RegType>,
}

fn df_to_doc(df: &decompiler::DecompiledFunction) -> Document {
    let mil = 'mil: {
        let Some(mil) = df.mil() else {
            break 'mil None;
        };

        let mut body = Vec::new();
        for ndx in 0..mil.len() {
            let iv = mil.get(ndx).unwrap();
            body.push(Insn {
                addr: Some(iv.addr),
                dest: iv.dest.get().reg_index(),
                insn: iv.insn.get(),
                tyid: None,
                reg_type: None,
            });
        }

        Some(MIL { body })
    };

    let ssa = 'ssa: {
        let Some(ssa) = df.ssa() else { break 'ssa None };

        let mut body_of_block = HashMap::new();

        for bid in ssa.cfg().block_ids() {
            let cont = ssa.cfg().block_cont(bid);

            let mut body = Vec::new();
            for reg in ssa.block_regs(bid) {
                body.push(Insn {
                    addr: None,
                    dest: reg.reg_index(),
                    insn: ssa.get(reg).unwrap(),
                    tyid: ssa.value_type(reg),
                    reg_type: Some(ssa.reg_type(reg)),
                });
            }

            body_of_block.insert(bid, SSABlock { body, cont });
        }

        Some(SSA {
            blocks: body_of_block,
        })
    };

    Document { mil, ssa }
}

fn write_text<W: std::io::Write>(doc: &Document, mut wrt: W) -> std::io::Result<()> {
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
    insn: &Insn,
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
    write!(wrt, " <- {:?}", insn.insn)?;
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
