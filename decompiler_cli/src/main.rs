use std::{fs::File, io::Read, path::PathBuf};

use anyhow::anyhow;
use iced_x86::Formatter;

pub struct CliOptions {
    elf_filename: PathBuf,
    function_name: Option<String>,
    quiet: bool,
}

fn main() {
    tracing_subscriber::fmt::init();

    let mut args = std::env::args();
    let program_name = args.next().unwrap();
    let opts = match parse_cli(args) {
        Ok(opts) => opts,
        Err(err) => {
            eprintln!("error: {}", err);
            eprintln!(
                "usage: {} --exe EXE [--func FUNCTION] [--quiet]",
                program_name
            );
            eprintln!("       EXE = path to the executable (only ELF is supported)");
            eprintln!("  FUNCTION = name of the function to analyze (e.g. 'main')");
            eprintln!("             (If absent, all functions are decompiled in parallel)");
            eprintln!("   --quiet = don't print output (useful for development");
            eprintln!("             with env var RUST_LOG=trace)");
            return;
        }
    };

    // TODO Replace with memory mapping? (but it requires locking, see memmap2's docs)
    // https://docs.rs/memmap2/0.9.5/memmap2/struct.Mmap.html#safety
    let contents = {
        let mut contents = Vec::new();
        let mut elf = File::open(&opts.elf_filename).expect("could not open executable");
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

    let function_names: Vec<_> = match opts.function_name {
        Some(name) => vec![name],
        None => exe.function_names().map(|s| s.to_string()).collect(),
    };

    function_names
        // TODO ready to change to .into_par_iter()
        // (but first I have to pass large-scale testing)
        .into_iter()
        .enumerate()
        .for_each(|(ndx, function_name)| {
            eprintln!("------ function #{}: {}", ndx + 1, function_name);
            let res = std::panic::catch_unwind(|| {
                let df = exe
                    .decompile_function(&function_name)
                    .expect("decompiling function");

                if !opts.quiet {
                    let mut stdout = std::io::stdout().lock();

                    println!(" --- asm");
                    let decoder = df.disassemble(&exe);
                    dump_assembly(decoder, df.machine_code(&exe));
                    println!();

                    if let Some(mil) = df.mil() {
                        println!(" --- mil");
                        println!("{:?}", mil);
                        println!();
                    }

                    if let Some(ssa) = df.ssa_pre_xform() {
                        let mut buf = String::new();
                        ssa.dump(&mut buf, Some(exe.types())).unwrap();
                        println!(" --- ssa pre-xform");
                        println!("{buf}");
                    }

                    if let Some(ssa) = df.ssa() {
                        println!(" --- cfg");
                        let cfg = ssa.cfg();
                        println!("  entry: {:?}", cfg.direct().entry_bid());
                        for bid in cfg.block_ids() {
                            let regs: Vec<_> = ssa.block_regs(bid).collect();
                            println!("  {:?} -> {:?} {:?}", bid, cfg.block_cont(bid), regs);
                        }
                        print!("  domtree:\n    ");

                        let pp = &mut decompiler::pp::PrettyPrinter::start(&mut stdout);
                        cfg.dom_tree().dump(pp).unwrap();
                        println!();

                        println!(" --- ssa");
                        let mut buf = String::new();
                        ssa.dump(&mut buf, Some(exe.types())).unwrap();
                        println!("{buf}");
                        println!();

                        println!(" --- ast");
                        let mut ast = decompiler::Ast::new(&ssa, exe.types());
                        let pp = &mut decompiler::pp::PrettyPrinter::start(&mut stdout);
                        ast.pretty_print(pp).unwrap();
                    }
                }
            });

            if let Err(err) = res {
                eprintln!("PANIC: while decompiling [{}]: {:?}", function_name, err);
            }
        });
}

fn dump_assembly(decoder: iced_x86::Decoder<'_>, func_text: &[u8]) {
    let mut formatter = iced_x86::IntelFormatter::new();
    let mut instr_strbuf = String::new();
    let ip_start = decoder.ip();
    for instr in decoder {
        print!("{:16x}: ", instr.ip());
        let ofs = instr.ip() - ip_start;
        let len = instr.len();

        for i in 0..8 {
            let byte_ndx = ofs as usize + i;
            if i < len {
                print!("{:02x} ", func_text[byte_ndx]);
            } else {
                print!("   ");
            }
        }

        instr_strbuf.clear();
        formatter.format(&instr, &mut instr_strbuf);
        println!("{}", instr_strbuf);
    }
}

fn parse_cli<S: AsRef<str>>(mut opts: impl Iterator<Item = S>) -> anyhow::Result<CliOptions> {
    let mut elf_filename = None;
    let mut function_name = None;
    let mut quiet = false;

    while let Some(opt) = opts.next() {
        match opt.as_ref() {
            "--exe" => {
                let path = opts
                    .next()
                    .ok_or_else(|| anyhow!("argument required for --exe"))?;
                elf_filename = Some(PathBuf::from(path.as_ref()));
            }
            "--func" => {
                let path = opts
                    .next()
                    .ok_or_else(|| anyhow!("argument required for --func"))?;
                function_name = Some(path.as_ref().to_string());
            }
            "--quiet" => {
                quiet = true;
            }
            other => return Err(anyhow!("unrecognized argument: {}", other)),
        }
    }

    let elf_filename = elf_filename.ok_or_else(|| anyhow!("--exe is required"))?;

    Ok(CliOptions {
        elf_filename,
        function_name,
        quiet,
    })
}
