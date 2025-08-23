use std::{fs::File, io::Read, path::PathBuf};

use decompiler::pp;
use iced_x86::Formatter;

pub struct CliOptions {
    pub elf_filename: PathBuf,
    pub function_name: String,
}

fn main() {
    let mut args = std::env::args();
    let program_name = args.next().unwrap();
    let opts = match parse_cli(args) {
        Some(opts) => opts,
        None => {
            eprintln!("usage: {} EXEC FUNCTION", program_name);
            eprintln!("      EXEC = path to the executable (only ELF is supported)");
            eprintln!("  FUNCTION = name of the function to analyze (e.g. 'main')");
            return;
        }
    };
    let function_name = &opts.function_name;

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

    let df = exe
        .decompile_function(function_name)
        .expect("decompiling function");

    print!(" --- asm");
    let decoder = df.disassemble(&exe);
    dump_assembly(decoder, df.machine_code(&exe));
    println!();

    if let Some(mil) = df.mil() {
        println!(" --- mil");
        println!("{:?}", mil);
        println!();
    }

    if let Some(ssa) = df.ssa_pre_xform() {
        println!(" --- ssa pre-xform");
        println!("{:?}", ssa);
        println!();
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

        let mut out = std::io::stdout().lock();
        let pp = &mut decompiler::pp::PrettyPrinter::start(&mut out);
        cfg.dom_tree().dump(pp).unwrap();
        println!();

        println!(" --- ssa");
        println!("{:?}", ssa);
        println!();

        println!(" --- ast");
        let mut ast = decompiler::Ast::new(&ssa);
        let mut out = std::io::stdout().lock();
        let pp = &mut decompiler::pp::PrettyPrinter::start(&mut out);
        ast.pretty_print(pp).unwrap();
    }
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

fn parse_cli<S: AsRef<str>>(mut opts: impl Iterator<Item = S>) -> Option<CliOptions> {
    let elf_filename = opts.next()?;
    let elf_filename = PathBuf::from(elf_filename.as_ref());

    let function_name = opts.next()?.as_ref().to_owned();

    Some(CliOptions {
        elf_filename,
        function_name,
    })
}
