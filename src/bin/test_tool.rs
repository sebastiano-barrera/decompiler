use std::{fs::File, io::Read, path::PathBuf};

use decompiler::test_tool;

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

    // TODO Replace with memory mapping? (but it requires locking, see memmap2's docs)
    // https://docs.rs/memmap2/0.9.5/memmap2/struct.Mmap.html#safety
    let contents = {
        let mut contents = Vec::new();
        let mut elf = File::open(&opts.elf_filename).expect("could not open executable");
        elf.read_to_end(&mut contents).expect("read error");
        contents
    };

    let out = std::io::stdout().lock();
    let res = test_tool::run(&contents, &opts.function_name, &mut IoAsFmt(out));
    if let Err(err) = res {
        eprintln!("error: {}", err);
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

struct IoAsFmt<W>(W);

impl<W: std::io::Write> std::fmt::Write for IoAsFmt<W> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.write_all(s.as_bytes()).unwrap();
        Ok(())
    }
}
