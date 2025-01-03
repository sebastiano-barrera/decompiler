use decompiler::{pp, test_tool};
use rayon::prelude::*;
use std::{fs::File, io::Read};

fn main() {
    let exec_path = {
        let mut args = std::env::args();
        let program_name = args.next().unwrap();
        match args.next() {
            Some(opts) => opts,
            None => {
                eprintln!("usage: {} EXEC", program_name);
                eprintln!("      EXEC = path to the executable (only ELF is supported)");
                return;
            }
        }
    };

    let raw_binary = {
        let mut contents = Vec::new();
        let mut elf = File::open(&exec_path).expect("could not open executable");
        elf.read_to_end(&mut contents).expect("read error");
        contents
    };

    let function_names: Vec<_> = {
        let object = goblin::Object::parse(&raw_binary).expect("elf parse error");
        let elf = match object {
            goblin::Object::Elf(elf) => elf,
            _ => panic!("unsuppored binary format: {:?}", object),
        };

        elf.syms
            .iter()
            .filter(|sym| sym.is_function() && !sym.is_import())
            .map(|sym| elf.strtab.get_at(sym.st_name).unwrap().to_owned())
            .collect()
    };

    println!("parsing {} functions:", function_names.len());
    for name in &function_names {
        println!(" - {}", name);
    }

    for function_name in function_names {
        let mut out = pp::PrettyPrinter::start(NullFmtSink);
        let _ = test_tool::run(&raw_binary, &function_name, &mut out);
    }
}

struct NullFmtSink;

impl std::fmt::Write for NullFmtSink {
    fn write_str(&mut self, _: &str) -> std::fmt::Result {
        Ok(())
    }
}
