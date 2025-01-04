use decompiler::{
    pp::{self, IoAsFmt},
    test_tool,
};

// use rayon::prelude::*;
use std::{borrow::Cow, io::Write};
use std::{fmt::Debug, fs::File, io::Read, path::PathBuf};

struct Options {
    exec_path: PathBuf,
    out_dir: PathBuf,
}

fn main() {
    let opts = {
        let mut args = std::env::args();
        let program_name = args.next().unwrap();
        let exec_path = args.next();
        let out_dir = args.next();

        let (Some(exec_path), Some(out_dir)) = (exec_path, out_dir) else {
            eprintln!("usage: {} EXEC", program_name);
            eprintln!("      EXEC = path to the executable (only ELF is supported)");
            return;
        };

        Options {
            exec_path: exec_path.into(),
            out_dir: out_dir.into(),
        }
    };

    let raw_binary = {
        let mut contents = Vec::new();
        let mut elf = File::open(&opts.exec_path).expect("could not open executable");
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
        let filename = function_name.replace(|ch: char| !ch.is_alphanumeric(), "_");
        let path = opts.out_dir.join(filename);

        let res = std::panic::catch_unwind(|| {
            let out_file = File::create(&path).unwrap();
            let mut out = pp::PrettyPrinter::start(IoAsFmt(out_file));
            let _ = test_tool::run(&raw_binary, &function_name, &mut out);
        });

        if let Err(err) = res {
            let err = match err.downcast::<String>() {
                Ok(s) => Cow::Owned(*s),
                Err(err) => match err.downcast::<&str>() {
                    Ok(s) => Cow::Borrowed(*s),
                    Err(_) => Cow::Borrowed("<panic value is not a string>"),
                },
            };

            let mut out_file = File::options().append(true).open(path).unwrap();
            writeln!(out_file, "---- error:").unwrap();
            writeln!(out_file, "{}", err).unwrap();
        }
    }
}
