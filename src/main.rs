use std::io::Read;
use std::{fs::File, path::PathBuf};

use iced_x86::{Decoder, Formatter, IntelFormatter};

mod cfg;
mod mil;
mod ssa;
mod x86_to_mil;

struct CliOptions {
    elf_filename: PathBuf,
    function_name: String,
}

impl CliOptions {
    fn parse<S: AsRef<str>>(mut opts: impl Iterator<Item = S>) -> Option<Self> {
        let elf_filename = opts.next()?;
        let elf_filename = PathBuf::from(elf_filename.as_ref());

        let function_name = opts.next()?.as_ref().to_owned();

        Some(CliOptions {
            elf_filename,
            function_name,
        })
    }
}

fn main() {
    let mut args = std::env::args();
    let program_name = args.next().unwrap();
    let opts = match CliOptions::parse(args) {
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
        let mut elf = File::open(opts.elf_filename).expect("could not open executable");
        elf.read_to_end(&mut contents).expect("read error");
        contents
    };

    let object = goblin::Object::parse(&contents).expect("elf parse error");
    let elf = match object {
        goblin::Object::Elf(elf) => elf,
        _ => {
            eprintln!("unsupported executable format: {:?}", object);
            return;
        }
    };

    let func_sym = elf
        .syms
        .iter()
        .find(|sym| elf.strtab.get_at(sym.st_name) == Some(&opts.function_name))
        .expect("symbol not found");

    if !func_sym.is_function() {
        eprintln!("symbol `{}` is not a function", opts.function_name);
        return;
    }

    let func_addr = func_sym.st_value as usize;
    let func_size = func_sym.st_size as usize;
    let func_end = func_addr + func_size;

    let text_section = elf
        .section_headers
        .iter()
        .find(|sec| sec.is_executable() && elf.shdr_strtab.get_at(sec.sh_name) == Some(".text"))
        .expect("no .text section?!");

    let vm_range = text_section.vm_range();
    if vm_range.start > func_addr || vm_range.end < func_end {
        eprintln!(
            "function memory range (0x{:x}-0x{:x}) out of .text section vm range (0x{:x}-0x{:x})",
            func_addr, func_end, vm_range.start, vm_range.end
        );
    }

    // function's offset into the file
    let func_section_ofs = func_addr - vm_range.start;
    let func_fofs = text_section.sh_offset as usize + func_section_ofs as usize;
    let func_text = &contents[func_fofs..func_fofs + func_size];
    println!(
        "{} 0x{:x}+{} (file 0x{:x})",
        opts.function_name, func_addr, func_size, func_fofs,
    );

    let decoder = Decoder::with_ip(
        64,
        func_text,
        func_addr.try_into().unwrap(),
        iced_x86::DecoderOptions::NONE,
    );
    let mut formatter = IntelFormatter::new();
    let mut instr_strbuf = String::new();
    for instr in decoder {
        print!("{:16x}: ", instr.ip());
        let ofs = instr.ip() as usize - func_addr;
        let len = instr.len();
        for i in 0..8 {
            if i < len {
                print!("{:02x} ", func_text[ofs + i]);
            } else {
                print!("   ");
            }
        }

        instr_strbuf.clear();
        formatter.format(&instr, &mut instr_strbuf);
        println!("{}", instr_strbuf);
    }

    println!();
    let mut decoder = Decoder::with_ip(
        64,
        func_text,
        func_addr.try_into().unwrap(),
        iced_x86::DecoderOptions::NONE,
    );
    let prog = x86_to_mil::translate(decoder.iter()).unwrap();
    println!("mil program = ");
    println!("{:?}", prog);

    println!();
    let prog = ssa::mil_to_ssa(prog);
    println!("{:?}", prog);
}
