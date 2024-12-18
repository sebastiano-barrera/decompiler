use std::{any::Any, fs::File};

use gimli::{EndianSlice, Reader};

fn main() {
    let executable_path = std::env::args()
        .skip(1)
        .next()
        .expect("usage: lab_debug_info <executable>");

    let contents = {
        use std::io::Read;
        let mut contents = Vec::new();
        File::open(&executable_path)
            .expect("could not open executable")
            .read_to_end(&mut contents)
            .expect("read error");
        contents
    };
    let object = goblin::Object::parse(&contents).expect("could not parse ELF");
    let elf = match object {
        goblin::Object::Elf(elf) => elf,
        _ => {
            eprintln!("unsupported executable format: {:?}", object);
            return;
        }
    };

    let dwarf = gimli::Dwarf::load(|section| -> Result<_, ()> {
        let bytes = elf
            .section_headers
            .iter()
            .find(|sec| elf.shdr_strtab.get_at(sec.sh_name) == Some(section.name()))
            .and_then(|sec_hdr| {
                let file_range = sec_hdr.file_range()?;
                Some(&contents[file_range])
            })
            .unwrap_or(&[]);

        let endianity = if elf.little_endian {
            gimli::RunTimeEndian::Little
        } else {
            gimli::RunTimeEndian::Big
        };

        Ok(EndianSlice::new(bytes, endianity))
    })
    .expect("loading DWARF");

    // "unit" means "compilation unit" here
    {
        let mut units = dwarf.debug_info.units();
        let mut count = 0;
        while let Some(unit_hdr) = units.next().unwrap() {
            let unit = dwarf.unit(unit_hdr).unwrap();
            println!(
                "compilation unit @ {:?} -- {:?}",
                unit.header.offset(),
                unit.header.format()
            );

            let mut entries = unit.entries();
            let mut ind_lvl = 0;
            while let Some((ind_diff, entry)) = entries.next_dfs().unwrap() {
                ind_lvl += ind_diff;
                let tag = entry.tag().static_string().unwrap_or("???");
                let name = entry
                    .attr(gimli::constants::DW_AT_name)
                    .unwrap()
                    .map(|attr| dwarf.attr_string(&unit, attr.value()).unwrap().slice())
                    .unwrap_or(b"???");
                let name = std::str::from_utf8(name).unwrap_or("<not utf8>");
                indent(ind_lvl);
                println!("@ {:?} entry {} {:30}", entry.offset().0, tag, name);

                let mut attrs = entry.attrs();
                while let Some(attr) = attrs.next().unwrap() {
                    indent(ind_lvl);
                    println!(
                        "attr {:20} {:?}",
                        attr.name().static_string().unwrap_or("???"),
                        attr.value()
                    );
                }
            }

            count += 1;
        }
        println!("{} compilation units", count);
    }
}

fn indent(count: isize) {
    for _ in 0..count {
        print!("|   ");
    }
}
