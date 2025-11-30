use std::fs::File;

fn main() {
    let executable_path = std::env::args()
        .nth(1)
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

    let exe = decompiler::Executable::parse(&contents).expect("could not parse executable");
    println!(" --- types");

    let mut stdout = std::io::stdout();
    let mut pp = decompiler::pp::PrettyPrinter::start(&mut stdout);
    let types = exe.types().read_tx().unwrap();
    types.read().dump(&mut pp).unwrap();
}
