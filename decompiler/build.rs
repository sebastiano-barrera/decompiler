use std::process::Command;

fn main() {
    const SCRIPT_GEN_CALLCONV_TESTS: &str = "scripts/gen_callconv_tests.rb";

    println!("cargo::rerun-if-changed={}", SCRIPT_GEN_CALLCONV_TESTS);
    Command::new(SCRIPT_GEN_CALLCONV_TESTS)
        .arg("--out-c")
        .arg("test-data/x86_64_callconv.c")
        .arg("--out-rs")
        .arg("tests/generated_callconv.rs")
        .status()
        .unwrap();

    // generate the test executable
    //
    // don't use the `cc` crate because we really want to use GCC and this
    // specific set of flags
    Command::new("gcc")
        // higher levels of optimizations completely hide the calling
        // conventions
        .arg("-O1")
        .arg("-gdwarf")
        .arg("-o")
        .arg("test-data/x86_64_callconv")
        .arg("test-data/x86_64_callconv.c")
        .status()
        .unwrap();
}
