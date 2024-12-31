use decompiler::{pp::PrettyPrinter, test_tool};

use include_dir::{include_dir, Dir};
use insta::assert_snapshot;
use std::path::Path;

static DATA_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/test-data/");

fn test_with_sample(rel_path: &Path, function_name: &str) -> test_tool::Result<String> {
    let raw = DATA_DIR.get_file(rel_path).unwrap().contents();

    let mut buf = String::with_capacity(8192);
    let mut pp = PrettyPrinter::start(&mut buf);
    test_tool::run(raw, function_name, &mut pp)?;

    Ok(buf)
}

#[test]
fn test_with_sample_capstone() {
    let out = test_with_sample(Path::new("integration/sample_capstone"), "main").unwrap();
    assert_snapshot!(out);
}

#[test]
fn test_with_sample_matrix() {
    let out = test_with_sample(Path::new("integration/sample_matrix"), "sum_matrix").unwrap();
    assert_snapshot!(out);
}

#[ignore]
#[test]
fn test_with_redis_server() {
    let out = test_with_sample(Path::new("integration/redis-server"), "main").unwrap();
    assert_snapshot!(out);
}
