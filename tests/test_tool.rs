use decompiler::{pp::PrettyPrinter, test_tool};

use include_dir::{include_dir, Dir};
use insta::assert_snapshot;
use std::path::Path;

static DATA_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/test-data/");

fn test_with_sample(rel_path: &Path, function_name: &str) -> test_tool::Result<String> {
    let raw = DATA_DIR.get_file(rel_path).unwrap().contents();

    // let mut buf = Vec::with_capacity(8192);
    let mut buf = std::io::stdout();
    let mut pp = PrettyPrinter::start(&mut buf);
    test_tool::run(raw, function_name, &mut pp)?;

    // let strbuf = String::from_utf8(buf).unwrap();
    // Ok(strbuf)
    Ok(todo!())
}

#[test]
fn test_with_sample_composite_type_list_len() {
    let out = test_with_sample(Path::new("ty/test_composite_type.so"), "list_len").unwrap();
    assert_snapshot!(out);
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

#[test]
fn test_with_redis_server_ctl_lookup() {
    let out = test_with_sample(Path::new("integration/redis-server"), "ctl_lookup").unwrap();
    assert_snapshot!(out);
}
