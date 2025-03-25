use decompiler::{pp::PrettyPrinter, test_tool};

use include_dir::{include_dir, Dir};
use insta::assert_snapshot;
use std::path::Path;

static DATA_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/test-data/");

fn test_with_sample(rel_path: &Path, function_name: &str) -> test_tool::Result<String> {
    let raw = DATA_DIR.get_file(rel_path).unwrap().contents();

    let mut buf = Vec::new();
    let mut pp = PrettyPrinter::start(&mut buf);
    test_tool::run(raw, function_name, &mut pp)?;

    let strbuf = String::from_utf8(buf).unwrap();
    Ok(strbuf)
}

macro_rules! case {
    ($case:ident, $exe_path:expr, $func_name:ident) => {
        #[test]
        #[allow(non_snake_case)]
        fn $case() {
            let out =
                $crate::test_with_sample(std::path::Path::new($exe_path), stringify!($func_name))
                    .unwrap();
            $crate::assert_snapshot!(out);
        }
    };
}

macro_rules! tests_in_binary {
    ($group:ident, $exe_path:expr; $($funcs:ident),*) => {
        mod $group {
            $(case!($funcs, $exe_path, $funcs);)*
        }
    };
}

tests_in_binary!(
    exe_composite_type, "ty/test_composite_type.so";
    list_len
);
tests_in_binary!(exe_capstone, "integration/sample_capstone"; main);
tests_in_binary!(exe_matrix, "integration/sample_matrix"; sum_matrix);

tests_in_binary!(redis_server, "integration/redis-server";
    ctl_lookup,
    listNext,
    listTypePush
);
