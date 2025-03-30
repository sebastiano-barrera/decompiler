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

tests_in_binary!(
    x86_64_callconv, "x86_64_callconv.o";
    func0,
    func1,
    func2,
    func3,
    func4,
    func5,
    func6,
    func7,
    func8,
    func9,
    func10,
    func11,
    func12,
    func13,
    func14,
    func15,
    func16,
    func17,
    func18,
    func19,
    func20,
    func21,
    func22,
    func23,
    func24,
    func25,
    func26,
    func27,
    func28,
    func29,
    func30,
    func31,
    func32,
    func33,
    func34,
    func35,
    func36,
    func37,
    func38,
    func39,
    func40,
    func41,
    func42,
    func43,
    func44,
    func45,
    func46,
    func47,
    func48,
    func49,
    func50,
    func51,
    func52,
    func53,
    func54,
    func55,
    func56,
    func57,
    func58,
    func59,
    func60,
    func61,
    func62,
    func63,
    func64,
    func65,
    func66,
    func67,
    func68,
    func69,
    func70,
    func71,
    func72,
    func73,
    func74,
    func75,
    func76,
    func77,
    func78,
    func79,
    func80,
    func81,
    func82,
    func83,
    func84,
    func85,
    func86,
    func87,
    func88,
    func89,
    func90,
    func91,
    func92,
    func93,
    func94,
    func95,
    func96,
    func97,
    func98,
    func99,
    func100,
    func101,
    func102,
    func103,
    func104,
    func105,
    func106,
    func107,
    func108,
    func109,
    func110,
    func111,
    func112,
    func113,
    func114,
    func115,
    func116,
    func117,
    func118,
    func119,
    func120,
    func121,
    func122,
    func123,
    func124,
    func125,
    func126,
    func127,
    func128,
    func129,
    func130,
    func131,
    func132
);
