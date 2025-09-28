//! Tests based on "test_tool"
//!
//! These are pretty weak, but they work on real code. They only check that the
//! decompiler doesn't panic, and they write the decompiler's output as text to
//! a version-controlled file, just so that it's somewhere easy to diff and hard
//! to forget updating.

use include_dir::{include_dir, Dir};
use insta::assert_snapshot;

use std::{
    path::Path,
    sync::{Mutex, MutexGuard, OnceLock},
};

static DATA_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/test-data/");

macro_rules! case {
    ($case:ident, $exe:expr, $func_name:ident) => {
        #[test_log::test]
        #[allow(non_snake_case)]
        fn $case() {
            let function_name = stringify!($func_name);
            let out = $exe.process_function(function_name);
            crate::assert_snapshot!(out);
        }
    };
}

macro_rules! tests_in_binary {
    ($group:ident, $exe_path:expr; $($funcs:ident),* $(,)?) => {
        mod $group {
            use crate::{Exe, test_decompile_all_no_panic};
            use test_log::test;
            static EXE: Exe = Exe::new($exe_path);

            #[test]
            fn no_panic() {
                let exe = EXE.get_or_init();
                test_decompile_all_no_panic(&*exe);
            }

            $(case!($funcs, EXE, $funcs);)*
        }
    };
}

fn test_decompile_all_no_panic(exe: &decompiler::Executable) {
    // I prefer having a deterministic order so that I don't have to keep track
    // of which function specifically I have to fix nor create specific test
    // cases. I just repeat the test every time, and get the same error every
    // time until it's fixed.
    let mut names: Vec<_> = exe.function_names().collect();
    names.sort();

    // TODO parallelize
    for func_name in names {
        // discard result
        //
        // we don't care if we got any errors; we only want to check that all
        // functions can be decompiled without panicking.
        //
        // more specific tests are covered by test cases associated with
        // specific functions.
        let _ = exe.decompile_function(func_name);
    }
}

struct Exe {
    once_lock: OnceLock<Mutex<decompiler::Executable<'static>>>,
    path: &'static str,
}
impl Exe {
    const fn new(path: &'static str) -> Self {
        Exe {
            once_lock: OnceLock::new(),
            path,
        }
    }

    fn get_or_init(&self) -> MutexGuard<'_, decompiler::Executable<'static>> {
        let mutex = self.once_lock.get_or_init(|| {
            let rel_path = Path::new(self.path);
            let raw = DATA_DIR.get_file(rel_path).unwrap().contents();
            let exe = decompiler::Executable::parse(raw).unwrap();
            Mutex::new(exe)
        });
        mutex.lock().unwrap()
    }

    fn process_function(&self, function_name: &str) -> String {
        use std::fmt::Write;

        let exe = self.get_or_init();

        let df = exe
            .decompile_function(function_name)
            .expect("decompiling function");

        let mut log_buf = String::new();

        if let Some(mil) = df.mil() {
            writeln!(log_buf, " --- mil").unwrap();
            writeln!(log_buf, "{:?}\n", mil).unwrap();
        }

        if let Some(ssa) = df.ssa_pre_xform() {
            writeln!(log_buf, " --- ssa pre-xform").unwrap();
            writeln!(log_buf, "{:?}\n", ssa).unwrap();
        }

        if let Some(ssa) = df.ssa() {
            writeln!(log_buf, " --- cfg").unwrap();
            let cfg = ssa.cfg();
            writeln!(log_buf, "  entry: {:?}", cfg.direct().entry_bid()).unwrap();
            for bid in cfg.block_ids() {
                let regs: Vec<_> = ssa.block_regs(bid).collect();
                writeln!(
                    log_buf,
                    "  {:?} -> {:?} {:?}",
                    bid,
                    cfg.block_cont(bid),
                    regs
                )
                .unwrap();
            }
            write!(log_buf, "  domtree:\n    ").unwrap();

            let mut pp_buf = decompiler::pp::FmtAsIoUTF8(&mut log_buf);
            let pp = &mut decompiler::pp::PrettyPrinter::start(&mut pp_buf);
            cfg.dom_tree().dump(pp).unwrap();
            writeln!(log_buf).unwrap();

            writeln!(log_buf, " --- ssa").unwrap();
            writeln!(log_buf, "{:?}\n", ssa).unwrap();

            writeln!(log_buf, " --- ast").unwrap();
            let mut ast = decompiler::Ast::new(&ssa, exe.types());
            let mut pp_buf = decompiler::pp::FmtAsIoUTF8(&mut log_buf);
            let pp = &mut decompiler::pp::PrettyPrinter::start(&mut pp_buf);
            ast.pretty_print(pp).unwrap();
        }

        log_buf
    }
}

tests_in_binary!(
    exe_composite_type, "ty/test_composite_type.so";
    list_len
);
tests_in_binary!(exe_capstone, "integration/sample_capstone"; main);
tests_in_binary!(exe_matrix, "integration/sample_matrix"; sum_matrix);
tests_in_binary!(
    redis_server, "integration/redis-server";
    ctl_lookup,
    listNext,
    listTypePush,
    geoArrayCreate,
    // regression: infinite loop
    processAnnotations,
    redisReconnect,
    redisConnectWithOptions,
);
