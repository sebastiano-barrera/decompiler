//! Tests based on "test_tool"
//!
//! These are pretty weak, but they work on real code. They only check that the
//! decompiler doesn't panic, and they write the decompiler's output as text to
//! a version-controlled file, just so that it's somewhere easy to diff and hard
//! to forget updating.

use decompiler::Insn;
use include_dir::{include_dir, Dir};
use insta::assert_snapshot;
use test_log::test;

use std::path::Path;

static DATA_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/test-data/");

macro_rules! case {
    ($case:ident, $func_name:ident) => {
        #[test_log::test]
        #[allow(non_snake_case)]
        fn $case() {
            let exe = crate::open_exe(PATH);
            let function_name = stringify!($func_name);
            let out = crate::process_function(&exe, function_name);
            crate::assert_snapshot!(out);
        }
    };
}

macro_rules! tests_in_binary {
    ($group:ident, $exe_path:expr; $($funcs:ident),* $(,)?) => {
        mod $group {
            use crate::test_decompile_all_no_panic;
            use test_log::test;

            pub static PATH: &'static str = $exe_path;

            #[test]
            fn no_panic() {
                // for each executable (= invocation of the tests_in_binary! macro)
                // add a `no_panic` test
                let exe = crate::open_exe(PATH);
                test_decompile_all_no_panic(&exe);
            }

            $(case!($funcs, $funcs);)*
        }
    };
}

fn test_decompile_all_no_panic(exe: &decompiler::Executable) {
    use std::cell::RefCell;
    thread_local! {
        static CURRENT_FUNC: RefCell<String> = RefCell::new(String::new());
    }
    let super_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _out_lock = std::io::stdout().lock();
        CURRENT_FUNC.with_borrow(|cur_func_name| {
            println!("---- panicked while decompiling: {}", cur_func_name);
        });
        super_hook(panic_info);
        println!("---- end");
    }));

    // I prefer having a deterministic order so that I don't have to keep track
    // of which function specifically I have to fix nor create specific test
    // cases. I just repeat the test every time, and get the same error every
    // time until it's fixed.
    let mut names: Vec<_> = exe.function_names().collect();
    names.sort();

    // TODO get the actual number with `num_cpus`
    let num_threads = 8;

    std::thread::scope(|s| {
        for thread_ndx in 0..num_threads {
            let names = &names;
            s.spawn(move || {
                for func_name in names.iter().skip(thread_ndx).step_by(num_threads - 1) {
                    CURRENT_FUNC.with_borrow_mut(|cur_func_name| {
                        *cur_func_name = func_name.to_string();
                    });
                    // discard result
                    //
                    // we don't care if we got any errors; we only want to check
                    // that all functions can be decompiled without panicking.
                    //
                    // more specific tests are covered by test cases associated
                    // with specific functions.
                    let _ = exe.decompile_function(func_name);
                }
            });
        }
    });
}

fn process_function(exe: &decompiler::Executable<'_>, function_name: &str) -> String {
    use std::fmt::Write;

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
        let ast = decompiler::AstBuilder::new(&ssa).build();
        let mut pp_buf = decompiler::pp::FmtAsIoUTF8(&mut log_buf);
        let pp = &mut decompiler::pp::PrettyPrinter::start(&mut pp_buf);
        decompiler::write_ast(pp, &ast, ssa, exe.types()).unwrap();

        if ssa.faults().len() > 0 {
            println!(" --- ssa faults [{}]", ssa.faults().len());
            for fault in ssa.faults() {
                println!(" * {:?}", fault);
            }
            panic!("panicking due to {} faults", ssa.faults().len());
        }
    }

    log_buf
}

fn open_exe(s: &str) -> decompiler::Executable<'_> {
    let rel_path = Path::new(s);
    let raw = DATA_DIR.get_file(rel_path).unwrap().contents();
    decompiler::Executable::parse(raw).unwrap()
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

#[test]
fn test_function_type_id_propagation() {
    let exe = open_exe("ty/test_composite_type.so");
    let function_name = "list_len";

    let df = exe
        .decompile_function(function_name)
        .expect("decompiling function");

    let ssa_program = df.ssa().expect("SSA program should exist");

    // Assuming list_len has a known TypeID in the test_composite_type.so
    // This TypeID would typically be retrieved from the Executable's type system
    // For this test, we'll assert it's Some, as the exact ID might vary
    // based on how the test_composite_type.so is generated and its types are registered.
    assert!(
        ssa_program.function_type_id().is_some(),
        "Function TypeID should be propagated to SSA program"
    );
}

#[test]
fn test_named_callee() {
    let exe = crate::open_exe(&redis_server::PATH);
    let func = exe.decompile_function("createSortOperation").unwrap();
    let ssa = func.ssa().unwrap();

    let mut s = String::new();
    ssa.dump(&mut s, Some(exe.types())).unwrap();
    println!("{}", s);

    // find the Insn::Call, then check that its callee is a Insn::Global
    let callee = ssa
        .registers()
        .find_map(|reg| {
            if let Insn::Call { callee, .. } = ssa.get(reg).unwrap() {
                Some(callee)
            } else {
                None
            }
        })
        .unwrap();
    let callee_insn = ssa.get(callee).unwrap();
    assert_eq!(Insn::Global("zmalloc"), callee_insn);
}
