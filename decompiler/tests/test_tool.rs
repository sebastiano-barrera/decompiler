//! Tests based on "test_tool"
//!
//! These are pretty weak, but they work on real code. They only check that the
//! decompiler doesn't panic, and they write the decompiler's output as text to
//! a version-controlled file, just so that it's somewhere easy to diff and hard
//! to forget updating.

use include_dir::{include_dir, Dir};
use insta::assert_snapshot;

use std::{path::Path, sync::OnceLock};

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
            use crate::Exe;
            static EXE: Exe = Exe::new($exe_path);

            $(case!($funcs, EXE, $funcs);)*
        }
    };
}

struct Exe {
    once_lock: OnceLock<decompiler::Result<decompiler::Executable<'static>>>,
    path: &'static str,
}
impl Exe {
    const fn new(path: &'static str) -> Self {
        Exe {
            once_lock: OnceLock::new(),
            path,
        }
    }

    fn get_or_init(&self) -> &decompiler::Executable<'static> {
        self.once_lock
            .get_or_init(|| {
                let rel_path = Path::new(self.path);
                let raw = DATA_DIR.get_file(rel_path).unwrap().contents();
                decompiler::Executable::parse(raw)
            })
            .as_ref()
            .unwrap()
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
            let mut ast = decompiler::Ast::new(&ssa);
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

tests_in_binary!(
    x86_64_callconv, "x86_64_callconv";
    func000,
    func001,
    func002,
    func003,
    func004,
    func005,
    func006,
    func007,
    func008,
    func009,
    func010,
    func011,
    func012,
    func013,
    func014,
    func015,
    func016,
    func017,
    func018,
    func019,
    func020,
    func021,
    func022,
    func023,
    func024,
    func025,
    func026,
    func027,
    func028,
    func029,
    func030,
    func031,
    func032,
    func033,
    func034,
    func035,
    func036,
    func037,
    func038,
    func039,
    func040,
    func041,
    func042,
    func043,
    func044,
    func045,
    func046,
    func047,
    func048,
    func049,
    func050,
    func051,
    func052,
    func053,
    func054,
    func055,
    func056,
    func057,
    func058,
    func059,
    func060,
    func061,
    func062,
    func063,
    func064,
    func065,
    func066,
    func067,
    func068,
    func069,
    func070,
    func071,
    func072,
    func073,
    func074,
    func075,
    func076,
    func077,
    func078,
    func079,
    func080,
    func081,
    func082,
    func083,
    func084,
    func085,
    func086,
    func087,
    func088,
    func089,
    func090,
    func091,
    func092,
    func093,
    func094,
    func095,
    func096,
    func097,
    func098,
    func099,
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
    func132,
    func133,
    func134,
    func135,
    func136,
    func137,
    func138,
    func139,
    func140,
    func141,
    func142,
    func143,
    func144,
    func145,
    func146,
    func147,
    func148,
    func149,
    func150,
    func151,
    func152,
    func153
);
