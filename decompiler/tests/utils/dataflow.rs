use std::{
    collections::HashMap,
    ops::DerefMut,
    sync::{Mutex, MutexGuard, OnceLock},
};

use include_dir::{include_dir, Dir};

static DATA_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/test-data/");

/// The executable that contains a bunch of compiled test functions, one per test case.
///
/// Wrapped in OnceLock<Mutex<_>> so that a panic in one test case doesn't affect other
/// test cases (after the Mutex is poisoned as a result of the first panic, this is
/// detected by the subsequent test case and decompiled from scratch to restore the data).
static EXE: OnceLock<Mutex<decompiler::Executable<'static>>> = OnceLock::new();

fn get_exe() -> MutexGuard<'static, decompiler::Executable<'static>> {
    let exe_mutex = EXE.get_or_init(|| Mutex::new(make_exe()));
    let res = exe_mutex.lock();
    match res {
        Ok(exe) => return exe,
        Err(mut err) => {
            println!("warning: decompiling again after Executable was poisoned by the last test");
            // mutex poisoned, refresh it
            *err.get_mut().deref_mut() = make_exe();
            exe_mutex.clear_poison();
        }
    }
    exe_mutex.lock().unwrap()
}

fn make_exe() -> decompiler::Executable<'static> {
    let raw = DATA_DIR.get_file("x86_64_callconv").unwrap().contents();
    decompiler::Executable::parse(raw).unwrap()
}

#[macro_export]
macro_rules! assert_matches {
    ($value:expr, $pattern:pat) => { assert_matches!($value, $pattern, ()); };
    ($value:expr, $pattern:pat, $result_xform:expr) => {{
        match $value {
            $pattern => $result_xform,
            value => {
                let pattern_str = stringify!($pattern);
                panic!("assertion failed. value was expected to match pattern:\n  value: {:?}\n  pattern: {}", value, pattern_str);
            },
        }
    }};
}

pub fn compute_data_flow(func_name: &str) -> DataFlow {
    let exe = get_exe();
    let df = exe.decompile_function(func_name).unwrap();

    println!("---- mil");
    let mil = df.mil().unwrap();
    println!("{:?}", mil);

    // find function return value
    // (only works in the absence of control flow, which we're going to check
    // later)
    let ssa = df.ssa().unwrap();
    let types = exe.types().read_tx().unwrap();

    // simplifying hypothesis
    assert_eq!(ssa.cfg().block_count(), 1);

    println!("---- ssa");
    println!("{:?}", ssa);

    let ret_val = ssa
        .find_last_matching(ssa.cfg().entry_block_id(), |insn| match insn {
            decompiler::Insn::SetReturnValue(x) => Some(x),
            _ => None,
        })
        .expect("no return value?");

    let input_chain = ssa.find_input_chain(ret_val);
    println!("---- input chain");
    for &reg in &input_chain {
        let insn = ssa.get(reg).unwrap();
        println!(" r{:<4} = {:?}", reg.reg_index(), insn);
        match ssa.value_type(reg) {
            Some(tyid) => {
                let ty = types.read().get_through_alias(tyid).unwrap();
                println!("   : {:?}", ty);
            }
            None => {
                println!("   : ?");
            }
        }
    }

    DataFlow::from_input_chain(ssa, &input_chain)
}

/// A representation of a single input chain
/// (as defined and returned by [`decompiler::SSAProgram::find_input_chain`]),
/// in a form that is entirely standalone (independent of the SSAProgram it was
/// generated from), and easy to inspect/check.
#[derive(Debug)]
pub struct DataFlow {
    insns: Vec<decompiler::Insn>,
}
impl DataFlow {
    pub fn as_slice(&self) -> &[decompiler::Insn] {
        self.insns.as_slice()
    }
    fn from_input_chain(ssa: &decompiler::SSAProgram, input_chain: &[decompiler::Reg]) -> Self {
        // the gist of the problem:
        // - DataFlow is a mini-SSA, flattened with no control flow
        //   - therefore, Reg's are indices into the vector in DataFlow::insns
        // - to create this structure, we must map all registers

        let mut map = HashMap::new();
        let mut insns = Vec::new();

        for &reg in input_chain.iter().rev() {
            let mut insn = ssa.get(reg).unwrap();
            for input in insn.input_regs() {
                *input = *map.get(input).expect("input chain incomplete");
            }

            insns.push(insn);
            let new_reg = decompiler::Reg((insns.len() - 1).try_into().unwrap());
            map.insert(reg, new_reg);
        }

        DataFlow { insns }
    }
}
