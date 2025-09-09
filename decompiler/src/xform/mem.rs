/*

Observations:

- every LoadMem instruction can potentially be replaced with the value written
by a relevant StoreMem at the same memory region.

- We define a reference register R.  Every LoadMem and StoreMem takes a general
value as address; here we only consider addresses in the form R + constant,
and only "see" the constant part. By doing so, we treat mem insns as operating
on a linear memory address space. Insns that take addresses in other forms or
by other base registers are ignored (in some cases they may be processed by
repeating the procedure with a base register).

- a LoadMem is parameterized as (start addr, size). we can trivially gather the
memory interval [start, end) with start = start addr; end = start+size <= start.

    - a StoreMem is parameterized as (start addr, value); value has a size in
    bytes (gathered by looking up the defining insn; we already have an function
    for that). similarly to LoadMem, its memory interval is [start, end), with:

        - start = start addr
        - end = start + value.size()
        - end > start
        - if value.size() is 0, the StoreMem is ineffective, so we just pretend
        not to see.

- intuitively: given a LoadMem, the relevant StoreMem instructions (there can be
multiple!) are those whose effects are visible at the LoadMem's "time" (program
location).

- we can simplify by utilizing part/concat instructions (at least temporarily
or logically)

- Given a Load(start_l, end_l);

    - Find the last dominating Store(start_s, end_s, v) insn, such that
    Store.intv = [start_s, end_s) conflicts with Load.intv = [start_l, end_l)
        - To do this, we direct the scan through the dom tree, BB by BB

    - in the base case, if Load.intv == Store.intv, then the Load can just be
    replaced with v

    - more generally,
        Let intr_intv
            = Store.intv ∩ Load.intv
            = [start_s, end_s) ∩ [start_l, end_l)
            = [start_i, end_i)
        Then all the following hold:
            start_s <= start_i
            start_l <= start_i
            start_i <= end_i
            end_i <= end_s
            end_i <= end_l

        In particular:
            start_l <= start_i <= end_i <= end_l

        Then Load.intv = [start_l, start_i) ++ [start_i, end_i) ++ [end_i, end_l)
            (of course, some of these intervals may be empty, but no big deal)

        Then
            Load(start_l, end_l)
            == Concat(
                Load(start_l, start_i),
                Part(value: value, offset: 0, size: end_i - start_i),
                Load(end_i, end_s),
            )

            (Note that we take a part of the store value, rebasing the interval from start_i to 0)

            This is the substitution we perform.
            We recursively process the two new Load insns.

*/

use crate::{
    cfg,
    mil::{self, Insn, Reg},
    ssa,
};

pub fn fold_load_store(
    prog: &mut ssa::OpenProgram,
    ref_reg: mil::Reg,
    load_reg: mil::Reg,
    load_bid: cfg::BlockID,
    load_ndx_in_blk: mil::Index,
) -> bool {
    //
    // check if we're looking at a Load that we know how to transform
    // Load(ArithK(Add, ref_reg, offset), size)
    //
    let Insn::LoadMem {
        addr: addr_l,
        size: size_l,
    } = prog.get(load_reg).unwrap()
    else {
        return false;
    };
    let load = {
        let Insn::ArithK(mil::ArithOp::Add, offset_reg, start) = prog.get(addr_l).unwrap() else {
            // not in register-relative form; we can't work with this
            return false;
        };

        if offset_reg != ref_reg {
            // wrong reference register; we can't work with this
            return false;
        }

        let end = start + size_l as i64;
        LoadInt { start, end }
    };

    let Some(store) = find_dominating_conflicting_store(prog, ref_reg, &load, load_reg, load_bid)
    else {
        return false;
    };

    // intersect LoadInt, StoreInt
    let start_i = load.start.max(store.start);
    let end_i = load.end.min(store.end);

    let left_size = (start_i - load.start).try_into().unwrap();
    let left = prog.insert(
        load_bid,
        load_ndx_in_blk,
        load_or_void(
            // same as load.start; but we can reuse the same ArithK as in the original load
            addr_l,    // addr
            left_size, // size
        ),
    );

    let mid_size = (end_i - start_i).try_into().unwrap();
    let mid_offset = (start_i - store.start).try_into().unwrap();
    let mid = prog.insert(
        load_bid,
        load_ndx_in_blk,
        Insn::Part {
            src: store.value,
            offset: mid_offset,
            size: mid_size,
        },
    );

    let right_addr = prog.insert(
        load_bid,
        load_ndx_in_blk,
        Insn::ArithK(mil::ArithOp::Add, ref_reg, end_i),
    );
    let right_size = (load.end - end_i).try_into().unwrap();
    let right = prog.insert(
        load_bid,
        load_ndx_in_blk + 1,
        load_or_void(
            right_addr, // addr
            right_size, // size
        ),
    );

    let mid_left = prog.insert(
        load_bid,
        load_ndx_in_blk,
        Insn::Concat { lo: mid, hi: left },
    );

    // replace the load with the final replacement value (the outermost Concat)
    prog.set(
        load_reg,
        Insn::Concat {
            lo: right,
            hi: mid_left,
        },
    );

    true
}

struct LoadInt {
    start: i64,
    end: i64,
}
struct StoreInt {
    start: i64,
    end: i64,
    value: mil::Reg,
}

fn load_or_void(addr: Reg, size: u32) -> Insn {
    if size == 0 {
        Insn::Void
    } else {
        Insn::LoadMem { addr, size }
    }
}

/// Find the last dominating store instruction.
fn find_dominating_conflicting_store(
    prog: &ssa::Program,
    ref_reg: mil::Reg,
    load: &LoadInt,
    load_reg: mil::Reg,
    load_bid: cfg::BlockID,
) -> Option<StoreInt> {
    assert!(load.start <= load.end);
    if load.end == load.start {
        return None;
    }

    let select_store = |insn: &mil::Insn| {
        let &Insn::StoreMem {
            addr: addr_s,
            value: value_s,
        } = insn
        else {
            return None;
        };

        let Insn::ArithK(mil::ArithOp::Add, offset_reg, start_s) = prog.get(addr_s).unwrap() else {
            // not in register-relative form; we can't work with this
            return None;
        };

        if offset_reg != ref_reg {
            // wrong reference register; we can't work with this
            return None;
        }

        let size_s = prog
            .reg_type(value_s)
            .bytes_size()
            .expect("StoreMem: value with unsized type?!");
        let end_s = start_s + size_s as i64;

        if start_s < load.end && end_s > load.start {
            // we found a relevant StoreMem: represent it in interval+value form and use it
            return Some(StoreInt {
                start: start_s,
                end: end_s,
                value: value_s,
            });
        }

        None
    };

    if let Some(store) = prog
        .block_regs(load_bid)
        .rev()
        .skip_while(|&r| r != load_reg)
        .find_map(|r| select_store(&prog.get(r).unwrap()))
    {
        return Some(store);
    }

    for bid in prog.cfg().dom_tree().imm_doms(load_bid) {
        if let Some(store) = prog
            .block_regs(bid)
            .rev()
            .find_map(|r| select_store(&prog.get(r).unwrap()))
        {
            return Some(store);
        }
    }

    return None;
}

#[cfg(test)]
mod tests {
    use crate::{
        mil::{self, ArithOp, Control, Insn, Reg},
        ssa, ty, x86_to_mil, xform,
    };

    define_ancestral_name!(ANC_MEM, "memory");

    #[test]
    fn single_bb_direct() {
        for size in [1, 2, 4, 5, 8] {
            let mut program = mil::Program::new(Reg(0));
            program.push(Reg(0), Insn::Ancestral(ANC_MEM));
            program.push(Reg(1), Insn::Const { size, value: -123 });
            program.push(Reg(2), Insn::Ancestral(x86_to_mil::ANC_RSP));
            program.push(Reg(3), Insn::ArithK(ArithOp::Add, Reg(2), 16));
            program.push(
                Reg(4),
                Insn::StoreMem {
                    addr: Reg(3),
                    value: Reg(1),
                },
            );
            program.push(
                Reg(5),
                Insn::LoadMem {
                    addr: Reg(3),
                    size: size.try_into().unwrap(),
                },
            );
            program.push(Reg(6), Insn::SetReturnValue(Reg(5)));
            program.push(Reg(6), Insn::Control(Control::Ret));

            let mut program = ssa::Program::from_mil(program);

            println!("ssa pre-xform:\n{program:?}");
            xform::canonical(&mut program, &ty::TypeSet::new());
            println!("ssa post-xform:\n{program:?}");

            let insn = program.get(Reg(6)).unwrap();
            assert_eq!(insn, Insn::SetReturnValue(Reg(1)));
        }
    }

    #[test]
    fn single_bb_part() {
        let mut program = mil::Program::new(Reg(0));
        program.push(Reg(0), Insn::Ancestral(ANC_MEM));
        program.push(
            Reg(1),
            Insn::Const {
                size: 8,
                value: -123,
            },
        );
        program.push(Reg(2), Insn::Ancestral(x86_to_mil::ANC_RSP));
        program.push(Reg(3), Insn::ArithK(ArithOp::Add, Reg(2), 16));
        program.push(
            Reg(4),
            Insn::StoreMem {
                addr: Reg(3),
                value: Reg(1),
            },
        );
        program.push(Reg(5), Insn::ArithK(mil::ArithOp::Add, Reg(3), 2));
        program.push(
            Reg(6),
            Insn::LoadMem {
                addr: Reg(5),
                size: 3,
            },
        );
        program.push(Reg(7), Insn::SetReturnValue(Reg(6)));
        program.push(Reg(7), Insn::Control(Control::Ret));

        let mut program = ssa::Program::from_mil(program);

        println!("ssa pre-xform:\n{program:?}");
        xform::canonical(&mut program, &ty::TypeSet::new());
        println!("ssa post-xform:\n{program:?}");

        let ret = program.get(Reg(7)).unwrap();
        let Insn::SetReturnValue(ret_val) = ret else {
            panic!()
        };

        assert!(matches!(
            program.get(ret_val).unwrap(),
            Insn::Part {
                src: Reg(1),
                offset: 2,
                size: 3
            }
        ));
    }

    #[test]
    fn single_bb_concat() {
        let mut program = mil::Program::new(Reg(0));
        program.push(Reg(0), Insn::Ancestral(ANC_MEM));
        program.push(
            Reg(1),
            Insn::Const {
                size: 8,
                value: -123,
            },
        );
        program.push(Reg(2), Insn::Ancestral(x86_to_mil::ANC_RSP));
        program.push(Reg(3), Insn::ArithK(ArithOp::Add, Reg(2), 16));
        program.push(
            Reg(4),
            Insn::StoreMem {
                addr: Reg(3),
                value: Reg(1),
            },
        );
        program.push(Reg(5), Insn::ArithK(mil::ArithOp::Add, Reg(3), 2));
        program.push(
            Reg(6),
            Insn::LoadMem {
                addr: Reg(5),
                size: 23,
            },
        );
        program.push(Reg(7), Insn::SetReturnValue(Reg(6)));
        program.push(Reg(7), Insn::Control(Control::Ret));

        let mut program = ssa::Program::from_mil(program);

        println!("ssa pre-xform:\n{program:?}");
        xform::canonical(&mut program, &ty::TypeSet::new());
        println!("ssa post-xform:\n{program:?}");

        let Insn::SetReturnValue(ret_val) = program.get(Reg(7)).unwrap() else {
            panic!()
        };
        let Insn::Concat { hi, lo } = program.get(ret_val).unwrap() else {
            panic!()
        };
        assert_eq!(
            Insn::Part {
                src: Reg(1),
                offset: 2,
                size: 6
            },
            program.get(hi).unwrap()
        );

        let Insn::LoadMem { addr, size: 17 } = program.get(lo).unwrap() else {
            panic!()
        };

        let Insn::ArithK(ArithOp::Add, base_reg, 24) = program.get(addr).unwrap() else {
            panic!()
        };
        assert_eq!(
            Insn::Ancestral(x86_to_mil::ANC_RSP),
            program.get(base_reg).unwrap()
        );
    }
}
