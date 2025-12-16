#![cfg(test)]

use crate::{
    mil::{self, Control},
    ssa, ty,
};
use mil::{ArithOp, Insn, Reg};

mod constant_folding {
    use crate::{
        mil::{self, Control},
        ssa, ty, xform,
    };
    use mil::{ArithOp, Insn, Reg};

    #[test]
    fn addk() {
        let mut prog = mil::Program::new(Reg(0));
        prog.push(
            Reg(0),
            Insn::Ancestral {
                anc_name: mil::ANC_STACK_BOTTOM,
                reg_type: mil::RegType::Bytes(8),
            },
        );
        prog.push(Reg(1), Insn::Int { value: 5, size: 8 });
        prog.push(Reg(2), Insn::Int { value: 44, size: 8 });
        prog.push(Reg(0), Insn::Arith(ArithOp::Add, Reg(1), Reg(0)));
        prog.push(Reg(3), Insn::Arith(ArithOp::Add, Reg(0), Reg(1)));
        prog.push(Reg(4), Insn::Arith(ArithOp::Add, Reg(2), Reg(1)));
        prog.push(
            Reg(0),
            Insn::StoreMem {
                addr: Reg(4),
                value: Reg(3),
            },
        );
        prog.push(Reg(3), Insn::Int { value: 0, size: 8 });
        prog.push(
            Reg(4),
            Insn::Ancestral {
                anc_name: mil::ANC_STACK_BOTTOM,
                reg_type: mil::RegType::Bytes(8),
            },
        );
        prog.push(Reg(3), Insn::Arith(ArithOp::Add, Reg(3), Reg(4)));
        prog.push(Reg(0), Insn::SetReturnValue(Reg(3)));
        prog.push(Reg(0), Insn::Control(Control::Ret));

        let mut prog = ssa::Program::from_mil(prog);
        xform::canonical(&mut prog, &ty::TypeSet::new());

        assert_eq!(prog.cfg().block_count(), 1);
        assert_eq!(
            prog.get(Reg(4)).unwrap(),
            Insn::ArithK(ArithOp::Add, Reg(0), 10)
        );
        assert_eq!(prog.get(Reg(5)).unwrap(), Insn::Int { value: 49, size: 8 });
        assert_eq!(
            prog.get(Reg(8)).unwrap(),
            Insn::Ancestral {
                anc_name: mil::ANC_STACK_BOTTOM,
                reg_type: mil::RegType::Bytes(8),
            }
        );
        assert_eq!(prog.get(Reg(10)).unwrap(), Insn::SetReturnValue(Reg(8)));
    }

    #[test]
    fn mulk() {
        let mut prog = mil::Program::new(Reg(0));
        prog.push(
            Reg(0),
            Insn::Ancestral {
                anc_name: mil::ANC_STACK_BOTTOM,
                reg_type: mil::RegType::Bytes(8),
            },
        );
        prog.push(Reg(1), Insn::Int { value: 5, size: 8 });
        prog.push(Reg(2), Insn::Int { value: 44, size: 8 });
        prog.push(Reg(0), Insn::Arith(ArithOp::Mul, Reg(1), Reg(0)));
        prog.push(Reg(3), Insn::Arith(ArithOp::Mul, Reg(0), Reg(1)));
        prog.push(Reg(4), Insn::Arith(ArithOp::Mul, Reg(2), Reg(3)));
        prog.push(Reg(3), Insn::Int { value: 1, size: 8 });
        prog.push(
            Reg(0),
            Insn::StoreMem {
                addr: Reg(3),
                value: Reg(4),
            },
        );
        prog.push(
            Reg(4),
            Insn::Ancestral {
                anc_name: mil::ANC_STACK_BOTTOM,
                reg_type: mil::RegType::Bytes(8),
            },
        );
        prog.push(Reg(4), Insn::Arith(ArithOp::Mul, Reg(3), Reg(4)));
        prog.push(Reg(0), Insn::SetReturnValue(Reg(4)));
        prog.push(Reg(0), Insn::Control(Control::Ret));

        let mut prog = ssa::Program::from_mil(prog);
        xform::canonical(&mut prog, &ty::TypeSet::new());
        ssa::eliminate_dead_code(&mut prog);

        assert_eq!(prog.insns_rpo().count(), 6);
        assert_eq!(
            prog.get(Reg(5)).unwrap(),
            Insn::ArithK(ArithOp::Mul, Reg(0), 1100)
        );
        assert_eq!(prog.get(Reg(10)).unwrap(), Insn::SetReturnValue(Reg(8)));
    }
}

mod subreg_folding {
    use crate::{
        mil::{self, Control},
        ssa, ty, xform,
    };

    use test_log::test;

    define_ancestral_name!(ANC_A, "A");
    define_ancestral_name!(ANC_B, "B");

    #[test_log::test]
    fn part_of_concat() {
        use mil::{Insn, Reg};

        #[derive(Clone, Copy, Debug)]
        struct VariantParams {
            anc_a_sz: u16,
            anc_b_sz: u16,
            offset: u16,
            size: u16,
        }
        fn gen_prog(vp: VariantParams) -> mil::Program {
            let mut p = mil::Program::new(Reg(0));
            p.push(
                Reg(0),
                Insn::Ancestral {
                    anc_name: ANC_A,
                    reg_type: mil::RegType::Bytes(vp.anc_a_sz as _),
                },
            );
            p.push(
                Reg(1),
                Insn::Ancestral {
                    anc_name: ANC_B,
                    reg_type: mil::RegType::Bytes(vp.anc_b_sz as _),
                },
            );
            p.push(
                Reg(2),
                Insn::Concat {
                    lo: Reg(0),
                    hi: Reg(1),
                },
            );
            p.push(
                Reg(3),
                Insn::Part {
                    src: Reg(2),
                    offset: vp.offset,
                    size: vp.size,
                },
            );
            p.push(Reg(0), Insn::SetReturnValue(Reg(3)));
            p.push(Reg(0), Insn::Control(Control::Ret));
            p
        }

        let types = ty::TypeSet::new();

        for anc_a_sz in 1..=7 {
            for anc_b_sz in 1..=(8 - anc_a_sz) {
                let concat_sz = anc_a_sz + anc_b_sz;

                // case: fall within lo
                for offset in 0..=(anc_a_sz - 1) {
                    for size in 1..=(anc_a_sz - offset) {
                        let variant_params = VariantParams {
                            anc_a_sz,
                            anc_b_sz,
                            offset,
                            size,
                        };
                        let prog = gen_prog(variant_params);
                        let mut prog = ssa::Program::from_mil(prog);
                        xform::canonical(&mut prog, &types);

                        assert_eq!(
                            prog.get(Reg(3)).unwrap(),
                            if offset == 0 && size == anc_a_sz {
                                Insn::Get(Reg(0))
                            } else {
                                Insn::Part {
                                    src: Reg(0),
                                    offset,
                                    size,
                                }
                            }
                        );
                    }
                }

                // case: fall within hi
                for offset in anc_a_sz..concat_sz {
                    for size in 1..=(concat_sz - offset) {
                        let prog = gen_prog(VariantParams {
                            anc_a_sz,
                            anc_b_sz,
                            offset,
                            size,
                        });
                        let mut prog = ssa::Program::from_mil(prog);
                        xform::canonical(&mut prog, &types);

                        assert_eq!(
                            prog.get(Reg(3)).unwrap(),
                            if offset == anc_a_sz && size == anc_b_sz {
                                Insn::Get(Reg(1))
                            } else {
                                Insn::Part {
                                    src: Reg(1),
                                    offset: offset - anc_a_sz,
                                    size,
                                }
                            }
                        );
                    }
                }

                // case: crossing lo/hi
                for offset in 0..anc_a_sz {
                    for end in (anc_a_sz + 1)..concat_sz {
                        let size = end - offset;
                        if size == 0 {
                            continue;
                        }

                        dbg!((anc_a_sz, anc_b_sz, offset, size));

                        let prog = gen_prog(VariantParams {
                            anc_a_sz,
                            anc_b_sz,
                            offset,
                            size,
                        });
                        let mut prog = ssa::Program::from_mil(prog);
                        let orig_insn = prog.get(Reg(3)).unwrap();

                        xform::canonical(&mut prog, &types);
                        assert_eq!(prog.get(Reg(3)).unwrap(), orig_insn);
                    }
                }
            }
        }
    }

    #[test]
    fn part_of_part() {
        use mil::{Insn, Reg};

        #[derive(Clone, Copy)]
        struct VariantParams {
            src_sz: u16,
            offs0: u16,
            size0: u16,
            offs1: u16,
            size1: u16,
        }

        fn gen_prog(vp: VariantParams) -> mil::Program {
            let mut p = mil::Program::new(Reg(10_000));
            p.push(
                Reg(0),
                Insn::Ancestral {
                    anc_name: ANC_A,
                    reg_type: mil::RegType::Bytes(vp.src_sz as _),
                },
            );
            p.push(
                Reg(1),
                Insn::Part {
                    src: Reg(0),
                    offset: vp.offs0,
                    size: vp.size0,
                },
            );
            p.push(
                Reg(2),
                Insn::Part {
                    src: Reg(1),
                    offset: vp.offs1,
                    size: vp.size1,
                },
            );
            p.push(Reg(0), Insn::SetReturnValue(Reg(2)));
            p.push(Reg(0), Insn::Control(Control::Ret));
            p
        }

        let types = ty::TypeSet::new();
        let sample_data = b"12345678";

        for src_sz in 1..=8 {
            for offs0 in 0..src_sz {
                for size0 in 1..=(src_sz - offs0) {
                    for offs1 in 0..size0 {
                        for size1 in 1..=(size0 - offs1) {
                            let prog = gen_prog(VariantParams {
                                src_sz,
                                offs0,
                                size0,
                                offs1,
                                size1,
                            });
                            let mut prog = ssa::Program::from_mil(prog);
                            xform::canonical(&mut prog, &types);

                            let exp_offset = offs0 + offs1;
                            let exp_size = size1;
                            assert_eq!(
                                prog.get(Reg(2)).unwrap(),
                                if offs1 == 0 && size1 == src_sz {
                                    Insn::Get(Reg(0))
                                } else {
                                    Insn::Part {
                                        src: Reg(0),
                                        offset: exp_offset,
                                        size: exp_size,
                                    }
                                }
                            );

                            let offs0 = offs0 as usize;
                            let size0 = size0 as usize;
                            let offs1 = offs1 as usize;
                            let size1 = size1 as usize;
                            let exp_offset = exp_offset as usize;
                            let exp_size = exp_size as usize;
                            let range0 = offs0..offs0 + size0;
                            let range1 = offs1..offs1 + size1;
                            let exp_range = exp_offset..exp_offset + exp_size;
                            assert_eq!(&sample_data[range0][range1], &sample_data[exp_range]);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn combined_with_fold_get() {
    // check that a transform "sees through" the Insn::Get introduced by an
    // earlier transform

    let mut prog = mil::Program::new(Reg(0));
    prog.push(Reg(1), Insn::Int { value: 5, size: 8 });
    prog.push(Reg(2), Insn::Int { value: 44, size: 8 });

    // removed by fold_bitops
    prog.push(Reg(1), Insn::Arith(ArithOp::BitAnd, Reg(1), Reg(1)));
    prog.push(Reg(2), Insn::Arith(ArithOp::BitAnd, Reg(2), Reg(2)));

    // removed by fold_constants IF the Insn::Get's added by fold_bitops
    // is dereferenced
    prog.push(Reg(0), Insn::Arith(ArithOp::Mul, Reg(1), Reg(2)));
    prog.push(Reg(0), Insn::SetReturnValue(Reg(0)));
    prog.push(Reg(0), Insn::Control(Control::Ret));

    let mut prog = ssa::Program::from_mil(prog);
    super::canonical(&mut prog, &ty::TypeSet::new());
    ssa::eliminate_dead_code(&mut prog);

    assert_eq!(prog.insns_rpo().count(), 2);
    assert_eq!(
        prog.get(Reg(4)).unwrap(),
        Insn::Int {
            value: 5 * 44,
            size: 8
        }
    );
}

mod struct_ptr_member_access {
    use crate::{
        mil::{self, ArithOp, Control, Insn, Reg, RegType},
        ssa,
        ty::{self, TypeID},
        xform,
    };
    use test_log::test;

    define_ancestral_name!(ANC_STRUCT_PTR, "struct_ptr");

    /// Helper to create a test struct type:
    /// ```c
    /// typedef struct {  // size: 16
    ///     int64_t first;   // offset 0, size 8
    ///     int32_t second;  // offset 8, size 4
    ///     int32_t third;   // offset 12, size 4
    /// } *TestStruct;
    /// ```
    fn create_test_types(types: &ty::TypeSet) -> TestTypeIds {
        let mut wtx = types.write_tx().unwrap();

        // Allocate TypeIDs up-front
        let i64_tyid = TypeID(10);
        let i32_tyid = TypeID(11);
        let struct_tyid = TypeID(12);

        // Define primitives
        wtx.write()
            .set(
                i64_tyid,
                ty::Ty::Int(ty::Int {
                    size: 8,
                    signed: ty::Signedness::Signed,
                }),
            )
            .unwrap();
        wtx.write().set_name(i64_tyid, "i64".to_owned()).unwrap();

        wtx.write()
            .set(
                i32_tyid,
                ty::Ty::Int(ty::Int {
                    size: 4,
                    signed: ty::Signedness::Signed,
                }),
            )
            .unwrap();
        wtx.write().set_name(i32_tyid, "i32".to_owned()).unwrap();

        // Define struct
        wtx.write()
            .set(
                struct_tyid,
                ty::Ty::Struct(ty::Struct {
                    size: 16,
                    members: vec![
                        ty::StructMember {
                            name: "first".to_string().into(),
                            tyid: i64_tyid,
                            offset: 0,
                        },
                        ty::StructMember {
                            name: "second".to_string().into(),
                            tyid: i32_tyid,
                            offset: 8,
                        },
                        ty::StructMember {
                            name: "third".to_string().into(),
                            tyid: i32_tyid,
                            offset: 12,
                        },
                    ],
                }),
            )
            .unwrap();
        wtx.write()
            .set_name(struct_tyid, "TestStruct".to_owned())
            .unwrap();

        // Add a type that's a pointer to struct_tyid
        let struct_ptr_tyid = TypeID(13);
        wtx.write()
            .set(struct_ptr_tyid, ty::Ty::Ptr(struct_tyid))
            .unwrap();

        wtx.commit().unwrap();

        TestTypeIds { struct_ptr_tyid }
    }

    struct TestTypeIds {
        struct_ptr_tyid: ty::TypeID,
    }

    /// Test: Member at offset 0 (first member)
    ///
    /// Input:
    ///   r0 <- struct_ptr (typed as TestStruct)
    ///   r1 <- ArithK(Add, r0, 0)
    ///   r2 <- LoadMem(r1, 8)
    ///
    /// Expected: LoadMem of full struct, then StructGetMember for "first"
    #[test]
    fn member_at_offset_0() {
        let types = ty::TypeSet::new();
        let ty_ids = create_test_types(&types);

        let mut prog = mil::Program::new(Reg(0));
        prog.push(
            Reg(0),
            Insn::Ancestral {
                anc_name: ANC_STRUCT_PTR,
                reg_type: RegType::Bytes(8),
            },
        );
        // ArithK with offset 0 to first member
        prog.push(Reg(1), Insn::ArithK(ArithOp::Add, Reg(0), 0));
        prog.push(
            Reg(2),
            Insn::LoadMem {
                addr: Reg(1),
                size: 8,
            },
        );
        prog.push(Reg(0), Insn::SetReturnValue(Reg(2)));
        prog.push(Reg(0), Insn::Control(Control::Ret));

        prog.set_value_type(0, ty_ids.struct_ptr_tyid);

        let mut prog = ssa::Program::from_mil(prog);

        xform::canonical(&mut prog, &types);

        // Verify: should have LoadMem of full struct (size 16)
        // followed by StructGetMember for "first"
        let struct_load_reg = find_insn_matching(&prog, |insn| {
            matches!(
                insn,
                Insn::LoadMem {
                    addr: Reg(0),
                    size: 16
                }
            )
        });
        assert!(
            struct_load_reg.is_some(),
            "Expected LoadMem of full struct (size 16)"
        );

        let member_access = find_insn_matching(&prog, |insn| {
            matches!(insn, Insn::StructGetMember { name: "first", .. })
        });
        assert!(
            member_access.is_some(),
            "Expected StructGetMember for 'first'"
        );
    }

    /// Test: Member at non-zero offset
    ///
    /// Input:
    ///   r0 <- struct_ptr (typed as TestStruct)
    ///   r1 <- ArithK(Add, r0, 8)
    ///   r2 <- LoadMem(r1, 4)
    ///
    /// Expected: LoadMem of full struct, then StructGetMember for "second"
    #[test]
    fn member_at_nonzero_offset() {
        let types = ty::TypeSet::new();
        let ty_ids = create_test_types(&types);

        let mut prog = mil::Program::new(Reg(0));
        prog.push(
            Reg(0),
            Insn::Ancestral {
                anc_name: ANC_STRUCT_PTR,
                reg_type: RegType::Bytes(8),
            },
        );
        prog.push(Reg(1), Insn::ArithK(ArithOp::Add, Reg(0), 8));
        prog.push(
            Reg(2),
            Insn::LoadMem {
                addr: Reg(1),
                size: 4,
            },
        );
        prog.push(Reg(0), Insn::SetReturnValue(Reg(2)));
        prog.push(Reg(0), Insn::Control(Control::Ret));

        prog.set_value_type(0, ty_ids.struct_ptr_tyid);

        let mut prog = ssa::Program::from_mil(prog);

        xform::canonical(&mut prog, &types);

        let struct_load_reg = find_insn_matching(&prog, |insn| {
            matches!(
                insn,
                Insn::LoadMem {
                    addr: Reg(0),
                    size: 16
                }
            )
        });
        assert!(
            struct_load_reg.is_some(),
            "Expected LoadMem of full struct (size 16)"
        );

        let member_access = find_insn_matching(&prog, |insn| {
            matches!(insn, Insn::StructGetMember { name: "second", .. })
        });
        assert!(
            member_access.is_some(),
            "Expected StructGetMember for 'second'"
        );
    }

    /// Test: Last member of struct
    #[test]
    fn member_at_last_offset() {
        let types = ty::TypeSet::new();
        let ty_ids = create_test_types(&types);

        let mut prog = mil::Program::new(Reg(0));
        prog.push(
            Reg(0),
            Insn::Ancestral {
                anc_name: ANC_STRUCT_PTR,
                reg_type: RegType::Bytes(8),
            },
        );
        prog.push(Reg(1), Insn::ArithK(ArithOp::Add, Reg(0), 12));
        prog.push(
            Reg(2),
            Insn::LoadMem {
                addr: Reg(1),
                size: 4,
            },
        );
        prog.push(Reg(0), Insn::SetReturnValue(Reg(2)));
        prog.push(Reg(0), Insn::Control(Control::Ret));

        prog.set_value_type(0, ty_ids.struct_ptr_tyid);

        let mut prog = ssa::Program::from_mil(prog);

        xform::canonical(&mut prog, &types);

        let member_access = find_insn_matching(&prog, |insn| {
            matches!(insn, Insn::StructGetMember { name: "third", .. })
        });
        assert!(
            member_access.is_some(),
            "Expected StructGetMember for 'third'"
        );
    }

    /// Test: Offset doesn't match any member - should NOT transform
    ///
    /// Accessing offset 4 which falls in the middle of "first" (size 8)
    #[test]
    fn invalid_offset_no_transform() {
        let types = ty::TypeSet::new();
        let ty_ids = create_test_types(&types);

        let mut prog = mil::Program::new(Reg(0));
        prog.push(
            Reg(0),
            Insn::Ancestral {
                anc_name: ANC_STRUCT_PTR,
                reg_type: RegType::Bytes(8),
            },
        );
        // Offset 4 doesn't align with any member start
        prog.push(Reg(1), Insn::ArithK(ArithOp::Add, Reg(0), 4));
        prog.push(
            Reg(2),
            Insn::LoadMem {
                addr: Reg(1),
                size: 4,
            },
        );
        prog.push(Reg(0), Insn::SetReturnValue(Reg(2)));
        prog.push(Reg(0), Insn::Control(Control::Ret));

        prog.set_value_type(0, ty_ids.struct_ptr_tyid);

        let mut prog = ssa::Program::from_mil(prog);

        xform::canonical(&mut prog, &types);

        // Should NOT have StructGetMember since offset doesn't match
        let member_access =
            find_insn_matching(&prog, |insn| matches!(insn, Insn::StructGetMember { .. }));
        assert!(
            member_access.is_none(),
            "Should not transform when offset doesn't match any member"
        );
    }

    /// Test: Size mismatch - offset correct but size wrong
    ///
    /// Accessing "second" (offset 8, size 4) but loading 8 bytes
    #[test]
    fn size_mismatch_no_transform() {
        let types = ty::TypeSet::new();
        let ty_ids = create_test_types(&types);

        let mut prog = mil::Program::new(Reg(0));
        prog.push(
            Reg(0),
            Insn::Ancestral {
                anc_name: ANC_STRUCT_PTR,
                reg_type: RegType::Bytes(8),
            },
        );
        prog.push(Reg(1), Insn::ArithK(ArithOp::Add, Reg(0), 8));
        // Wrong size: "second" is 4 bytes, but we load 8
        prog.push(
            Reg(2),
            Insn::LoadMem {
                addr: Reg(1),
                size: 8,
            },
        );
        prog.push(Reg(0), Insn::SetReturnValue(Reg(2)));
        prog.push(Reg(0), Insn::Control(Control::Ret));

        prog.set_value_type(0, ty_ids.struct_ptr_tyid);

        let mut prog = ssa::Program::from_mil(prog);

        xform::canonical(&mut prog, &types);

        // Should NOT transform to member "second" with wrong size
        let member_access = find_insn_matching(&prog, |insn| {
            matches!(insn, Insn::StructGetMember { name: "second", .. })
        });
        assert!(
            member_access.is_none(),
            "Should not transform when size doesn't match member size"
        );
    }

    /// Test: Base register has no type info - should NOT transform
    #[test]
    fn no_type_info_no_transform() {
        let types = ty::TypeSet::new();
        let _ = create_test_types(&types);

        let mut prog = mil::Program::new(Reg(0));
        prog.push(
            Reg(0),
            Insn::Ancestral {
                anc_name: ANC_STRUCT_PTR,
                reg_type: RegType::Bytes(8),
            },
        );
        prog.push(Reg(1), Insn::ArithK(ArithOp::Add, Reg(0), 8));
        prog.push(
            Reg(2),
            Insn::LoadMem {
                addr: Reg(1),
                size: 4,
            },
        );
        prog.push(Reg(0), Insn::SetReturnValue(Reg(2)));
        prog.push(Reg(0), Insn::Control(Control::Ret));

        let mut prog = ssa::Program::from_mil(prog);
        // Deliberately NOT setting type info

        xform::canonical(&mut prog, &types);

        let member_access =
            find_insn_matching(&prog, |insn| matches!(insn, Insn::StructGetMember { .. }));
        assert!(
            member_access.is_none(),
            "Should not transform when no type info is available"
        );
    }

    /// Test: Nested struct access
    ///
    /// struct Inner { int32_t x; int32_t y; };  // size 8
    /// struct Outer { int64_t a; Inner b; };    // size 16
    ///
    /// Accessing outer.b.y should work
    #[test]
    fn nested_struct_member() {
        let types = ty::TypeSet::new();
        let mut wtx = types.write_tx().unwrap();

        // Allocate TypeIDs explicitly and uniquely
        let i64_tyid = ty::TypeID(10);
        let i32_tyid = ty::TypeID(11);
        let inner_tyid = ty::TypeID(12);
        let outer_tyid = ty::TypeID(13);
        let outer_ptr_tyid = ty::TypeID(14);

        // i64
        wtx.write()
            .set(
                i64_tyid,
                ty::Ty::Int(ty::Int {
                    size: 8,
                    signed: ty::Signedness::Signed,
                }),
            )
            .unwrap();
        wtx.write().set_name(i64_tyid, "i64".to_owned()).unwrap();

        // i32
        wtx.write()
            .set(
                i32_tyid,
                ty::Ty::Int(ty::Int {
                    size: 4,
                    signed: ty::Signedness::Signed,
                }),
            )
            .unwrap();
        wtx.write().set_name(i32_tyid, "i32".to_owned()).unwrap();

        // Inner { i32 x @0; i32 y @4; } size 8
        wtx.write()
            .set(
                inner_tyid,
                ty::Ty::Struct(ty::Struct {
                    size: 8,
                    members: vec![
                        ty::StructMember {
                            name: "x".to_string().into(),
                            tyid: i32_tyid,
                            offset: 0,
                        },
                        ty::StructMember {
                            name: "y".to_string().into(),
                            tyid: i32_tyid,
                            offset: 4,
                        },
                    ],
                }),
            )
            .unwrap();
        wtx.write()
            .set_name(inner_tyid, "Inner".to_owned())
            .unwrap();

        // Outer { i64 a @0; Inner b @8; } size 16
        wtx.write()
            .set(
                outer_tyid,
                ty::Ty::Struct(ty::Struct {
                    size: 16,
                    members: vec![
                        ty::StructMember {
                            name: "a".to_string().into(),
                            tyid: i64_tyid,
                            offset: 0,
                        },
                        ty::StructMember {
                            name: "b".to_string().into(),
                            tyid: inner_tyid,
                            offset: 8,
                        },
                    ],
                }),
            )
            .unwrap();
        wtx.write()
            .set_name(outer_tyid, "Outer".to_owned())
            .unwrap();

        wtx.write()
            .set(outer_ptr_tyid, ty::Ty::Ptr(outer_tyid))
            .unwrap();

        wtx.commit().unwrap();

        let mut prog = mil::Program::new(Reg(0));
        prog.push(
            Reg(0),
            Insn::Ancestral {
                anc_name: ANC_STRUCT_PTR,
                reg_type: RegType::Bytes(8),
            },
        );
        // Access outer.b.y: offset 8 (b) + 4 (y) = 12
        prog.push(Reg(1), Insn::ArithK(ArithOp::Add, Reg(0), 12));
        prog.push(
            Reg(2),
            Insn::LoadMem {
                addr: Reg(1),
                size: 4,
            },
        );
        prog.push(Reg(0), Insn::SetReturnValue(Reg(2)));
        prog.push(Reg(0), Insn::Control(Control::Ret));

        prog.set_value_type(0, outer_ptr_tyid);

        let mut prog = ssa::Program::from_mil(prog);

        xform::canonical(&mut prog, &types);

        // Should produce nested member access: outer.b.y
        // First get member "b", then get member "y" from that
        let outer_member = find_insn_matching(&prog, |insn| {
            matches!(insn, Insn::StructGetMember { name: "b", .. })
        });
        assert!(outer_member.is_some(), "Expected StructGetMember for 'b'");

        let inner_member = find_insn_matching(&prog, |insn| {
            matches!(insn, Insn::StructGetMember { name: "y", .. })
        });
        assert!(inner_member.is_some(), "Expected StructGetMember for 'y'");
    }

    /// Test: Negative offset should NOT transform
    #[test]
    fn negative_offset_no_transform() {
        let types = ty::TypeSet::new();
        let ty_ids = create_test_types(&types);

        let mut prog = mil::Program::new(Reg(0));
        prog.push(
            Reg(0),
            Insn::Ancestral {
                anc_name: ANC_STRUCT_PTR,
                reg_type: RegType::Bytes(8),
            },
        );
        prog.push(Reg(1), Insn::ArithK(ArithOp::Add, Reg(0), -4));
        prog.push(
            Reg(2),
            Insn::LoadMem {
                addr: Reg(1),
                size: 4,
            },
        );
        prog.push(Reg(0), Insn::SetReturnValue(Reg(2)));
        prog.push(Reg(0), Insn::Control(Control::Ret));

        prog.set_value_type(0, ty_ids.struct_ptr_tyid);

        let mut prog = ssa::Program::from_mil(prog);

        xform::canonical(&mut prog, &types);

        let member_access =
            find_insn_matching(&prog, |insn| matches!(insn, Insn::StructGetMember { .. }));
        assert!(
            member_access.is_none(),
            "Should not transform with negative offset"
        );
    }

    /// Helper: find an instruction in the program matching a predicate
    fn find_insn_matching<F>(prog: &ssa::Program, pred: F) -> Option<Reg>
    where
        F: Fn(Insn) -> bool,
    {
        for (_, reg) in prog.insns_rpo() {
            if let Some(insn) = prog.get(reg) {
                if pred(insn) {
                    return Some(reg);
                }
            }
        }
        None
    }
}
