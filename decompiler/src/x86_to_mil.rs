use crate::mil::{self, AncestralName, Control};
use std::borrow::Cow;
use std::sync::Arc;

use crate::ty;
use crate::util::{ToWarnings, Warnings};
use iced_x86::{Formatter, IntelFormatter};
use iced_x86::{OpKind, Register};

use anyhow::{Context as _, Result};
use tracing::{event, instrument, span, Level};

pub mod callconv;

pub fn import(
    insns: impl Iterator<Item = iced_x86::Instruction>,
    types: Arc<ty::TypeSet>,
    func_tyid_opt: Option<ty::TypeID>,
) -> anyhow::Result<mil::Program> {
    let (mil, _) = Importer::new(types)?.translate(insns, func_tyid_opt)?;
    Ok(mil)
}

// TODO make this private
struct Importer {
    // NOTE The order here is load-bearing. See safety notes in Self::new
    pb: mil::Program,
    // formally unused, but keeps `rtx` valid (borrowed unbeknownst to the compiler)
    _types: Arc<ty::TypeSet>,
    // 'static here is false; we're borrowing from pb via unsafe
    rtx: ty::ReadTx<'static>,
}

impl Importer {
    fn new(types: Arc<ty::TypeSet>) -> ty::Result<Self> {
        // SAFETY: rtx borrows from types, which we ensure outlives rtx via
        // borrow order.
        let rtx: ty::ReadTx<'static> = unsafe {
            let types: &'static _ = &*Arc::as_ptr(&types);
            types.read_tx()?
        };
        let mut bld = Importer {
            pb: mil::Program::new(Self::R_TMP_FIRST, Some(Arc::clone(&types))),
            _types: types,
            rtx,
        };

        bld.init_ancestral(Self::RSP, mil::ANC_STACK_BOTTOM, mil::LLType::Bytes(8));

        // ensure all registers are initialized at least once. most of these
        // instructions get "deleted" (masked) if the program is valid and the
        // decompilation correct. if not, this allows the program to still be
        // decompiled into something (albeit, with some "holes")

        bld.init_ancestral(Self::PF, ANC_CF, mil::LLType::Bool);
        bld.init_ancestral(Self::PF, ANC_PF, mil::LLType::Bool);
        bld.init_ancestral(Self::AF, ANC_AF, mil::LLType::Bool);
        bld.init_ancestral(Self::ZF, ANC_ZF, mil::LLType::Bool);
        bld.init_ancestral(Self::SF, ANC_SF, mil::LLType::Bool);
        bld.init_ancestral(Self::TF, ANC_TF, mil::LLType::Bool);
        bld.init_ancestral(Self::IF, ANC_IF, mil::LLType::Bool);
        bld.init_ancestral(Self::DF, ANC_DF, mil::LLType::Bool);
        bld.init_ancestral(Self::OF, ANC_OF, mil::LLType::Bool);
        bld.init_ancestral(Self::RBP, ANC_RBP, mil::LLType::Bytes(8));
        bld.init_ancestral(Self::RSP, ANC_RSP, mil::LLType::Bytes(8));
        bld.init_ancestral(Self::RIP, ANC_RIP, mil::LLType::Bytes(8));
        bld.init_ancestral(Self::RDI, ANC_RDI, mil::LLType::Bytes(8));
        bld.init_ancestral(Self::RSI, ANC_RSI, mil::LLType::Bytes(8));
        bld.init_ancestral(Self::RAX, ANC_RAX, mil::LLType::Bytes(8));
        bld.init_ancestral(Self::RBX, ANC_RBX, mil::LLType::Bytes(8));
        bld.init_ancestral(Self::RCX, ANC_RCX, mil::LLType::Bytes(8));
        bld.init_ancestral(Self::RDX, ANC_RDX, mil::LLType::Bytes(8));
        bld.init_ancestral(Self::R8, ANC_R8, mil::LLType::Bytes(8));
        bld.init_ancestral(Self::R9, ANC_R9, mil::LLType::Bytes(8));
        bld.init_ancestral(Self::R10, ANC_R10, mil::LLType::Bytes(8));
        bld.init_ancestral(Self::R11, ANC_R11, mil::LLType::Bytes(8));
        bld.init_ancestral(Self::R12, ANC_R12, mil::LLType::Bytes(8));
        bld.init_ancestral(Self::R13, ANC_R13, mil::LLType::Bytes(8));
        bld.init_ancestral(Self::R14, ANC_R14, mil::LLType::Bytes(8));
        bld.init_ancestral(Self::R15, ANC_R15, mil::LLType::Bytes(8));
        bld.init_ancestral(Self::ZMM0, ANC_ZMM0, mil::LLType::Bytes(64));
        bld.init_ancestral(Self::ZMM1, ANC_ZMM1, mil::LLType::Bytes(64));
        bld.init_ancestral(Self::ZMM2, ANC_ZMM2, mil::LLType::Bytes(64));
        bld.init_ancestral(Self::ZMM3, ANC_ZMM3, mil::LLType::Bytes(64));
        bld.init_ancestral(Self::ZMM4, ANC_ZMM4, mil::LLType::Bytes(64));
        bld.init_ancestral(Self::ZMM5, ANC_ZMM5, mil::LLType::Bytes(64));
        bld.init_ancestral(Self::ZMM6, ANC_ZMM6, mil::LLType::Bytes(64));
        bld.init_ancestral(Self::ZMM7, ANC_ZMM7, mil::LLType::Bytes(64));
        bld.init_ancestral(Self::ZMM8, ANC_ZMM8, mil::LLType::Bytes(64));
        bld.init_ancestral(Self::ZMM9, ANC_ZMM9, mil::LLType::Bytes(64));
        bld.init_ancestral(Self::ZMM10, ANC_ZMM10, mil::LLType::Bytes(64));
        bld.init_ancestral(Self::ZMM11, ANC_ZMM11, mil::LLType::Bytes(64));
        bld.init_ancestral(Self::ZMM12, ANC_ZMM12, mil::LLType::Bytes(64));
        bld.init_ancestral(Self::ZMM13, ANC_ZMM13, mil::LLType::Bytes(64));
        bld.init_ancestral(Self::ZMM14, ANC_ZMM14, mil::LLType::Bytes(64));
        bld.init_ancestral(Self::ZMM15, ANC_ZMM15, mil::LLType::Bytes(64));

        Ok(bld)
    }

    /// Emit an Ancestral instruction for the given AncestralName and assign the
    /// result to register to `reg`.
    ///
    /// The register is assigned the given LLType. The high-level type is a
    /// corresponding ty::Ty::Unknown.
    fn init_ancestral(
        &mut self,
        reg: mil::Reg,
        anc_name: AncestralName,
        llt: mil::LLType,
    ) -> mil::Index {
        // TODO this function could be removed. this used to make more sense,
        // when it was more complex
        self.emit(
            reg,
            mil::Insn::Ancestral {
                anc_name,
                ll_type: llt,
            },
        )
    }

    pub fn set_value_type(&mut self, index: mil::Index, tyid: ty::TypeID) {
        self.pb.set_value_type(index, tyid);
    }

    fn tmp_gen(&mut self) -> mil::Reg {
        self.pb.tmp_gen()
    }

    #[instrument(skip_all, level = "trace")]
    pub fn translate(
        mut self,
        insns: impl Iterator<Item = iced_x86::Instruction>,
        func_tyid_opt: Option<ty::TypeID>,
    ) -> Result<(mil::Program, Warnings)> {
        self.pb.set_function_type_id(func_tyid_opt);
        use iced_x86::{OpKind, Register};

        let mut func_ty = None;
        if let Some(tyid) = func_tyid_opt {
            if let Some(cow) = self.rtx.read().get_through_alias(tyid)? {
                if let ty::Ty::Subroutine(subr_ty) = cow.into_owned() {
                    func_ty = Some(subr_ty);
                }
            }
        }
        event!(Level::DEBUG, ?func_ty, "function type resolved");

        let mut warnings = Warnings::default();

        if let Some(func_ty) = &func_ty {
            let param_count = func_ty.param_tyids.len();
            let res = callconv::unpack_params(&mut self, &func_ty.param_tyids, func_ty.return_tyid)
                .context("while applying the calling convention for parameters");

            match res {
                Ok(report) => {
                    if report.ok_count < param_count {
                        event!(
                            Level::WARN,
                            ok_count = report.ok_count,
                            total_count = param_count,
                            "partial parameter unpacking"
                        );
                    }
                }
                Err(err) => {
                    warnings.add(err.into());
                    // fine to continue with partially non-decoded arguments/return values
                }
            }
        }

        let mut formatter = IntelFormatter::new();
        for insn in insns {
            // Temporary abstract registers
            //    These are used in the mil program to compute 'small stuff' (memory
            //    offsets, some arithmetic).  Never reused across different
            //    instructions.  "Generated" via self.pb (tmp_gen, tmp_reset)
            self.pb.tmp_reset();
            self.pb.set_input_addr(insn.ip());

            let mut output = String::new();
            formatter.format(&insn, &mut output);
            let _insn_span = span!(Level::TRACE, "translating instruction", insn = output);
            let _insn_span_entered = _insn_span.enter();

            use iced_x86::Mnemonic as M;
            match insn.mnemonic() {
                /*
                    To be implemented:
                    * movaps Rxmm, Mem
                    * movaps Rxmm, Rxmm
                    * movups Mem, Rxmm
                    * movss  Rxmm, Mem
                */
                M::Nop => {}

                M::Push => {
                    assert_eq!(insn.op_count(), 1);

                    let (value, sz) = self.emit_read(&insn, 0);

                    self.emit(
                        Self::RSP,
                        mil::Insn::ArithK(mil::ArithOp::Add, Self::RSP, -(sz as i64)),
                    );
                    let v0 = self.pb.tmp_gen();
                    self.emit(
                        v0,
                        mil::Insn::StoreMem {
                            addr: Self::RSP,
                            value,
                        },
                    );
                }
                M::Pop => {
                    assert_eq!(insn.op_count(), 1);

                    let v0 = self.pb.tmp_gen();

                    let sz = Self::op_size(&insn, 0);
                    if sz != 8 && sz != 2 {
                        panic!("assertion failed: pop dest size must be either 8 or 2 bytes");
                    }
                    self.emit(
                        v0,
                        mil::Insn::LoadMem {
                            addr: Self::RSP,
                            size: sz.into(),
                        },
                    );

                    self.emit_write(&insn, 0, v0, sz);
                    self.emit(
                        Self::RSP,
                        mil::Insn::ArithK(mil::ArithOp::Add, Self::RSP, sz as i64),
                    );
                }
                M::Leave => {
                    self.emit(Self::RSP, mil::Insn::Get(Self::RBP));
                    self.emit(
                        Self::RBP,
                        mil::Insn::LoadMem {
                            addr: Self::RSP,
                            size: 8,
                        },
                    );
                    self.emit(
                        Self::RSP,
                        mil::Insn::ArithK(mil::ArithOp::Add, Self::RSP, 8),
                    );
                }
                M::Ret => {
                    let ret_val = func_ty
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("no function type"))
                        .and_then(|func_ty| {
                            callconv::pack_return_value(&mut self, func_ty.return_tyid)
                                .context("decoding return value")
                        })
                        .or_warn(&mut warnings)
                        .unwrap_or(Self::RAX);

                    let v0 = self.pb.tmp_gen();
                    self.emit(v0, mil::Insn::SetReturnValue(ret_val));
                    self.emit(v0, mil::Insn::Control(Control::Ret));
                }

                // assuming that the instruction is correct and correctly
                // decoded by iced_x86, the same code should serve all these
                // variants of mov
                //
                // Note that movaps, movapd are not the same as their AVX
                // versions vmovaps, vmovapd
                // `movdqa`/`movdqu` are MIL-identical to `movaps`/`movups`:
                // all four move 16 bytes between xmm/mem.  In real hardware the
                // only difference is alignment checking — `movdqa`/`movaps`/
                // `movapd` raise `#GP` on a misaligned address, while `movdqu`/
                // `movups`/`movupd` tolerate any alignment.  This decompiler does
                // not model alignment faults, so they collapse into one arm.
                M::Mov | M::Movaps | M::Movapd | M::Movups | M::Movdqa | M::Movdqu | M::Movzx => {
                    // movzx is implicitly handled by emit_write.
                    // the other opcodes are requried to have same-size source
                    // and destination operands
                    let (value, sz) = self.emit_read(&insn, 1);
                    self.emit_write(&insn, 0, value, sz);
                }
                M::Movsd => {
                    // correct because the set of valid/encodable x86_64
                    // instruction is already restricted to valid cases
                    let (value, _sz) = self.emit_read(&insn, 1);
                    self.emit(
                        value,
                        mil::Insn::Part {
                            src: value,
                            offset: 0,
                            size: 8,
                        },
                    );
                    self.emit_write(&insn, 0, value, 8);
                }
                M::Movd | M::Movss => {
                    let (value, _sz) = self.emit_read(&insn, 1);
                    self.emit(
                        value,
                        mil::Insn::Part {
                            src: value,
                            offset: 0,
                            size: 4,
                        },
                    );
                    self.emit_write(&insn, 0, value, 4);
                }

                // Sign-extending moves.
                //
                // `movsxd` is always 32->64.  `movsx` is 8/16 -> 16/32/64.
                // Unlike `movzx` (which is implicitly zero-extended by
                // `emit_write`'s `extend_zero`), these need an explicit
                // `Widen { sign: true }` to the destination size before the
                // write, so that `emit_write` does not zero-extend.
                //
                // Note on the 32-bit-destination case (e.g. `movsx eax, …`):
                // the sign-widen here targets 4 bytes, then
                // `emit_write_machine_reg` applies its 32->64 zero-extend.
                // The two widens compose correctly (the low 32 bits already
                // hold the sign-extended result, so zeroing bits 32..63 yields
                // the x86_64-defined 32-bit-result-zero-extended-to-64). Do
                // not remove either widen without re-checking this.
                M::Movsxd | M::Movsx => {
                    let (value, src_sz) = self.emit_read(&insn, 1);
                    let dest_sz = Self::op_size(&insn, 0);
                    assert!(
                        dest_sz > src_sz,
                        "movsx/movsxd: destination must be larger than source"
                    );
                    self.emit(
                        value,
                        mil::Insn::Widen {
                            reg: value,
                            target_size: dest_sz,
                            sign: true,
                        },
                    );
                    self.emit_write(&insn, 0, value, dest_sz);
                }

                // `movq` transfers 8 bytes (the low qword of an xmm source).
                // Unlike `movsd` (which preserves the upper bits of the
                // destination xmm), `movq xmm, …` zero-extends the upper 64
                // bits of the 16-byte xmm.  The zmm bits above 128 are
                // preserved, consistent with the rest of the decoder's
                // legacy-SSE upper-bit approximation.
                M::Movq => {
                    let (value, _sz) = self.emit_read(&insn, 1);
                    self.emit(
                        value,
                        mil::Insn::Part {
                            src: value,
                            offset: 0,
                            size: 8,
                        },
                    );
                    if matches!(insn.op0_kind(), OpKind::Register) && insn.op0_register().is_xmm() {
                        let full = Importer::xlat_reg(insn.op0_register().full_register());
                        let zero8 = self.pb.tmp_gen();
                        self.emit(zero8, mil::Insn::Int { value: 0, size: 8 });
                        let hi48 = self.pb.tmp_gen();
                        self.emit(
                            hi48,
                            mil::Insn::Part {
                                src: full,
                                offset: 16,
                                size: 48,
                            },
                        );
                        let low16 = self.pb.tmp_gen();
                        self.emit(
                            low16,
                            mil::Insn::Concat {
                                lo: value,
                                hi: zero8,
                            },
                        );
                        self.emit(
                            full,
                            mil::Insn::Concat {
                                lo: low16,
                                hi: hi48,
                            },
                        );
                    } else {
                        // movq r64/m64, src: plain 8-byte move.
                        self.emit_write(&insn, 0, value, 8);
                    }
                }

                // `movhps xmm, [mem]` loads 8 bytes into the high qword
                // (offset 8) of the xmm.  `movhps [mem], xmm` stores the
                // high qword of the xmm to memory.
                M::Movhps => {
                    if matches!(insn.op0_kind(), OpKind::Register) {
                        let (value, _sz) = self.emit_read(&insn, 1);
                        self.emit(
                            value,
                            mil::Insn::Part {
                                src: value,
                                offset: 0,
                                size: 8,
                            },
                        );
                        let full = Importer::xlat_reg(insn.op0_register().full_register());
                        let lo8 = self.pb.tmp_gen();
                        self.emit(
                            lo8,
                            mil::Insn::Part {
                                src: full,
                                offset: 0,
                                size: 8,
                            },
                        );
                        let hi48 = self.pb.tmp_gen();
                        self.emit(
                            hi48,
                            mil::Insn::Part {
                                src: full,
                                offset: 16,
                                size: 48,
                            },
                        );
                        let low16 = self.pb.tmp_gen();
                        self.emit(low16, mil::Insn::Concat { lo: lo8, hi: value });
                        self.emit(
                            full,
                            mil::Insn::Concat {
                                lo: low16,
                                hi: hi48,
                            },
                        );
                    } else {
                        // movhps [mem], xmm: store the high qword of the xmm.
                        assert_eq!(
                            insn.op1_kind(),
                            OpKind::Register,
                            "movhps store: source must be a register"
                        );
                        let src_full = Importer::xlat_reg(insn.op1_register().full_register());
                        let hi8 = self.pb.tmp_gen();
                        self.emit(
                            hi8,
                            mil::Insn::Part {
                                src: src_full,
                                offset: 8,
                                size: 8,
                            },
                        );
                        self.emit_write(&insn, 0, hi8, 8);
                    }
                }

                // `movhlps xmm1, xmm2` copies the high qword of xmm2 into
                // the low qword of xmm1 (the upper bits of xmm1 are
                // preserved, which `emit_write`'s `Concat` path does for us).
                M::Movhlps => {
                    assert_eq!(
                        insn.op1_kind(),
                        OpKind::Register,
                        "movhlps: source must be a register"
                    );
                    let src_full = Importer::xlat_reg(insn.op1_register().full_register());
                    let hi8 = self.pb.tmp_gen();
                    self.emit(
                        hi8,
                        mil::Insn::Part {
                            src: src_full,
                            offset: 8,
                            size: 8,
                        },
                    );
                    self.emit_write(&insn, 0, hi8, 8);
                }

                M::Add => {
                    let (a, a_sz) = self.emit_read(&insn, 0);
                    let (b, b_sz) = self.emit_read(&insn, 1);
                    self.emit_arith(a, a_sz, b, b_sz, mil::ArithOp::Add);
                    self.emit_write(&insn, 0, a, a_sz);
                    self.emit_set_flags_arith(a);
                }
                M::Sub => {
                    let (a, a_sz) = self.emit_read(&insn, 0);
                    let (b, b_sz) = self.emit_read(&insn, 1);
                    assert_eq!(a_sz, b_sz, "sub: operands must be the same size");
                    self.emit_arith(a, a_sz, b, b_sz, mil::ArithOp::Sub);
                    self.emit_write(&insn, 0, a, a_sz);
                    self.emit_set_flags_arith(a);
                }
                M::Xor | M::Pxor => {
                    let (a, a_sz) = self.emit_read(&insn, 0);
                    let (b, b_sz) = self.emit_read(&insn, 1);
                    self.emit_bit_op(a_sz, b_sz, a, b, mil::ArithOp::BitXor, insn);
                }
                M::Or => {
                    let (a, a_sz) = self.emit_read(&insn, 0);
                    let (b, b_sz) = self.emit_read(&insn, 1);
                    self.emit_bit_op(a_sz, b_sz, a, b, mil::ArithOp::BitOr, insn);
                }
                M::And => {
                    let (a, a_sz) = self.emit_read(&insn, 0);
                    let (b, b_sz) = self.emit_read(&insn, 1);
                    self.emit_bit_op(a_sz, b_sz, a, b, mil::ArithOp::BitAnd, insn);
                }

                M::Inc => {
                    let (a, a_sz) = self.emit_read(&insn, 0);
                    self.emit(a, mil::Insn::ArithK(mil::ArithOp::Add, a, 1));
                    self.emit_write(&insn, 0, a, a_sz);

                    self.emit(Self::OF, mil::Insn::False);
                    self.emit(Self::CF, mil::Insn::False);
                    // TODO implement: AF
                    self.emit(Self::SF, mil::Insn::SignOf(a));
                    self.emit(Self::ZF, mil::Insn::IsZero(a));
                    let v0 = self.pb.tmp_gen();
                    self.emit(
                        v0,
                        mil::Insn::Part {
                            src: a,
                            offset: 0,
                            size: 1,
                        },
                    );
                    self.emit(Self::PF, mil::Insn::Parity(v0));
                }

                M::Test => {
                    let (a, a_sz) = self.emit_read(&insn, 0);
                    let (b, b_sz) = self.emit_read(&insn, 1);
                    let v0 = self.pb.tmp_gen();
                    self.emit_arith(a, a_sz, b, b_sz, mil::ArithOp::BitAnd);
                    self.emit(Self::SF, mil::Insn::SignOf(a));
                    self.emit(Self::ZF, mil::Insn::IsZero(a));
                    self.emit(
                        v0,
                        mil::Insn::Part {
                            src: a,
                            offset: 0,
                            size: 1,
                        },
                    );
                    self.emit(Self::PF, mil::Insn::Parity(a));
                    self.emit(Self::CF, mil::Insn::False);
                    self.emit(Self::OF, mil::Insn::False);
                }

                M::Cmp => {
                    let (a, a_sz) = self.emit_read(&insn, 0);
                    let (b, b_sz) = self.emit_read(&insn, 1);
                    assert_eq!(a_sz, b_sz, "cmp: operands must be the same size");
                    // just put the result in a tmp reg, then ignore it (other than for the
                    // flags)
                    self.emit_arith(a, a_sz, b, b_sz, mil::ArithOp::Sub);
                    self.emit_set_flags_arith(a);
                }

                M::Lea => match insn.op1_kind() {
                    OpKind::Memory => {
                        let v0 = self.pb.tmp_gen();
                        self.emit_compute_address_into(&insn, v0);
                        assert_eq!(insn.op0_kind(), OpKind::Register);
                        let value_size = insn.op0_register().size().try_into().unwrap();
                        self.emit_write(&insn, 0, v0, value_size);
                    }
                    OpKind::MemorySegSI
                    | OpKind::MemorySegESI
                    | OpKind::MemorySegRSI
                    | OpKind::MemorySegDI
                    | OpKind::MemorySegEDI
                    | OpKind::MemorySegRDI
                    | OpKind::MemoryESDI
                    | OpKind::MemoryESEDI
                    | OpKind::MemoryESRDI => {
                        panic!("lea: segment-relative memory operands are not yet supported")
                    }
                    _ => {
                        panic!("lea: invalid operand: second operand must be of type memory")
                    }
                },

                M::Shl => {
                    self.emit_shift(insn, mil::ArithOp::Shl);
                }
                M::Shr => {
                    self.emit_shift(insn, mil::ArithOp::Shr);
                }
                M::Sar => {
                    self.emit_shift(insn, mil::ArithOp::Sar);
                }
                M::Rol => {
                    self.emit_shift(insn, mil::ArithOp::Rol);
                }
                M::Ror => {
                    self.emit_shift(insn, mil::ArithOp::Ror);
                }

                // Neg: dest = 0 - src; flags from the subtraction result.
                // CF = 1 iff src != 0 (which is the borrow from 0 - src).
                M::Neg => {
                    let (a, sz) = self.emit_read(&insn, 0);
                    let zero = self.pb.tmp_gen();
                    self.emit(zero, mil::Insn::Int { value: 0, size: sz });
                    self.emit(zero, mil::Insn::Arith(mil::ArithOp::Sub, zero, a));
                    self.emit_write(&insn, 0, zero, sz);
                    self.emit_set_flags_arith(zero);
                }

                // Not: bitwise complement (one's complement negation).
                // dest = src ^ all_ones.  Does not affect any flags.
                M::Not => {
                    let (a, sz) = self.emit_read(&insn, 0);
                    let all_ones = self.pb.tmp_gen();
                    self.emit(
                        all_ones,
                        mil::Insn::Int {
                            value: -1,
                            size: sz,
                        },
                    );
                    self.emit(a, mil::Insn::Arith(mil::ArithOp::BitXor, a, all_ones));
                    self.emit_write(&insn, 0, a, sz);
                    self.emit(Self::OF, mil::Insn::False);
                    self.emit(Self::CF, mil::Insn::False);
                    self.emit(Self::SF, mil::Insn::UndefinedBool);
                    self.emit(Self::ZF, mil::Insn::UndefinedBool);
                    self.emit(Self::AF, mil::Insn::UndefinedBool);
                    self.emit(Self::PF, mil::Insn::UndefinedBool);
                }

                // Adc: add with carry.
                //   t = a + b
                //   cf = Widen{CF, sz, false}
                //   dest = t + cf
                // Flags from final result (approximation of true carry chain).
                M::Adc => {
                    let (a, sz) = self.emit_read(&insn, 0);
                    let (b, b_sz) = self.emit_read(&insn, 1);
                    assert_eq!(sz, b_sz);
                    let t = self.pb.tmp_gen();
                    self.emit(t, mil::Insn::Arith(mil::ArithOp::Add, a, b));
                    let cf = self.pb.tmp_gen();
                    self.emit(
                        cf,
                        mil::Insn::Widen {
                            reg: Self::CF,
                            target_size: sz,
                            sign: false,
                        },
                    );
                    self.emit(t, mil::Insn::Arith(mil::ArithOp::Add, t, cf));
                    self.emit_write(&insn, 0, t, sz);
                    self.emit_set_flags_arith(t);
                }

                // Sbb: subtract with borrow.
                //   t = a - b
                //   cf = Widen{CF, sz, false}
                //   dest = t - cf
                M::Sbb => {
                    let (a, sz) = self.emit_read(&insn, 0);
                    let (b, b_sz) = self.emit_read(&insn, 1);
                    assert_eq!(sz, b_sz);
                    let t = self.pb.tmp_gen();
                    self.emit(t, mil::Insn::Arith(mil::ArithOp::Sub, a, b));
                    let cf = self.pb.tmp_gen();
                    self.emit(
                        cf,
                        mil::Insn::Widen {
                            reg: Self::CF,
                            target_size: sz,
                            sign: false,
                        },
                    );
                    self.emit(t, mil::Insn::Arith(mil::ArithOp::Sub, t, cf));
                    self.emit_write(&insn, 0, t, sz);
                    self.emit_set_flags_arith(t);
                }

                // Bt: bit test.  CF = (src >> bit_index) & 1.  No dest write.
                M::Bt => {
                    let (a, _sz) = self.emit_read(&insn, 0);
                    let (n, _n_sz) = self.emit_read(&insn, 1);
                    let shifted = self.pb.tmp_gen();
                    self.emit(shifted, mil::Insn::Arith(mil::ArithOp::Shr, a, n));
                    let lowbit = self.pb.tmp_gen();
                    self.emit(
                        lowbit,
                        mil::Insn::Part {
                            src: shifted,
                            offset: 0,
                            size: 1,
                        },
                    );
                    let is_zero = self.pb.tmp_gen();
                    self.emit(is_zero, mil::Insn::IsZero(lowbit));
                    self.emit(Self::CF, mil::Insn::Not(is_zero));
                    self.emit(Self::OF, mil::Insn::UndefinedBool);
                    self.emit(Self::SF, mil::Insn::UndefinedBool);
                    self.emit(Self::ZF, mil::Insn::UndefinedBool);
                    self.emit(Self::AF, mil::Insn::UndefinedBool);
                    self.emit(Self::PF, mil::Insn::UndefinedBool);
                }

                // Btr: bit test and reset.  CF from original src, then dest = src & ~(1 << n).
                M::Btr => {
                    let (a, sz) = self.emit_read(&insn, 0);
                    let (n, n_sz) = self.emit_read(&insn, 1);
                    let shifted = self.pb.tmp_gen();
                    self.emit(shifted, mil::Insn::Arith(mil::ArithOp::Shr, a, n));
                    let lowbit = self.pb.tmp_gen();
                    self.emit(
                        lowbit,
                        mil::Insn::Part {
                            src: shifted,
                            offset: 0,
                            size: 1,
                        },
                    );
                    let is_zero = self.pb.tmp_gen();
                    self.emit(is_zero, mil::Insn::IsZero(lowbit));
                    self.emit(Self::CF, mil::Insn::Not(is_zero));
                    // mask = 1 << n
                    let mask = self.pb.tmp_gen();
                    self.emit(mask, mil::Insn::Int { value: 1, size: sz });
                    self.emit_arith(mask, sz, n, n_sz, mil::ArithOp::Shl);
                    // not_mask = all_ones ^ mask
                    let not_mask = self.pb.tmp_gen();
                    self.emit(
                        not_mask,
                        mil::Insn::Int {
                            value: -1,
                            size: sz,
                        },
                    );
                    self.emit(
                        not_mask,
                        mil::Insn::Arith(mil::ArithOp::BitXor, not_mask, mask),
                    );
                    // dest = a & not_mask
                    self.emit(a, mil::Insn::Arith(mil::ArithOp::BitAnd, a, not_mask));
                    self.emit_write(&insn, 0, a, sz);
                    self.emit(Self::OF, mil::Insn::UndefinedBool);
                    self.emit(Self::SF, mil::Insn::UndefinedBool);
                    self.emit(Self::ZF, mil::Insn::UndefinedBool);
                    self.emit(Self::AF, mil::Insn::UndefinedBool);
                    self.emit(Self::PF, mil::Insn::UndefinedBool);
                }

                // Bts: bit test and set.  CF from original src, then dest = src | (1 << n).
                M::Bts => {
                    let (a, sz) = self.emit_read(&insn, 0);
                    let (n, n_sz) = self.emit_read(&insn, 1);
                    let shifted = self.pb.tmp_gen();
                    self.emit(shifted, mil::Insn::Arith(mil::ArithOp::Shr, a, n));
                    let lowbit = self.pb.tmp_gen();
                    self.emit(
                        lowbit,
                        mil::Insn::Part {
                            src: shifted,
                            offset: 0,
                            size: 1,
                        },
                    );
                    let is_zero = self.pb.tmp_gen();
                    self.emit(is_zero, mil::Insn::IsZero(lowbit));
                    self.emit(Self::CF, mil::Insn::Not(is_zero));
                    // mask = 1 << n
                    let mask = self.pb.tmp_gen();
                    self.emit(mask, mil::Insn::Int { value: 1, size: sz });
                    self.emit_arith(mask, sz, n, n_sz, mil::ArithOp::Shl);
                    // dest = a | mask
                    self.emit(a, mil::Insn::Arith(mil::ArithOp::BitOr, a, mask));
                    self.emit_write(&insn, 0, a, sz);
                    self.emit(Self::OF, mil::Insn::UndefinedBool);
                    self.emit(Self::SF, mil::Insn::UndefinedBool);
                    self.emit(Self::ZF, mil::Insn::UndefinedBool);
                    self.emit(Self::AF, mil::Insn::UndefinedBool);
                    self.emit(Self::PF, mil::Insn::UndefinedBool);
                }

                // Mul (1-op): unsigned multiply.  Same as the existing 1-op Imul
                // (signed) but uses `ArithOp::Mul` (which already produces a 2N-bit
                // result).  The 2N-bit product is split into hi/lo and written to
                // RDX:RAX (or EDX:EAX, etc.).
                M::Mul => {
                    match insn.op_count() {
                        1 => {
                            let src_b = self.emit_read_value(&insn, 0);
                            let a_size = match insn.op0_kind() {
                                OpKind::Register => insn.op0_register().size(),
                                OpKind::Memory => insn.memory_size().size(),
                                kind => panic!("imul: invalid operand: {:?}", kind),
                            };
                            let (dest_hi, dest_lo) = match a_size {
                                1 => (Register::AH, Register::AL),
                                2 => (Register::DX, Register::AX),
                                4 => (Register::EDX, Register::EAX),
                                8 => (Register::RDX, Register::RAX),
                                _ => panic!("imul: invalid operand size: {}", a_size),
                            };

                            let v0 = self.emit_read_machine_reg(dest_lo);
                            self.emit(v0, mil::Insn::Arith(mil::ArithOp::Mul, v0, src_b));
                            self.emit(Self::OF, mil::Insn::OverflowOf(v0));
                            self.emit(Self::CF, mil::Insn::Get(Self::OF));
                            self.emit(Self::SF, mil::Insn::UndefinedBool);
                            self.emit(Self::ZF, mil::Insn::UndefinedBool);
                            self.emit(Self::AF, mil::Insn::UndefinedBool);
                            self.emit(Self::PF, mil::Insn::UndefinedBool);

                            let a_size = a_size.try_into().unwrap();
                            let result_hi = self.pb.tmp_gen();
                            self.emit(
                                result_hi,
                                mil::Insn::Part {
                                    src: v0,
                                    offset: a_size,
                                    size: a_size,
                                },
                            );
                            self.emit_write_machine_reg(dest_hi, a_size, result_hi);
                            self.emit(
                                v0,
                                mil::Insn::Part {
                                    src: v0,
                                    offset: 0,
                                    size: a_size,
                                },
                            );
                            self.emit_write_machine_reg(dest_lo, a_size, v0);
                        }

                        op_count @ (2 | 3) => {
                            let a = self.emit_read_value(&insn, 0);
                            let result_size = insn
                                .try_op_register(0)
                                .expect("imul: register as first operand")
                                .size()
                                .try_into()
                                .unwrap();
                            let b = self.emit_read_value(&insn, 1);
                            let result = self.pb.tmp_gen();
                            self.emit(result, mil::Insn::Arith(mil::ArithOp::Mul, a, b));

                            if op_count == 3 {
                                // this panics if the 3rd operand is not an
                                // immediate, but there is no such case in
                                // x86_64. if this actually panics in the wild,
                                // it's probably a bug in iced_x86, but I bet
                                // it won't.
                                let imm64 = insn.immediate(2) as i64;
                                let k = self.pb.tmp_gen();
                                self.emit(
                                    k,
                                    mil::Insn::Int {
                                        value: imm64,
                                        size: 8,
                                    },
                                );
                                self.emit(result, mil::Insn::Arith(mil::ArithOp::Mul, result, k));
                            }

                            self.emit_write(&insn, 0, result, result_size);
                            self.emit(Self::OF, mil::Insn::OverflowOf(result));
                            self.emit(Self::CF, mil::Insn::Get(Self::OF));
                            self.emit(Self::SF, mil::Insn::UndefinedBool);
                            self.emit(Self::ZF, mil::Insn::UndefinedBool);
                            self.emit(Self::AF, mil::Insn::UndefinedBool);
                            self.emit(Self::PF, mil::Insn::UndefinedBool);
                        }

                        other => panic!("mul: invalid operands count: {}", other),
                    };
                }

                M::Call => {
                    let (callee, sz) = self.emit_read(&insn, 0);
                    assert_eq!(
                        sz, 8,
                        "invalid call instruction: operand must be 8 bytes, not {}",
                        sz
                    );

                    let v1 = self.tmp_gen();
                    match self.resolve_call(&insn)? {
                        Some((subr_tyid, param_values)) => {
                            // .unwrap(): resolve_call() is supposed to check this already
                            let subr_ty =
                                check_subroutine_type(self.rtx.read(), subr_tyid).unwrap();
                            let return_tyid = subr_ty.return_tyid;

                            event!(Level::TRACE, ?subr_tyid, ?param_values, "resolved call");

                            let ret_size = self.rtx.read().bytes_size(return_tyid).ok().flatten().unwrap_or_else(|| {
                                event!(Level::WARN, ret_reg = ?v1, "call return type unavailable; approximated return LLType to 8 bytes");
                                8
                            });
                            self.emit_call(callee, param_values, v1, mil::LLType::Bytes(ret_size));
                            let callee_ndx = self.last_index_of_value(callee).unwrap();
                            self.set_value_type(callee_ndx, subr_tyid);

                            if let Err(err) =
                                callconv::unpack_return_value(&mut self, return_tyid, v1)
                            {
                                event!(Level::ERROR, ?err, "could not unpack return value");
                            }
                        }
                        None => {
                            // just a dumb approximation of a likely case
                            event!(Level::ERROR, "call unresolved, using a default fallback");
                            let param_values = vec![Self::RDI, Self::RSI, Self::RDX, Self::RCX];
                            self.emit_call(callee, param_values, v1, mil::LLType::Bytes(8));
                            self.emit(v1, mil::Insn::Get(Self::RAX));
                        }
                    }
                }

                //
                // Jumps
                //
                M::Jmp => {
                    // refactor with emit_jmpif?
                    let v0 = self.pb.tmp_gen();
                    match insn.op0_kind() {
                        OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => {
                            let target = insn.near_branch_target();
                            self.emit(v0, mil::Insn::Control(Control::JmpExt(target)));
                        }
                        _ => {
                            let addr_reg = self.emit_compute_address(&insn);
                            self.emit(addr_reg, mil::Insn::SetJumpTarget(addr_reg));
                            self.emit(v0, mil::Insn::Control(Control::JmpIndirect));
                        }
                    }
                }
                M::Ja => {
                    let cond = self.emit_cmp_a();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Je => {
                    let cond = self.emit_cmp_e();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jne => {
                    let cond = self.emit_cmp_ne();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jb => {
                    let cond = self.emit_cmp_b();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jl => {
                    let cond = self.emit_cmp_l();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jle => {
                    let cond = self.emit_cmp_le();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jae => {
                    let cond = self.emit_cmp_ae();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jbe => {
                    let cond = self.emit_cmp_be();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jcxz => {
                    let cond = self.emit_cmp_cxz();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jecxz => {
                    let cond = self.emit_cmp_ecxz();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jg => {
                    let cond = self.emit_cmp_g();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jge => {
                    let cond = self.emit_cmp_ge();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jno => {
                    let cond = self.emit_cmp_no();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jnp => {
                    let cond = self.emit_cmp_np();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jns => {
                    let cond = self.emit_cmp_ns();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jo => {
                    let cond = self.emit_cmp_o();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jp => {
                    let cond = self.emit_cmp_p();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Jrcxz => {
                    let cond = self.emit_cmp_rcxz();
                    self.emit_jmpif(insn, 0, cond);
                }
                M::Js => {
                    let cond = self.emit_cmp_s();
                    self.emit_jmpif(insn, 0, cond);
                }

                M::Seta => {
                    let v0 = self.emit_cmp_a();
                    self.emit_flag_to_byte(v0);
                }
                M::Setae => {
                    let v0 = self.emit_cmp_ae();
                    self.emit_flag_to_byte(v0);
                }
                M::Setb => {
                    let v0 = self.emit_cmp_b();
                    self.emit_flag_to_byte(v0);
                }
                M::Setbe => {
                    let v0 = self.emit_cmp_be();
                    self.emit_flag_to_byte(v0);
                }
                M::Sete => {
                    let v0 = self.emit_cmp_e();
                    self.emit_flag_to_byte(v0);
                }
                M::Setg => {
                    let v0 = self.emit_cmp_g();
                    self.emit_flag_to_byte(v0);
                }
                M::Setge => {
                    let v0 = self.emit_cmp_ge();
                    self.emit_flag_to_byte(v0);
                }
                M::Setl => {
                    let v0 = self.emit_cmp_l();
                    self.emit_flag_to_byte(v0);
                }
                M::Setle => {
                    let v0 = self.emit_cmp_le();
                    self.emit_flag_to_byte(v0);
                }
                M::Setne => {
                    let v0 = self.emit_cmp_ne();
                    self.emit_flag_to_byte(v0);
                }
                M::Setno => {
                    let v0 = self.emit_cmp_no();
                    self.emit_flag_to_byte(v0);
                }
                M::Setnp => {
                    let v0 = self.emit_cmp_np();
                    self.emit_flag_to_byte(v0);
                }
                M::Setns => {
                    let v0 = self.emit_cmp_ns();
                    self.emit_flag_to_byte(v0);
                }
                M::Seto => {
                    let v0 = self.emit_cmp_o();
                    self.emit_flag_to_byte(v0);
                }
                M::Setp => {
                    let v0 = self.emit_cmp_p();
                    self.emit_flag_to_byte(v0);
                }

                // Conditional moves (cmovCC dst, src): if the condition is
                // true, dst := src, else dst keeps its old value.  Modeled as
                //   dst := Select { cond, then_val: src, else_val: old_dst }
                // which is a pure (side-effect-free) value selection.  The
                // condition is computed by the same `emit_cmp_*` helpers used
                // by the matching conditional-jump / set families.
                M::Cmova => {
                    self.emit_cmov(&insn, Self::emit_cmp_a);
                }
                M::Cmovae => {
                    self.emit_cmov(&insn, Self::emit_cmp_ae);
                }
                M::Cmovb => {
                    self.emit_cmov(&insn, Self::emit_cmp_b);
                }
                M::Cmovbe => {
                    self.emit_cmov(&insn, Self::emit_cmp_be);
                }
                M::Cmove => {
                    self.emit_cmov(&insn, Self::emit_cmp_e);
                }
                M::Cmovg => {
                    self.emit_cmov(&insn, Self::emit_cmp_g);
                }
                M::Cmovge => {
                    self.emit_cmov(&insn, Self::emit_cmp_ge);
                }
                M::Cmovl => {
                    self.emit_cmov(&insn, Self::emit_cmp_l);
                }
                M::Cmovle => {
                    self.emit_cmov(&insn, Self::emit_cmp_le);
                }
                M::Cmovne => {
                    self.emit_cmov(&insn, Self::emit_cmp_ne);
                }
                M::Cmovns => {
                    self.emit_cmov(&insn, Self::emit_cmp_ns);
                }
                M::Cmovs => {
                    self.emit_cmov(&insn, Self::emit_cmp_s);
                }

                M::Cbw => {
                    let al = self.pb.tmp_gen();
                    self.emit(
                        al,
                        mil::Insn::Part {
                            src: Self::RAX,
                            offset: 0,
                            size: 1,
                        },
                    );
                    self.emit(
                        Self::RAX,
                        mil::Insn::Widen {
                            reg: al,
                            target_size: 2,
                            sign: true,
                        },
                    );
                }
                M::Cwde => {
                    let ax = self.pb.tmp_gen();
                    self.emit(
                        ax,
                        mil::Insn::Part {
                            src: Self::RAX,
                            offset: 0,
                            size: 2,
                        },
                    );
                    self.emit(
                        Self::RAX,
                        mil::Insn::Widen {
                            reg: ax,
                            target_size: 4,
                            sign: true,
                        },
                    );
                }
                M::Cdqe => {
                    let eax = self.pb.tmp_gen();
                    self.emit(
                        eax,
                        mil::Insn::Part {
                            src: Self::RAX,
                            offset: 0,
                            size: 4,
                        },
                    );
                    self.emit(
                        Self::RAX,
                        mil::Insn::Widen {
                            reg: eax,
                            target_size: 8,
                            sign: true,
                        },
                    );
                }

                _ => {
                    let mut output = String::new();
                    formatter.format(&insn, &mut output);
                    let description = format!("unsupported: {}", output);
                    let v0 = self.pb.tmp_gen();
                    self.emit(v0, mil::Insn::NotYetImplemented(description.leak()));
                }
            }
        }

        Ok((self.pb, warnings))
    }

    fn resolve_call(
        &mut self,
        insn: &iced_x86::Instruction,
    ) -> Result<Option<(ty::TypeID, Vec<mil::Reg>)>> {
        let target_tyid = match insn.op0_kind() {
            OpKind::NearBranch64 => {
                let return_pc = insn.next_ip();
                let target = insn.near_branch_target();
                self.rtx
                    .read()
                    .resolve_call(ty::CallSiteKey { return_pc, target })?
            }
            _ => None,
        };
        let Some(subr_tyid) = target_tyid else {
            event!(Level::WARN, "no type hints for this callsite");
            return Ok(None);
        };

        let param_values = match self.pack_params(subr_tyid) {
            Ok(params) => params,
            Err(err) => {
                event!(Level::ERROR, ?err, "could not pack params");
                return Ok(None);
            }
        };

        Ok(Some((subr_tyid, param_values)))
    }

    /// Emit a call whose arguments are stored directly on the `Insn::Call`.
    fn emit_call(
        &mut self,
        callee: mil::Reg,
        param_values: Vec<mil::Reg>,
        ret_reg: mil::Reg,
        ret_ll_type: mil::LLType,
    ) {
        self.emit(
            ret_reg,
            mil::Insn::Call {
                callee,
                args: param_values,
                ret_ll_type,
            },
        );
        self.reset_all_flags();
    }

    fn reset_all_flags(&mut self) {
        self.emit(Self::CF, mil::Insn::UndefinedBool);
        self.emit(Self::PF, mil::Insn::UndefinedBool);
        self.emit(Self::AF, mil::Insn::UndefinedBool);
        self.emit(Self::ZF, mil::Insn::UndefinedBool);
        self.emit(Self::SF, mil::Insn::UndefinedBool);
        self.emit(Self::TF, mil::Insn::UndefinedBool);
        self.emit(Self::IF, mil::Insn::UndefinedBool);
        self.emit(Self::DF, mil::Insn::UndefinedBool);
        self.emit(Self::OF, mil::Insn::UndefinedBool);
    }

    fn pack_params(&mut self, subr_tyid: ty::TypeID) -> Result<Vec<mil::Reg>> {
        let (_report, param_values) =
            callconv::pack_params(self, subr_tyid).context("while applying calling convention")?;
        assert_eq!(_report.ok_count, param_values.len());
        Ok(param_values)
    }

    fn emit_bit_op(
        &mut self,
        a_sz: u16,
        b_sz: u16,
        a: mil::Reg,
        b: mil::Reg,
        arith_op: mil::ArithOp,
        insn: iced_x86::Instruction,
    ) {
        assert_eq!(a_sz, b_sz, "bit op: operands must be the same size");
        self.emit_arith(a, a_sz, b, b_sz, arith_op);
        self.emit_write(&insn, 0, a, a_sz);

        self.emit(Self::OF, mil::Insn::False);
        self.emit(Self::CF, mil::Insn::False);
        // TODO implement: AF
        self.emit(Self::SF, mil::Insn::SignOf(a));
        self.emit(Self::ZF, mil::Insn::IsZero(a));
        let v0 = self.pb.tmp_gen();
        self.emit(
            v0,
            mil::Insn::Part {
                src: a,
                offset: 0,
                size: 1,
            },
        );
        self.emit(Self::PF, mil::Insn::Parity(v0));
    }

    fn emit_arith(&mut self, a: mil::Reg, a_sz: u16, b: mil::Reg, b_sz: u16, op: mil::ArithOp) {
        // integer or xmm/ymm registers
        assert!([1, 2, 4, 8, 16, 32].contains(&a_sz));
        assert!([1, 2, 4, 8, 16, 32].contains(&b_sz));
        let sz = a_sz.max(b_sz);
        self.extend_zero(a, a_sz, sz);
        self.extend_zero(b, b_sz, sz);
        self.emit(a, mil::Insn::Arith(op, a, b));
    }

    fn emit_shift(&mut self, insn: iced_x86::Instruction, arith_op: mil::ArithOp) {
        let (value, sz) = self.emit_read(&insn, 0);
        let (bits_count, bits_count_size) = self.emit_read(&insn, 1);
        self.emit_arith(value, sz, bits_count, bits_count_size, arith_op);
        self.emit_write(&insn, 0, value, sz);

        // TODO implement flag cahnges: CF, OF
        // (these are more complex than others, as they depend on the exact value
        // of the bit count)
        self.emit(Self::SF, mil::Insn::SignOf(value));
        self.emit(Self::ZF, mil::Insn::IsZero(value));
        let v0 = self.pb.tmp_gen();
        self.emit(
            v0,
            mil::Insn::Part {
                src: value,
                offset: 0,
                size: 1,
            },
        );
        self.emit(Self::PF, mil::Insn::Parity(v0));
        // ignored: AF
    }

    fn emit_jmpif(&mut self, insn: iced_x86::Instruction, op_ndx: u32, cond: mil::Reg) {
        match insn.op_kind(op_ndx) {
            OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => {
                let target = insn.near_branch_target();
                let v0 = self.pb.tmp_gen();
                self.emit(v0, mil::Insn::SetJumpCondition(cond));
                self.emit(v0, mil::Insn::Control(Control::JmpExtIf(target)));
            }
            _ => {
                todo!("indirect jmpif");
            }
        }
    }

    /// Emit Insns that set the x86_64 flags as it happens after an arithmetic x86_64
    /// instruction.
    ///
    /// Argument `a` is the destination register of the arthmetic operation.  It is both used
    /// to refer to the result, or to otherwise identify the arithmetic operation itself.
    fn emit_set_flags_arith(&mut self, a: mil::Reg) {
        self.emit(Self::OF, mil::Insn::OverflowOf(a));
        self.emit(Self::CF, mil::Insn::CarryOf(a));
        // TODO implement: AF
        self.emit(Self::SF, mil::Insn::SignOf(a));
        self.emit(Self::ZF, mil::Insn::IsZero(a));
        let v0 = self.pb.tmp_gen();
        self.emit(
            v0,
            mil::Insn::Part {
                src: a,
                offset: 0,
                size: 1,
            },
        );
        self.emit(Self::PF, mil::Insn::Parity(v0));
    }

    fn op_size(insn: &iced_x86::Instruction, op_ndx: u32) -> u16 {
        match insn.op_kind(op_ndx) {
            OpKind::Register => insn.op_register(op_ndx).size().try_into().unwrap(),
            OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => 8,
            OpKind::FarBranch16 | OpKind::FarBranch32 => {
                todo!("not supported: far branch operands")
            }
            OpKind::Immediate8 => 1,
            OpKind::Immediate8_2nd => 1,
            OpKind::Immediate16 => 2,
            OpKind::Immediate32 => 4,
            OpKind::Immediate64 => 8,
            // these are sign-extended (to different sizes). the conversion to u64 keeps the same bits,
            // so I think we don't lose any info (semantic or otherwise)
            OpKind::Immediate8to16 => 2,
            OpKind::Immediate8to32 => 4,
            OpKind::Immediate8to64 => 8,
            OpKind::Immediate32to64 => 8,

            OpKind::MemorySegSI
            | OpKind::MemorySegESI
            | OpKind::MemorySegRSI
            | OpKind::MemorySegDI
            | OpKind::MemorySegEDI
            | OpKind::MemorySegRDI
            | OpKind::MemoryESDI
            | OpKind::MemoryESEDI
            | OpKind::MemoryESRDI
            | OpKind::Memory => insn.memory_size().size().try_into().unwrap(),
        }
    }

    /// Emit MIL instructions for reading the given operand.
    ///
    /// The operand to be read is taken in as an instruction and an index; the operand to be
    /// read is the nth operand of the instruction.
    ///
    /// Return value: the register that stores the read value (in the MIL text), and the
    /// value's size in bytes (either 1, 2, 4, or 8).
    fn emit_read_value(&mut self, insn: &iced_x86::Instruction, op_ndx: u32) -> mil::Reg {
        let v0 = self.pb.tmp_gen();

        match insn.op_kind(op_ndx) {
            OpKind::Register => {
                let reg = insn.op_register(op_ndx);
                self.emit_read_machine_reg(reg)
            }
            OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => {
                self.emit(
                    v0,
                    mil::Insn::Int {
                        value: insn.near_branch_target() as i64,
                        size: 8,
                    },
                );
                v0
            }
            OpKind::FarBranch16 | OpKind::FarBranch32 => {
                todo!("not supported: far branch operands")
            }

            OpKind::Immediate8 => {
                self.emit(
                    v0,
                    mil::Insn::Int {
                        value: insn.immediate8() as _,
                        size: 1,
                    },
                );
                v0
            }
            OpKind::Immediate8_2nd => {
                self.emit(
                    v0,
                    mil::Insn::Int {
                        value: insn.immediate8_2nd() as _,
                        size: 1,
                    },
                );
                v0
            }
            OpKind::Immediate16 => {
                self.emit(
                    v0,
                    mil::Insn::Int {
                        value: insn.immediate16() as _,
                        size: 2,
                    },
                );
                v0
            }
            OpKind::Immediate32 => {
                self.emit(
                    v0,
                    mil::Insn::Int {
                        value: insn.immediate32() as _,
                        size: 4,
                    },
                );
                v0
            }
            OpKind::Immediate64 => {
                self.emit(
                    v0,
                    mil::Insn::Int {
                        value: insn.immediate64() as _,
                        size: 8,
                    },
                );
                v0
            }
            OpKind::Immediate8to16 => {
                self.emit(
                    v0,
                    mil::Insn::Int {
                        value: insn.immediate8to16() as _,
                        size: 2,
                    },
                );
                v0
            }
            OpKind::Immediate8to32 => {
                self.emit(
                    v0,
                    mil::Insn::Int {
                        value: insn.immediate8to32() as _,
                        size: 4,
                    },
                );
                v0
            }
            OpKind::Immediate8to64 => {
                self.emit(
                    v0,
                    mil::Insn::Int {
                        value: insn.immediate8to64() as _,
                        size: 8,
                    },
                );
                v0
            }
            OpKind::Immediate32to64 => {
                self.emit(
                    v0,
                    mil::Insn::Int {
                        value: insn.immediate32to64() as _,
                        size: 8,
                    },
                );
                v0
            }

            OpKind::MemorySegSI
            | OpKind::MemorySegESI
            | OpKind::MemorySegRSI
            | OpKind::MemorySegDI
            | OpKind::MemorySegEDI
            | OpKind::MemorySegRDI
            | OpKind::MemoryESDI
            | OpKind::MemoryESEDI
            | OpKind::MemoryESRDI => {
                self.emit(
                    v0,
                    mil::Insn::NotYetImplemented("segment-relative memory operands"),
                );
                v0
            }

            OpKind::Memory => {
                use iced_x86::MemorySize;

                let addr = self.emit_compute_address(insn);

                // we're ignoring what's in the memory structure (for non-Offset
                // size types). assuming that the rest of the assembly handles
                // it correctly, we should be able to recover the correct
                // information later anyway.
                let memory_size = insn.memory_size();
                self.emit(
                    v0,
                    mil::Insn::LoadMem {
                        addr,
                        size: memory_size.size().try_into().unwrap(),
                    },
                );

                match memory_size {
                    MemorySize::WordOffset => {
                        self.emit(v0, mil::Insn::LoadMem { addr: v0, size: 2 });
                    }
                    MemorySize::DwordOffset => {
                        self.emit(v0, mil::Insn::LoadMem { addr: v0, size: 4 });
                    }
                    MemorySize::QwordOffset => {
                        self.emit(v0, mil::Insn::LoadMem { addr: v0, size: 8 });
                    }
                    _ => {}
                }

                v0
            }
        }
    }

    /// Read a register of any size, emitting mil::Insn::Part as necessary
    fn emit_read_machine_reg(&mut self, reg: Register) -> mil::Reg {
        let full_reg = reg.full_register();
        let value = Importer::xlat_reg(full_reg);
        if reg == full_reg {
            value
        } else {
            let dest = self.pb.tmp_gen();
            self.emit(
                dest,
                mil::Insn::Part {
                    src: value,
                    offset: 0,
                    size: reg.size().try_into().unwrap(),
                },
            );
            dest
        }
    }

    fn emit_read(&mut self, insn: &iced_x86::Instruction, op_ndx: u32) -> (mil::Reg, u16) {
        let value = self.emit_read_value(insn, op_ndx);
        let sz = Self::op_size(insn, op_ndx);
        (value, sz)
    }

    fn emit_write(
        &mut self,
        insn: &iced_x86::Instruction,
        dest_op_ndx: u32,
        value: mil::Reg,
        value_size: u16,
    ) {
        match insn.op_kind(dest_op_ndx) {
            OpKind::Register => {
                let dest = insn.op_register(dest_op_ndx);
                let dest_size: u16 = dest.size().try_into().unwrap();
                self.extend_zero(value, value_size, dest_size);
                self.emit_write_machine_reg(dest, dest_size, value);
            }

            OpKind::NearBranch16
            | OpKind::NearBranch32
            | OpKind::NearBranch64
            | OpKind::FarBranch16
            | OpKind::FarBranch32
            | OpKind::Immediate8
            | OpKind::Immediate8_2nd
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64 => {
                panic!("invalid mov dest operand: {:?}", insn.op0_kind())
            }

            op_kind @ (OpKind::MemorySegSI
            | OpKind::MemorySegESI
            | OpKind::MemorySegRSI
            | OpKind::MemorySegDI
            | OpKind::MemorySegEDI
            | OpKind::MemorySegRDI
            | OpKind::MemoryESDI
            | OpKind::MemoryESEDI
            | OpKind::MemoryESRDI
            | OpKind::Memory) => {
                if value_size as usize != insn.memory_size().size() {
                    event!(
                        Level::ERROR,
                        value_size,
                        memory_size = ?insn.memory_size(),
                        "destination memory operand is not the same size as the value",
                    );
                }

                let addr = self.pb.tmp_gen();

                if op_kind == OpKind::Memory {
                    self.emit_compute_address_into(insn, addr);
                } else {
                    self.emit(
                        addr,
                        mil::Insn::NotYetImplemented("segment-relative memory operand"),
                    );
                }

                assert_ne!(value, addr);
                self.emit(addr, mil::Insn::StoreMem { addr, value });
            }
        }
    }

    fn extend_zero(&mut self, value: mil::Reg, value_size: u16, dest_size: u16) {
        assert!(
            dest_size >= value_size,
            "dest ({dest_size} bytes) can't fit source ({value_size} bytes)"
        );
        if dest_size > value_size {
            self.emit(
                value,
                mil::Insn::Widen {
                    reg: value,
                    target_size: dest_size,
                    sign: false,
                },
            );
        }
    }

    fn emit_write_machine_reg(&mut self, dest_reg: Register, value_size: u16, value: mil::Reg) {
        let full_dest_reg = dest_reg.full_register();
        let full_size = full_dest_reg.size().try_into().unwrap();
        let full_dest = Importer::xlat_reg(full_dest_reg);

        // Intel 64 Software Developer's Manual,
        // 3.4.1.1 "General-Purpose Registers in 64-Bit Mode":
        //
        // >  When in 64-bit mode, operand size determines the
        // >  number of valid bits in the destination general-purpose
        // >  register:
        // >
        // >  * 64-bit operands generate a 64-bit result in the
        // >    destination general-purpose register.
        // >
        // >  * 8-bit and 16-bit operands generate an 8-bit or 16-bit
        // >    result. The upper 56 bits or 48 bits (respectively)
        // >    of the destination general-purpose register are not
        // >    modified by the operation. If the result of an 8-bit
        // >    or 16-bit operation is intended for 64-bit address
        // >    calculation, explicitly sign-extend the register to the
        // >    full 64-bits.
        // >
        // >  * 32-bit operands generate a 32-bit result, zero-extended
        // >    to a 64-bit result in the destination general-purpose
        // >    register.
        // >
        let (value, value_size) = if value_size == 4 && full_size == 8 {
            self.extend_zero(value, value_size, full_size);
            (value, 8)
        } else {
            (value, value_size)
        };

        if value_size == full_size {
            self.emit(full_dest, mil::Insn::Get(value));
            return;
        }

        assert!(value_size < full_size);
        let unchanged_part = self.pb.tmp_gen();
        self.emit(
            unchanged_part,
            mil::Insn::Part {
                src: full_dest,
                offset: value_size,
                size: full_size - value_size,
            },
        );
        self.emit(
            full_dest,
            mil::Insn::Concat {
                hi: unchanged_part,
                lo: value,
            },
        );
    }

    fn emit_compute_address(&mut self, insn: &iced_x86::Instruction) -> mil::Reg {
        let v0 = self.pb.tmp_gen();
        self.emit_compute_address_into(insn, v0);
        v0
    }
    fn emit_compute_address_into(&mut self, insn: &iced_x86::Instruction, dest: mil::Reg) {
        if insn.segment_prefix() != Register::None {
            event!(
                Level::WARN,
                segment_prefix = ?insn.segment_prefix(),
                "segment-relative memory address operands are not supported. assuming it is 0"
            );
        }

        self.pb.push(
            dest,
            mil::Insn::Int {
                value: insn.memory_displacement64() as i64,
                size: 8,
            },
        );

        match insn.memory_base() {
            Register::None => {}
            base => {
                // TODO make this recursive and use read_operand instead of xlat_reg?
                // addresses are 64-bit, so we use 8 bytes instructions
                self.pb.push(
                    dest,
                    mil::Insn::Arith(mil::ArithOp::Add, dest, Self::xlat_reg(base)),
                );
            }
        }

        match insn.memory_index() {
            Register::None => {}
            index_reg => {
                let v1 = self.pb.tmp_gen();
                let scale = insn.memory_index_scale() as i64;
                let reg = Self::xlat_reg(index_reg);
                // addresses are 64-bit in this architecture, so instructions are all with 8 bytes results
                let scaled = mil::Insn::ArithK(mil::ArithOp::Mul, reg, scale);
                self.pb.push(v1, scaled);
                self.pb
                    .push(dest, mil::Insn::Arith(mil::ArithOp::Add, dest, v1));
            }
        }
    }

    fn emit_cmp_a(&mut self) -> mil::Reg {
        // jmp if !SF && !ZF (also jnbe)
        let v0 = self.pb.tmp_gen();
        let v1 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::Not(Self::SF));
        self.emit(v1, mil::Insn::Not(Self::ZF));
        self.emit(v0, mil::Insn::Bool(mil::BoolOp::And, v0, v1));
        v0
    }

    fn emit_cmp_e(&mut self) -> mil::Reg {
        Self::ZF
    }

    fn emit_cmp_ne(&mut self) -> mil::Reg {
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::Not(Self::ZF));
        v0
    }

    fn emit_cmp_b(&mut self) -> mil::Reg {
        // also Jc, Jnae
        Self::CF
    }

    fn emit_cmp_l(&mut self) -> mil::Reg {
        // jmp if SF != OF
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::Cmp(mil::CmpOp::EQ, Self::SF, Self::OF));
        self.emit(v0, mil::Insn::Not(v0));
        v0
    }

    fn emit_cmp_le(&mut self) -> mil::Reg {
        // jmp if ZF=1 or SF != OF
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::Cmp(mil::CmpOp::EQ, Self::SF, Self::OF));
        self.emit(v0, mil::Insn::Not(v0));
        self.emit(v0, mil::Insn::Bool(mil::BoolOp::Or, v0, Self::ZF));
        v0
    }

    fn emit_cmp_ae(&mut self) -> mil::Reg {
        // also jnb, jnc
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::Not(Self::CF));
        v0
    }

    fn emit_cmp_be(&mut self) -> mil::Reg {
        // also jna
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::Bool(mil::BoolOp::Or, Self::CF, Self::ZF));
        v0
    }

    fn emit_cmp_cxz(&mut self) -> mil::Reg {
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::IsZero(Self::RCX));
        v0
    }

    fn emit_cmp_ecxz(&mut self) -> mil::Reg {
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::IsZero(Self::RCX));
        v0
    }

    fn emit_cmp_g(&mut self) -> mil::Reg {
        let v0 = self.pb.tmp_gen();
        let v1 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::Not(Self::ZF));
        self.emit(v1, mil::Insn::Cmp(mil::CmpOp::EQ, Self::SF, Self::OF));
        self.emit(v0, mil::Insn::Bool(mil::BoolOp::And, v0, v1));
        v0
    }

    fn emit_cmp_ge(&mut self) -> mil::Reg {
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::Cmp(mil::CmpOp::EQ, Self::SF, Self::OF));
        v0
    }

    fn emit_cmp_no(&mut self) -> mil::Reg {
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::Not(Self::OF));
        v0
    }

    fn emit_cmp_np(&mut self) -> mil::Reg {
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::Not(Self::PF));
        v0
    }

    fn emit_cmp_ns(&mut self) -> mil::Reg {
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::Not(Self::SF));
        v0
    }

    fn emit_cmp_o(&mut self) -> mil::Reg {
        Self::OF
    }

    fn emit_cmp_p(&mut self) -> mil::Reg {
        Self::PF
    }

    fn emit_cmp_rcxz(&mut self) -> mil::Reg {
        let v0 = self.pb.tmp_gen();
        self.emit(v0, mil::Insn::IsZero(Self::RCX));
        v0
    }

    fn emit_cmp_s(&mut self) -> mil::Reg {
        Self::SF
    }

    fn emit_flag_to_byte(&mut self, reg: mil::Reg) -> mil::Reg {
        self.emit(
            reg,
            mil::Insn::Widen {
                reg,
                target_size: 1,
                sign: false,
            },
        );
        reg
    }

    /// Emit a `cmovCC dst, src` instruction as
    ///   dst := Select { cond, then_val: src, else_val: old_dst }
    /// where `cond_fn` computes the boolean condition from the flags.
    ///
    /// `cond_fn` is a method on `Self` that takes `&mut self` and returns the
    /// boolean `Reg` holding the condition (e.g. `Self::emit_cmp_e`).  It is
    /// invoked once.
    fn emit_cmov(&mut self, insn: &iced_x86::Instruction, cond_fn: fn(&mut Self) -> mil::Reg) {
        let cond = cond_fn(self);
        let (src, src_sz) = self.emit_read(insn, 1);
        let (old_dst, dst_sz) = self.emit_read(insn, 0);
        assert_eq!(
            src_sz, dst_sz,
            "cmov: source and destination must be the same size"
        );
        let result = self.pb.tmp_gen();
        self.emit(
            result,
            mil::Insn::Select {
                cond,
                then_val: src,
                else_val: old_dst,
            },
        );
        self.emit_write(insn, 0, result, dst_sz);
    }

    // flags
    const CF: mil::Reg = mil::Reg(2); // Carry flag
    const PF: mil::Reg = mil::Reg(3); // Parity flag      true=even false=odd
    const AF: mil::Reg = mil::Reg(4); // Auxiliary Carry
    const ZF: mil::Reg = mil::Reg(5); // Zero flag        true=zero false=non-zero
    const SF: mil::Reg = mil::Reg(6); // Sign flag        true=neg false=pos
    const TF: mil::Reg = mil::Reg(7); // Trap flag
    const IF: mil::Reg = mil::Reg(8); // Interrupt enable true=enabled false=disabled
    const DF: mil::Reg = mil::Reg(9); // Direction flag   true=down false=up
    const OF: mil::Reg = mil::Reg(10); // Overflow flag    true=overflow false=no-overflow

    // general purpose regs
    const RBP: mil::Reg = mil::Reg(11);
    const RSP: mil::Reg = mil::Reg(12);
    const RIP: mil::Reg = mil::Reg(13);
    const RDI: mil::Reg = mil::Reg(14);
    const RSI: mil::Reg = mil::Reg(15);
    const RAX: mil::Reg = mil::Reg(16);
    const RBX: mil::Reg = mil::Reg(17);
    const RCX: mil::Reg = mil::Reg(18);
    const RDX: mil::Reg = mil::Reg(19);
    const R8: mil::Reg = mil::Reg(20);
    const R9: mil::Reg = mil::Reg(21);
    const R10: mil::Reg = mil::Reg(22);
    const R11: mil::Reg = mil::Reg(23);
    const R12: mil::Reg = mil::Reg(24);
    const R13: mil::Reg = mil::Reg(25);
    const R14: mil::Reg = mil::Reg(26);
    const R15: mil::Reg = mil::Reg(27);

    const ZMM0: mil::Reg = mil::Reg(28);
    const ZMM1: mil::Reg = mil::Reg(29);
    const ZMM2: mil::Reg = mil::Reg(30);
    const ZMM3: mil::Reg = mil::Reg(31);
    const ZMM4: mil::Reg = mil::Reg(32);
    const ZMM5: mil::Reg = mil::Reg(33);
    const ZMM6: mil::Reg = mil::Reg(34);
    const ZMM7: mil::Reg = mil::Reg(35);
    const ZMM8: mil::Reg = mil::Reg(36);
    const ZMM9: mil::Reg = mil::Reg(37);
    const ZMM10: mil::Reg = mil::Reg(38);
    const ZMM11: mil::Reg = mil::Reg(39);
    const ZMM12: mil::Reg = mil::Reg(40);
    const ZMM13: mil::Reg = mil::Reg(41);
    const ZMM14: mil::Reg = mil::Reg(42);
    const ZMM15: mil::Reg = mil::Reg(43);

    const R_TMP_FIRST: mil::Reg = mil::Reg(45);

    /// Translate a *full* register name
    fn xlat_reg(reg: iced_x86::Register) -> mil::Reg {
        match reg.full_register() {
            Register::None => panic!("invalid register: none"),
            Register::RBP => Self::RBP,
            Register::RSP => Self::RSP,
            Register::RIP => Self::RIP,
            Register::RDI => Self::RDI,
            Register::RSI => Self::RSI,
            Register::RAX => Self::RAX,
            Register::RBX => Self::RBX,
            Register::RCX => Self::RCX,
            Register::RDX => Self::RDX,
            Register::R8 => Self::R8,
            Register::R9 => Self::R9,
            Register::R10 => Self::R10,
            Register::R11 => Self::R11,
            Register::R12 => Self::R12,
            Register::R13 => Self::R13,
            Register::R14 => Self::R14,
            Register::R15 => Self::R15,
            Register::ZMM0 => Self::ZMM0,
            Register::ZMM1 => Self::ZMM1,
            Register::ZMM2 => Self::ZMM2,
            Register::ZMM3 => Self::ZMM3,
            Register::ZMM4 => Self::ZMM4,
            Register::ZMM5 => Self::ZMM5,
            Register::ZMM6 => Self::ZMM6,
            Register::ZMM7 => Self::ZMM7,
            Register::ZMM8 => Self::ZMM8,
            Register::ZMM9 => Self::ZMM9,
            Register::ZMM10 => Self::ZMM10,
            Register::ZMM11 => Self::ZMM11,
            Register::ZMM12 => Self::ZMM12,
            Register::ZMM13 => Self::ZMM13,
            Register::ZMM14 => Self::ZMM14,
            Register::ZMM15 => Self::ZMM15,
            _ => {
                panic!(
                    "unsupported register: {:?} (full: {:?})",
                    reg,
                    reg.full_register()
                )
            }
        }
    }

    fn emit(&mut self, dest: mil::Reg, insn: mil::Insn) -> mil::Index {
        self.pb.push(dest, insn)
    }

    fn last_index_of_value(&self, reg: mil::Reg) -> Option<mil::Index> {
        self.pb
            .iter()
            .enumerate()
            .filter_map(|(ndx, iv)| if *iv.dest == reg { Some(ndx) } else { None })
            .last()
            .map(|ndx| ndx.try_into().unwrap())
    }
}

pub fn check_subroutine_type<'t>(
    rtx: ty::ReadTxRef<'t>,
    tyid: ty::TypeID,
) -> Result<Cow<'t, ty::Subroutine>> {
    let ty = rtx.get_through_alias(tyid)?.expect("invalid type ID");
    match ty {
        Cow::Borrowed(ty::Ty::Subroutine(subr_ty)) => Ok(Cow::Borrowed(subr_ty)),
        Cow::Owned(ty::Ty::Subroutine(subr_ty)) => Ok(Cow::Owned(subr_ty)),
        _ => anyhow::bail!("not a subroutine type (ID: {:?})", tyid),
    }
}

define_ancestral_name!(ANC_CF, "CF");
define_ancestral_name!(ANC_PF, "PF");
define_ancestral_name!(ANC_AF, "AF");
define_ancestral_name!(ANC_ZF, "ZF");
define_ancestral_name!(ANC_SF, "SF");
define_ancestral_name!(ANC_TF, "TF");
define_ancestral_name!(ANC_IF, "IF");
define_ancestral_name!(ANC_DF, "DF");
define_ancestral_name!(ANC_OF, "OF");
define_ancestral_name!(ANC_RBP, "RBP");
define_ancestral_name!(ANC_RSP, "RSP");
define_ancestral_name!(ANC_RIP, "RIP");
define_ancestral_name!(ANC_RDI, "RDI");
define_ancestral_name!(ANC_RSI, "RSI");
define_ancestral_name!(ANC_RAX, "RAX");
define_ancestral_name!(ANC_RBX, "RBX");
define_ancestral_name!(ANC_RCX, "RCX");
define_ancestral_name!(ANC_RDX, "RDX");
define_ancestral_name!(ANC_R8, "R8");
define_ancestral_name!(ANC_R9, "R9");
define_ancestral_name!(ANC_R10, "R10");
define_ancestral_name!(ANC_R11, "R11");
define_ancestral_name!(ANC_R12, "R12");
define_ancestral_name!(ANC_R13, "R13");
define_ancestral_name!(ANC_R14, "R14");
define_ancestral_name!(ANC_R15, "R15");
define_ancestral_name!(ANC_ZMM0, "ZMM0");
define_ancestral_name!(ANC_ZMM1, "ZMM1");
define_ancestral_name!(ANC_ZMM2, "ZMM2");
define_ancestral_name!(ANC_ZMM3, "ZMM3");
define_ancestral_name!(ANC_ZMM4, "ZMM4");
define_ancestral_name!(ANC_ZMM5, "ZMM5");
define_ancestral_name!(ANC_ZMM6, "ZMM6");
define_ancestral_name!(ANC_ZMM7, "ZMM7");
define_ancestral_name!(ANC_ZMM8, "ZMM8");
define_ancestral_name!(ANC_ZMM9, "ZMM9");
define_ancestral_name!(ANC_ZMM10, "ZMM10");
define_ancestral_name!(ANC_ZMM11, "ZMM11");
define_ancestral_name!(ANC_ZMM12, "ZMM12");
define_ancestral_name!(ANC_ZMM13, "ZMM13");
define_ancestral_name!(ANC_ZMM14, "ZMM14");
define_ancestral_name!(ANC_ZMM15, "ZMM15");

#[cfg(test)]
mod tests {
    //! Decoder-level unit tests for individual x86_64 instructions.
    //!
    //! Each test assembles a single instruction with `iced_x86::code_asm`,
    //! feeds it through `import`, and asserts on the emitted MIL.

    use super::*;
    use iced_x86::code_asm::*;

    /// Assemble a single instruction and decode it to a MIL program.
    fn decode_one(build: impl FnOnce(&mut CodeAssembler) -> Result<(), IcedError>) -> mil::Program {
        let mut a = CodeAssembler::new(64).expect("create assembler");
        build(&mut a).expect("assemble instruction");
        let insns = a.take_instructions();
        let types = Arc::new(ty::TypeSet::new());
        import(insns.into_iter(), types, None).expect("import")
    }

    /// True if the program contains any `NotYetImplemented` instruction.
    fn has_nyi(prog: &mil::Program) -> bool {
        prog.iter()
            .any(|v| matches!(v.insn, mil::Insn::NotYetImplemented(_)))
    }

    /// True if the program contains a sign-widen to `target` bytes.
    fn has_sign_widen(prog: &mil::Program, target: u16) -> bool {
        prog.iter().any(|v| {
            matches!(
                v.insn,
                mil::Insn::Widen { sign: true, target_size, .. } if *target_size == target
            )
        })
    }

    #[test]
    fn movsxd_reg_reg() {
        // movsxd rax, ecx
        let prog = decode_one(|a| a.movsxd(rax, ecx));
        assert!(!has_nyi(&prog), "movsxd should be supported");
        assert!(
            has_sign_widen(&prog, 8),
            "movsxd rax, ecx must sign-extend to 8 bytes"
        );
    }

    #[test]
    fn movsxd_r12_r12d() {
        // movsxd r12, r12d
        let prog = decode_one(|a| a.movsxd(r12, r12d));
        assert!(!has_nyi(&prog));
        assert!(has_sign_widen(&prog, 8));
    }

    #[test]
    fn movsx_byte_to_dword_mem() {
        // movsx eax, byte ptr [rsi]
        let prog = decode_one(|a| a.movsx(eax, byte_ptr(rsi)));
        assert!(!has_nyi(&prog));
        assert!(has_sign_widen(&prog, 4));
    }

    #[test]
    fn movsx_word_to_dword_reg() {
        // movsx eax, word ptr [rsi]  (16-bit source -> 32-bit dest)
        let prog = decode_one(|a| a.movsx(eax, word_ptr(rsi)));
        assert!(!has_nyi(&prog));
        assert!(has_sign_widen(&prog, 4));
    }

    #[test]
    fn movsx_byte_to_qword_mem() {
        // movsx rax, byte ptr [rdx+0x32]
        let prog = decode_one(|a| a.movsx(rax, byte_ptr(rdx + 0x32)));
        assert!(!has_nyi(&prog));
        assert!(has_sign_widen(&prog, 8));
    }

    #[test]
    fn movsx_word_to_qword() {
        // movsx rax, word ptr [rsi]  (16-bit source -> 64-bit dest)
        let prog = decode_one(|a| a.movsx(rax, word_ptr(rsi)));
        assert!(!has_nyi(&prog));
        assert!(has_sign_widen(&prog, 8));
    }

    // ---- Phase 2: SSE moves ----

    #[test]
    fn movdqa_xmm_xmm() {
        // movdqa xmm0, xmm1
        let prog = decode_one(|a| a.movdqa(xmm0, xmm1));
        assert!(!has_nyi(&prog), "movdqa should be supported");
    }

    #[test]
    fn movdqu_xmm_mem() {
        // movdqu xmm0, [rsi]
        let prog = decode_one(|a| a.movdqu(xmm0, xmmword_ptr(rsi)));
        assert!(!has_nyi(&prog));
        assert!(prog.iter().any(|v| matches!(
            v.insn,
            mil::Insn::LoadMem { size, .. } if *size == 16
        )));
    }

    #[test]
    fn movq_xmm_reg_zeroes_upper() {
        // movq xmm0, rax  -> low 64 bits = rax, upper 64 bits zeroed
        let prog = decode_one(|a| a.movq(xmm0, rax));
        assert!(!has_nyi(&prog));
        // The zero-extension is emitted as an Int { value: 0, size: 8 }.
        assert!(
            prog.iter()
                .any(|v| matches!(v.insn, mil::Insn::Int { value: 0, size: 8 })),
            "movq xmm, reg must zero the upper 64 bits of the xmm"
        );
    }

    #[test]
    fn movq_reg_xmm_plain_move() {
        // movq rax, xmm0  -> plain 8-byte move, no zeroing
        let prog = decode_one(|a| a.movq(rax, xmm0));
        assert!(!has_nyi(&prog));
        assert!(
            !prog
                .iter()
                .any(|v| matches!(v.insn, mil::Insn::Int { value: 0, size: 8 })),
            "movq r64, xmm must not zero-extend anything"
        );
    }

    #[test]
    fn movq_xmm_xmm() {
        // movq xmm0, xmm1 -> low 64 bits copied, upper 64 bits zeroed
        let prog = decode_one(|a| a.movq(xmm0, xmm1));
        assert!(!has_nyi(&prog));
        assert!(
            prog.iter()
                .any(|v| matches!(v.insn, mil::Insn::Int { value: 0, size: 8 })),
            "movq xmm, xmm must zero the upper 64 bits of the destination xmm"
        );
    }

    #[test]
    fn movhps_xmm_mem() {
        // movhps xmm0, [rsi]
        let prog = decode_one(|a| a.movhps(xmm0, qword_ptr(rsi)));
        assert!(!has_nyi(&prog));
    }

    #[test]
    fn movhps_mem_xmm_store() {
        // movhps [rsi], xmm0  -> store the high qword of xmm0 to memory
        let prog = decode_one(|a| a.movhps(qword_ptr(rsi), xmm0));
        assert!(!has_nyi(&prog));
        // The high qword is extracted via Part { offset: 8, size: 8 }.
        assert!(prog.iter().any(|v| matches!(
            v.insn,
            mil::Insn::Part {
                offset: 8,
                size: 8,
                ..
            }
        )));
    }

    #[test]
    fn movhlps_xmm_xmm() {
        // movhlps xmm1, xmm0
        let prog = decode_one(|a| a.movhlps(xmm1, xmm0));
        assert!(!has_nyi(&prog));
    }

    // ---- Phase 3: Cmov* (Insn::Select) ----

    /// True if the program contains a `Select` instruction with the given
    /// condition, then, and else regs.
    fn has_select(prog: &mil::Program) -> bool {
        prog.iter()
            .any(|v| matches!(v.insn, mil::Insn::Select { .. }))
    }

    #[test]
    fn cmove_reg_reg() {
        // cmove rax, rdx  -> Select { cond: ZF, then: rdx, else: old_rax }
        let prog = decode_one(|a| a.cmove(rax, rdx));
        assert!(!has_nyi(&prog), "cmove should be supported");
        assert!(has_select(&prog), "cmove must emit Insn::Select");
    }

    #[test]
    fn cmovne_reg_reg() {
        // cmovne eax, ecx
        let prog = decode_one(|a| a.cmovne(eax, ecx));
        assert!(!has_nyi(&prog));
        assert!(has_select(&prog));
    }

    #[test]
    fn cmovl_reg_reg() {
        // cmovl ecx, eax
        let prog = decode_one(|a| a.cmovl(ecx, eax));
        assert!(!has_nyi(&prog));
        assert!(has_select(&prog));
    }

    #[test]
    fn cmova_mem_src() {
        // cmova eax, dword ptr [rdx+0x918]  (mem source, from unsupported_insns.txt)
        let prog = decode_one(|a| a.cmova(eax, dword_ptr(rdx + 0x918)));
        assert!(!has_nyi(&prog));
        assert!(has_select(&prog));
    }

    #[test]
    fn cmov_renders_as_ternary() {
        // End-to-end: a MIL program containing a Select must render with
        // C-like ternary syntax `(cond ? pos : neg)` through the AST
        // pretty-printer.
        use crate::ast::AstBuilder;
        use crate::pp::PrettyPrinter;
        use crate::ssa;

        let types = Arc::new(ty::TypeSet::new());
        let mut prog = mil::Program::new(mil::Reg(100), Some(Arc::clone(&types)));
        prog.push(
            mil::Reg(1),
            mil::Insn::Ancestral {
                anc_name: ANC_ZF,
                ll_type: mil::LLType::Bool,
            },
        );
        prog.push(mil::Reg(2), mil::Insn::Int { value: 42, size: 8 });
        prog.push(mil::Reg(3), mil::Insn::Int { value: 99, size: 8 });
        prog.push(
            mil::Reg(4),
            mil::Insn::Select {
                cond: mil::Reg(1),
                then_val: mil::Reg(2),
                else_val: mil::Reg(3),
            },
        );
        prog.push(mil::Reg(5), mil::Insn::SetReturnValue(mil::Reg(4)));
        prog.push(mil::Reg(5), mil::Insn::Control(Control::Ret));

        let ssa = ssa::Program::from_mil(prog);
        let ast = AstBuilder::new(&ssa).build();
        let mut buf = Vec::<u8>::new();
        let mut pp = PrettyPrinter::start(&mut buf);
        crate::ast::write_ast(&mut pp, &ast, &ssa, &types).expect("write_ast");
        let out = String::from_utf8(buf).expect("utf8");
        assert!(
            out.contains("?") && out.contains(" : "),
            "Select must render as `(cond ? pos : neg)`; got:\n{}",
            out
        );
    }

    // ---- Phase 4: Sar, Rol, Ror ----

    #[test]
    fn sar_reg_reg() {
        // sar rax, cl
        let prog = decode_one(|a| a.sar(rax, cl));
        assert!(!has_nyi(&prog));
    }

    #[test]
    fn sar_imm8() {
        // sar rcx, 2
        let prog = decode_one(|a| a.sar(rcx, 2));
        assert!(!has_nyi(&prog));
    }

    #[test]
    fn rol_reg_reg() {
        // rol eax, cl
        let prog = decode_one(|a| a.rol(eax, cl));
        assert!(!has_nyi(&prog));
    }

    #[test]
    fn ror_imm8() {
        // ror dx, 8
        let prog = decode_one(|a| a.ror(dx, 8));
        assert!(!has_nyi(&prog));
    }

    #[test]
    fn sar_ror_mem() {
        // sar qword ptr [rdi+0x28], 1 (from unsupported_insns.txt)
        let prog = decode_one(|a| a.sar(qword_ptr(rdi + 0x28), 1));
        assert!(!has_nyi(&prog));
        // ror qword ptr [rdi+0x28], 1
        let prog = decode_one(|a| a.ror(qword_ptr(rdi + 0x28), 1));
        assert!(!has_nyi(&prog));
    }
}
