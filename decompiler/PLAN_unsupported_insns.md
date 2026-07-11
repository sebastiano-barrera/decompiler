# Plan: Add unsupported x86_64 instructions to the decoder

`unsupported_insns.txt` lists **493 instruction instances across 63 mnemonics**
that currently fall through to the catch-all arm in `Importer::translate`
(`decompiler/src/x86_to_mil.rs`, the `_ =>` case), which emits
`Insn::NotYetImplemented("unsupported: …")`.

This plan categorizes every one of those 63 mnemonics by what it takes to
decode it, specifies the required MIL-IR changes and their downstream blast
radius, lays out an implementation order by ROI, and defines the testing
strategy.

All counts below are instance counts from `unsupported_insns.txt` (the mnemonic
in parentheses is the `iced_x86::Mnemonic` enum variant to match on).

---

## 1. Categorization

### Tier 0 — decoder match-arm only, **no MIL-IR change** (≈300 instances)

These can be expressed with `Insn`/`ArithOp` variants that already exist
(`Widen`, `Part`, `Concat`, `Arith{Add,Sub,Mul,Shl,Shr,BitAnd,BitOr,BitXor}`,
`ArithK`, `Not`, `LoadMem`, `StoreMem`, `Get`, `Int`, flag regs, `Control`).
Adding them is a localized new `M::Foo => { … }` arm in `translate`.

| Mnemonic | # | Approach |
| --- | --- | --- |
| `Movsxd` | 83 | Read 32-bit src, `Widen{target_size:8, sign:true}`, write 64-bit dest. **Single biggest win.** |
| `Movsx` (byte) | 26 | Read 1-byte src, `Widen{sign:true}` to dest size (4 or 8). Same pattern as `Movsxd`. |
| `Movdqu` | 35 | Full 16-byte move; `emit_read`/`emit_write` already handle 16-byte xmm↔mem/xmm (full_register→zmm, `Part`/`Concat` for the xmm-in-zmm). Just add the mnemonic. |
| `Movdqa` | 23 | Identical to `Movdqu` at the MIL level. |
| `Movq` (xmm↔reg/mem) | 40 | 8-byte move of the low qword. `emit_read`/`emit_write` handle the low-part-of-zmm via existing `Part`/`Concat` path. |
| `Movhps` | 20 | Load 8 bytes into the **high** qword of xmm: `lo = Part{zmm,0,8}`; `zmm = Concat{hi: new8, lo}`. Needs a dedicated arm (the generic `emit_write` only writes the low part) but uses only existing ops. |
| `Movhlps` | 1 | Same high-qword move, register source. |
| `Mul` (1-op) | 13 | Reuse the existing `Imul` 1-operand pattern verbatim: `v = Arith(Mul, RAX, src)`, split via `Part` into hi/lo, write `RDX`/`RAX`. (`Arith(Mul,…)` already yields a 2N-bit result — see the existing `Imul` arm.) |
| `Neg` | 8 | `zero = Int{0,size}`; `Arith(Sub, zero, a)`; flags: `CF = !IsZero(a)`, SF/ZF/OF from result. |
| `Not` | 8 | `Arith(BitXor, a, all_ones_const)`; no flags affected. |
| `Adc` | 2 | `t = Arith(Add, a, b)`; `cf = Widen{CF, size, false}`; `res = Arith(Add, t, cf)`; flags via `emit_set_flags_arith(t)` (approx — true carry is of `a+b+CF`). |
| `Sbb` | 11 | `t = Arith(Sub, a, b)`; `cf = Widen{CF, …}`; `res = Arith(Sub, t, cf)`; flags approx. |
| `Bt` | 4 | `CF = (a >> bit) & 1`. Read bit index (reg or imm), `shifted = Arith(Shr, a, n)`, `low = Part{shifted,0,1}`, `CF = !IsZero(low)`. No dest write. |
| `Btr` | 3 | `mask = 1 << n`; `res = Arith(BitAnd, a, Arith(BitXor, all_ones, mask))`; write back; `CF` as above. |
| `Bts` | 9 | `mask = 1 << n`; `res = Arith(BitOr, a, mask)`; write back; `CF` as above. |
| `Xchg` | 3 | reg/mem↔reg: `old = LoadMem(addr)` (or `Get`); `StoreMem(addr, r)`; `r = old`. Reg-reg: two `Get`/swap via tmp. |
| `Xadd` (+`lock`) | 3 | `old = LoadMem(addr)`; `new = Arith(Add, old, r)`; `StoreMem(addr, new)`; `r = old`. Ignore `lock` atomicity (sequential decompile is fine); optionally `event!(WARN)` if `insn.has_lock_prefix()`. |
| `Xorpd` | 1 | = `Pxor` on 16 bytes → `emit_bit_op(…, BitXor, …)`. |
| `Por` | 2 | `emit_bit_op(…, BitOr, …)` on 16 bytes. |
| `Psrldq` | 2 | 128-bit logical shift right by `imm8 * 8` bits → `Arith(Shr, xmm, imm8*8)` treating xmm as a 16-byte integer. |
| `Shld` | 2 | `Arith(BitOr, Arith(Shl, a, n), Arith(Shr, b, bits-n))`. (Flags TODO, like existing shifts.) |
| `Ud2` | 1 | Emit `SetReturnValue(UndefinedBytes{0})` then `Control::Ret` (model as an unreachable trap / function exit). |

**Caveat to document in code:** `emit_write_machine_reg` preserves the upper
bits of the underlying ZMM when writing an XMM (legacy SSE semantics zero them).
This is an acceptable approximation for decompilation and is already the
behavior used by the existing `Movaps`/`Movups`/`Movsd` arms.

### Tier 1 — small, localized **MIL-IR extension** (≈141 instances)

Requires adding either new `ArithOp` variants or new `Insn` variants, and
updating the small set of downstream exhaustive matches.

| Mnemonic | # | IR change | Downstream updates |
| --- | --- | --- | --- |
| `Cmov*` (a/ae/b/be/e/g/ge/l/le/ne/ns/s) | 66 | **New `Insn::Select { cond, then_val, else_val }`** (ternary mux). Decoder: `cond = emit_cmp_<cc>()` (reuse existing flag combinator helpers), `then = emit_read(op1)`, `els = emit_read(op0)` (old dest), `res = Select{cond, then, els}`, write op0. | `mil` Assoc `input_regs=[cond,then,else]`; `ast::precedence` (level ~205, just below `Cmp`); `ast::builder` render `cond ? a : b`; `pp`; `ssa` (side-effect-free value op); `xform` fold `cond=True→then`, `cond=False→else`, `then==else→then`. |
| `Rol` | 27 | **New `ArithOp::Rol`** (+ `Ror`). | `mil::symbol` (`rol`/`ror`); `xform::eval_const` (rotate of constant); `xform::assoc_const` → `None`; `ast::precedence` (group with `Shl`/`Shr` = 212). |
| `Ror` | 14 | as above | as above |
| `Sar` | 18 | **New `ArithOp::Sar`** (signed shift right). | `symbol` (`sar`/`>>` signed); `eval_const` (`checked_shr` on the sign-extended value); `assoc_const` → `None`; `precedence` with `Shl`/`Shr`. |
| `Bswap` | 9 | **New `Insn::ByteSwap { src, size }`** (cleaner than 8×`Part`+7×`Concat`; maps to `__builtin_bswap{32,64}`). | `mil` Assoc `input_regs=[src]`; `precedence` (~248, like `Widen`); `ssa` read src; `pp`; `xform` fold-on-constant (optional). |
| `Bsr` | 4 | **New `Insn::BitScanReverse { src }`** (+ optional `BitScanReverse`/`BitScanForward`). Sets `ZF = IsZero(src)`, dest = index. | as `ByteSwap` (new `Insn` variant) + flag writes. |
| `Div` (1-op, unsigned) | 2 | **New `ArithOp::DivU`, `RemU`** (+ signed `DivS`, `RemS` for `Idiv`). Decoder: `dividend = Concat{hi:RDX, lo:RAX}` (128-bit); `q = Arith(DivU, dividend, src)`; `r = Arith(RemU, dividend, src)`; write `RAX=q`, `RDX=r`. | `symbol`; `eval_const` (`checked_div`/`checked_rem`); `assoc_const` → `None`; `precedence` (with `Mul` = 211). |
| `Idiv` | 1 | as above (signed) | as above |

**Blast-radius of a new `ArithOp` variant** (exactly 4 sites, all small):

- `mil::ArithOp::symbol` (`src/mil.rs`)
- `xform::FoldConstants::eval_const` and `assoc_const` (`src/xform.rs`)
- `ast::precedence` (`src/ast.rs`)

**Blast-radius of a new `Insn` variant**:

- `mil::Insn` Assoc attributes (`input_regs`, `has_side_effects`, `is_replaceable`)
- `ast::precedence` (exhaustive match — add an arm)
- `ssa` construction (`src/ssa.rs`, ~90 `Insn::` refs — add a case)
- `pp` / `ast::builder` rendering
- `xform` (no-op or fold rule)

### Tier 3 — **float / vector / x87** — defer or intrinsic escape-hatch (≈49 instances)

These genuinely need either a float/vector type system in MIL or an intrinsic
escape-hatch. The MIL `LLType` today is only `Bytes | Bool | Effect | Error`
(no `F32`/`F64`, no vector, no x87 register file). Recommended interim: add a
generic **`Insn::Intrinsic { name: &'static str, args: Vec<Reg>, ret_size: u32 }`**
so these decode without panicking and render as a named builtin call; full
float/vector modeling is a separate, larger project.

| Group | Mnemonics (#) | Why deferred |
| --- | --- | --- |
| Scalar float arith | `Addsd`2 `Addss`3 `Subss`1 `Mulsd`2 `Divsd`3 `Divss`1 | Need `LLType::F64`/`F32` + float `ArithOp`s. |
| Scalar float compare | `Comisd`3 `Comiss`1 `Ucomiss`1 | Float compare → EFLAGS (NaN/unordered ordering). |
| Int↔float convert | `Cvtsi2sd`6 `Cvtsi2ss`6 | Int→float conversion op. |
| Packed integer (lane ops) | `Paddq`1 `Pcmpeqd`2 `Psrld`2 `Punpckldq`3 `Punpcklqdq`5 `Shufps`1 `Unpcklpd`1 | Per-lane / element-shuffle — not a clean 128-bit op. |
| x87 (80-bit FPU) | `Fld`2 `Fldz`1 `Fstp`2 | x87 register stack is not modeled at all (no `st(0..7)` in `xlat_reg`); 80-bit `tbyte` type. Low frequency; large effort. |

---

## 2. Implementation order (by ROI / unblocking the most lines)

1. **Tier 0 sign-extending moves** — `Movsxd` (83) + `Movsx` (26) = **109 instances** in two tiny arms. Do first.
2. **Tier 0 SSE moves** — `Movdqu`35 + `Movdqa`23 + `Movq`40 + `Movhps`20 + `Movhlps`1 = **119 instances**. Mostly one-liners reusing `emit_read`/`emit_write` plus a dedicated high-qword helper for `Movhps`/`Movhlps`.
3. **Tier 1 `Select` + `Cmov*`** — the single biggest non-move win (**66 instances**) and the most cross-cutting. Land the `Insn::Select` IR change + downstream first, then the 12 `Cmov*` arms.
4. **Tier 1 `Sar` + `Rol` + `Ror`** — **59 instances**; three new `ArithOp` variants, each a ~4-site update. Can be done together.
5. **Tier 0 arithmetic/bit** — `Neg`8 `Not`8 `Adc`2 `Sbb`11 `Bt`4 `Btr`3 `Bts`9 `Mul`13 = **58 instances**, all small arms.
6. **Tier 0 atomics/exchange/shift-misc** — `Xadd`3 `Xchg`3 `Shld`2 `Psrldq`2 `Por`2 `Xorpd`1 `Ud2`1 = **14 instances**.
7. **Tier 1 `Div`/`Idiv` + `Bswap` + `Bsr`** — **16 instances**; new `ArithOp`s (`Div`/`Rem` ± signed) and two new `Insn`s.
8. **Tier 3 intrinsic escape-hatch** — add `Insn::Intrinsic`, then 49 arms that emit it (so the decoder no longer panics and the output names the builtin). True float/vector/x87 support is future work.

Each phase is independently shippable and testable.

---

## 3. Testing strategy

Three complementary layers:

### 3a. Decoder-level unit tests (fast, precise, no ELF)

Add a `#[cfg(test)] mod tests` (in `x86_to_mil.rs` or a new `tests/x86_decoder_unit.rs`)
that, for each newly-supported mnemonic, builds an `iced_x86::Instruction`
(via `iced_x86::Decoder::decode` on hand-encoded bytes, or the `code_asm!`
macro already enabled by the `code_asm` feature), runs
`Importer::new(types).translate(iter, None)`, and asserts that the emitted MIL
contains the expected `Insn` variant and **not** `NotYetImplemented`.

These are the primary correctness tests for the decoder mapping (e.g. assert
`movsxd rax, ecx` produces a `Widen{sign:true, target_size:8}`; `cmovne`
produces `Select{cond=Not(ZF), …}`; `sar` produces `Arith(Sar, …)`).

A table-driven helper keeps this small:

```rust
// sketch
fn decode_one(bytes: &[u8]) -> Vec<(mil::Reg, mil::Insn)> { … }
assert!(decode_one(&[0x48, 0x63, 0xc1]) /* movsxd rax,rcx */
        .iter().any(|(_, i)| matches!(i, mil::Insn::Widen { sign: true, target_size: 8, .. })));
```

### 3b. End-to-end round-trip via a hand-authored test binary

The existing callconv test binary is *generated* (`do not edit`). Add a
**separate**, hand-authored `test-data/x86_64_insn_tests.c` of small
`__attribute__((noinline))` functions each exercising one instruction pattern
(sign-extend, conditional move, rotate, divide, bswap, …), compiled in
`build.rs` (gcc `-O1 -gdwarf`, like the existing rule). A new
`tests/insn_tests.rs` uses the `tests::utils::dataflow::compute_data_flow`
helper (or `insta::assert_snapshot`) to decompile each function and assert the
data-flow / rendered AST. This validates the **full** decoder→SSA→xform→AST
path for the new ops, including the downstream `Select`/`Sar`/`Rol`/`Div`
handling.

### 3c. Snapshot regression + coverage gate

- Re-run `cargo insta test --review` to accept updated `tests/snapshots/*.snap`;
  confirm the targeted `NotYetImplemented("unsupported: …")` lines are gone.
- The existing `redis_server::no_panic` test must still pass.
- Add a **coverage-gate test** that scans the decompilation output of the test
  binaries for `NotYetImplemented("unsupported: …")`, collects the remaining
  mnemonic set, and asserts it is a subset of the documented Tier-3 deferred
  set. This makes the contract explicit and prevents regressions: regenerating
  `unsupported_insns.txt` after the change must yield only Tier-3 mnemonics.

---

## 4. Risks / open questions

- **`Insn::Select` downstream**: the biggest cross-cutting change. Must confirm
  `ssa.rs` treats it as a pure value op (no block split) and that `ast::builder`
  renders `cond ? a : b` with correct precedence/parentheses. (Alternative
  rejected: lowering `cmov` to a branch + phi — would synthesize fall-through
  blocks in the linear decoder and disturb structure recovery.)
- **Signedness in `Div`/`Rem`**: decide whether to encode signedness in the
  `ArithOp` variant (`DivU`/`DivS`/`RemU`/`RemS`) or as a field on a single
  `Div`/`Rem` variant. Variants keep the `Assoc`-macro pattern simplest.
- **x87**: `xlat_reg` has no `st(0..7)` and MIL has no 80-bit type. The
  intrinsic escape-hatch avoids modeling the x87 stack; real support is a
  separate project.
- **SSE upper-bit semantics**: legacy SSE writes to XMM zero the upper YMM/ZMM
  bits; the decoder preserves them. Already the case for existing SSE arms;
  document, don't fix now.
- Confirm exact `iced_x86` accessor names during implementation
  (`has_rep_prefix`, `has_repe_prefix`, `has_repne_prefix`, `has_lock_prefix`,
  `is_string_instruction` all confirmed present in 1.21.0).
