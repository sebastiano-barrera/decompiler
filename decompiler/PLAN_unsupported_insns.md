# Plan: Add unsupported x86_64 instructions to the decoder — COMPLETED

`unsupported_insns.txt` lists **493 instruction instances across 63 mnemonics**
that currently fall through to the catch-all arm in `Importer::translate`
(`decompiler/src/x86_to_mil.rs`, the `_ =>` case), which emits
`Insn::NotYetImplemented("unsupported: …")`.

**Status: 488/493 implemented across 8 phases. 5 deferred (x87: fld, fldz, fstp).**

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
| `Movdqu` | 35 | Full 16-byte move; `emit_read`/`emit_write` already handle 16-byte xmm↔mem/xmm. |
| `Movdqa` | 23 | Identical to `Movdqu` at the MIL level. |
| `Movq` (xmm↔reg/mem) | 40 | 8-byte move of the low qword. |
| `Movhps` | 20 | Load 8 bytes into the **high** qword of xmm via `Part`+`Concat`. |
| `Movhlps` | 1 | Same high-qword move, register source. |
| `Mul` (1-op) | 13 | Reuse the existing `Imul` 1-operand pattern. |
| `Neg` | 8 | `zero = Int{0,size}`; `Arith(Sub, zero, a)`; flags from result. |
| `Not` | 8 | `Arith(BitXor, a, all_ones_const)`; no flags affected. |
| `Adc` | 2 | `t = Arith(Add, a, b)`; `cf = Widen{CF, size, false}`; `res = Arith(Add, t, cf)`. |
| `Sbb` | 11 | `t = Arith(Sub, a, b)`; `cf = Widen{CF, …}`; `res = Arith(Sub, t, cf)`. |
| `Bt` | 4 | `CF = (a >> bit) & 1`. No dest write. |
| `Btr` | 3 | `mask = 1 << n`; `res = a & ~mask`; write back. |
| `Bts` | 9 | `mask = 1 << n`; `res = a | mask`; write back. |
| `Xchg` | 3 | reg/mem↔reg: swap via temp save. |
| `Xadd` (+`lock`) | 3 | `old = Get(a)`; `sum = Add(a, b)`; write sum to op0, old to op1. |
| `Xorpd` | 1 | `emit_bit_op(…, BitXor, …)` on 16 bytes. |
| `Por` | 2 | `emit_bit_op(…, BitOr, …)` on 16 bytes. |
| `Psrldq` | 2 | 128-bit shift right by `imm8*8` bits. |
| `Shld` | 2 | `(dst<<cnt) | (src>>(bitsize-cnt))`. |
| `Ud2` | 1 | `SetReturnValue(UndefinedBytes{0})` + `Control::Ret`. |

### Tier 1 — small, localized **MIL-IR extension** (≈141 instances)

Requires adding either new `ArithOp` variants or new `Insn` variants, and
updating the small set of downstream exhaustive matches.

| Mnemonic | # | IR change | Downstream updates |
| --- | --- | --- | --- |
| `Cmov*` (a/ae/b/be/e/g/ge/l/le/ne/ns/s) | 66 | **New `Insn::Select { cond, then_val, else_val }`** (ternary mux). | `mil` Assoc; `ast::precedence`; `pp` (C-like ternary); `ssa`; `xform` (fold rules optional). |
| `Rol` | 27 | **New `ArithOp::Rol`** (+ `Ror`). | `symbol`; `eval_const`; `assoc_const` → `None`; `precedence`. |
| `Ror` | 14 | as above | as above |
| `Sar` | 18 | **New `ArithOp::Sar`** (signed shift right). | Same 4 sites. |
| `Bswap` | 9 | **New `Insn::ByteSwap { src, size }`**. | `mil` Assoc; `precedence`; `ssa`; `pp`. |
| `Bsr` | 4 | **New `Insn::BitScanReverse { src }`**. | Same as `ByteSwap` + flag writes. |
| `Div` (1-op, unsigned) | 2 | **New `ArithOp::DivU`, `RemU`** (+ signed `DivS`, `RemS`). | `symbol`; `eval_const`; `assoc_const` → `None`; `precedence`. |
| `Idiv` | 1 | as above (signed) | as above |

**Blast-radius of a new `ArithOp` variant** (exactly 4 sites, all small):

- `mil::ArithOp::symbol` (`src/mil.rs`)
- `xform::FoldConstants::eval_const` and `assoc_const` (`src/xform.rs`)
- `ast::precedence` (`src/ast.rs`)

**Blast-radius of a new `Insn` variant**:

- `mil::Insn` Assoc attributes (`input_regs`, `has_side_effects`, `is_replaceable`)
- `ast::precedence` (exhaustive match — add an arm)
- `ssa` construction (`src/ssa.rs`) — add a case in `deduce_one_reg`
- `pp` / `ast::builder` rendering (data-driven via `to_expanded` — no change needed)
- `xform` (no-op or fold rule)

### Tier 3 — **float / vector / x87** (49 instances — 44 implemented, 5 deferred)

These required either a float/vector type system in MIL or a structured
decomposition into per-lane operations.  Rather than the original plan of a
generic `Insn::Intrinsic`, the actual implementation added:

- **`LLType::Float(usize)`** — parametric float type (`Float(4)` = f32, `Float(8)` = f64)
- **`Insn::IntToDouble { src }`** and **`Insn::IntToFloat { src }`** — int→float conversion
- **`Insn::FloatToBytes { src }`** — float→bytes for `Concat`
- **`Arith` extended** to accept `Float(n)` operands (both same type → result is `Float(n)`)

| Group | Mnemonics (#) | Implementation |
| --- | --- | --- |
| Scalar float arith | `Addsd`2 `Addss`3 `Subss`1 `Mulsd`2 `Divsd`3 `Divss`1 | `Part` + `IntToDouble`/`IntToFloat` → `Arith(Float)` → `FloatToBytes` + `Concat` |
| Scalar float compare | `Comisd`3 `Comiss`1 `Ucomiss`1 | `Part` + `Cmp` on extracted bytes, flags approximated |
| Int↔float convert | `Cvtsi2sd`6 `Cvtsi2ss`6 | `IntToDouble`/`IntToFloat` + `Concat` with preserved upper bytes |
| Packed integer (lane ops) | `Paddq`1 `Pcmpeqd`2 `Psrld`2 `Punpckldq`3 `Punpcklqdq`5 `Shufps`1 `Unpcklpd`1 | Per-lane `Part` + `Arith`/`Cmp` + `Concat` sequences |
| **x87 (80-bit FPU)** | **`Fld`2 `Fldz`1 `Fstp`2** | **Deferred** — x87 register stack not modeled; 80-bit `tbyte` type. |

---

## 2. Implementation order (completed)

| Phase | Description | Instances |
| --- | --- | --- |
| 1 | Sign-extending moves (`Movsxd`, `Movsx`) | 109 |
| 2 | SSE moves (`Movdqa`, `Movdqu`, `Movq`, `Movhps`, `Movhlps`) | 119 |
| 3 | `Insn::Select` + `Cmov*` family | 66 |
| 4 | `ArithOp::Sar`, `Rol`, `Ror` | 59 |
| 5 | Tier-0 arithmetic/bit ops (`Neg`, `Not`, `Adc`, `Sbb`, `Bt`, `Btr`, `Bts`, `Mul`) | 58 |
| 6 | Atomics/exchange/shift-misc (`Xadd`, `Xchg`, `Shld`, `Psrldq`, `Por`, `Xorpd`, `Ud2`) | 14 |
| 7 | `Div`/`Idiv`, `Bswap`, `Bsr` | 16 |
| 8 | Float/SIMD/shuffle/convert (`LLType::Float`, `IntToDouble`/`Float`, per-lane decomposition) | 44 |
| **Deferred** | x87 (`Fld`, `Fldz`, `Fstp`) | **5** |
| **Total** | | **493** |

Each phase was independently shippable and testable.

---

## 3. Testing strategy

Three complementary layers:

### 3a. Decoder-level unit tests (fast, precise, no ELF)

A `#[cfg(test)] mod tests` in `x86_to_mil.rs` tests individual instructions
via `iced_x86::code_asm::CodeAssembler`.  Each test assembles one instruction,
runs `Importer::new(types).translate(iter, None)`, and asserts the emitted MIL
is correct (not `NotYetImplemented`).  Selected tests also assert specific
`Insn` variants (e.g. `Select`, `Widen`).

### 3b. End-to-end round-trip via the existing test binaries

The existing `test_tool` integration tests exercise the full decoder→SSA→xform→AST
path for the redis-server and callconv test binaries.  Updated snapshots confirm
that previously-`NotYetImplemented` instructions now decode to real MIL.

### 3c. Snapshot regression + coverage gate

- Re-run `cargo insta test --review` to accept updated `tests/snapshots/*.snap`;
  confirm the targeted `NotYetImplemented("unsupported: …")` lines are gone.
- The existing `redis_server::no_panic` test must still pass.
- Regenerating `unsupported_insns.txt` after the change should yield only
  the 5 deferred x87 mnemonics.

---

## 4. Outcomes / lessons learned

- **`Insn::Select` downstream**: worked as expected.  SSA treats it as a pure
  value op (no block split).  AST pretty-printer renders it as `(cond ? a : b)`.
- **Signedness in `Div`/`Rem`**: encoded as separate `ArithOp` variants
  (`DivU`/`DivS`/`RemU`/`RemS`).  Kept the `Assoc`-macro pattern simple.
- **Float type system**: `LLType::Float(usize)` was added instead of a generic
  `Intrinsic`.  Float-typed values flow through `Arith` (extended to accept
  `Float` operands).  `CastIntToFloat` / `CastFloatToInt` were replaced with
  dedicated `IntToDouble`/`IntToFloat`/`FloatToBytes` variants.
- **x87**: still deferred (5 instances).  `xlat_reg` has no `st(0..7)` and
  MIL has no 80-bit type.  True x87 support is a separate project.
- **SSE upper-bit semantics**: legacy SSE writes to XMM preserve the upper
  YMM/ZMM bits via `emit_write_machine_reg`'s existing `Concat` path.  This
  was already the behavior for existing SSE arms.
- **All `iced_x86` accessor names** confirmed present in version 1.21.0.
