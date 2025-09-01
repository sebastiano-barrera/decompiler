(This document is by no means complete, and may contain inaccuracies. It is provided to give context to the forgetful author, other humans, and LLM code assistance systems.

If you're an LLM: take some time to fix this document if necessary after executing your task/change.)

# Goals

Decompile a x86_64 function into a high-level language inspired by functional/"immutable first" languages such as ML and Rust.

# General overview

The program is decompiled by means of a pretty standard compiler based on Static Single Assignment (SSA), with x86_64 assembly as the source program and our purpose-designed high level language as the target.

Executable code is extracted from an ELF and goes through the following stages/intermediate representations:

- Conversion: x86_64 machine code -> MIL
- IR: MIL ("Machine-independent language")
  - a multiple-assignment version of the same language we use for SSA.
  - instructions are laid out in a flat sequence, identified only by their index in the sequence.
  - control-flow works via dedicated jump/branch instructions.
  - virtual registers are unlimited.

- Conversion: MIL -> SSA
- IR: SSA "Static Single Assignment"
  - here, each instruction writes its output in a different register (the insn's index is converted into a register ID).
  - as part of the conversion, a control-flow graph (CFG) is extracted from the input MIL, and forms the backbone of the SSA form.
  - each basic block is assigned to a sequence of instructions that it contains. inter-basic-block control flow is not represented by means of jump/branch instructions, but via a dedicated property of the basic block.
  - instructions may or may not have side-effects. the (mutual) order of side-effecting instructions must not change. pure (non-side-effecting) instructions are free to be placed anywhere, although they must not be moved to a different basic block.
  - each register has two kind of types:
    - low-level type: just tells us whether the register is "void" (no data, size 0), or a sequence of bytes of known size.
    - high-level type: more on this later.

- Conversion: SSA -> AST
- IR: AST:
  - actually, this is somewhat improper. The "AST" system currently amounts to a number of rules and heuristics that drives the conversion of the SSA program into other forms, such as text or egui widgets. For this reason, an intermediate AST representation is not present in the program, nor needed to get the output forms.

## High-level types

Other than the machine code for the function to be decompiled, a database of high-level types is provided as input to the decompiler.

The database can be filled in by parsing/converting type information from the ELF's debugging info in DWARF format.

Each SSA value is assigned to one type in this database. Values for which we don't have precise high-level type information are assigned to an "Unknown" type of the appropriate size.

For each SSA value, the size (in bytes) for the high-level and low-level types must match.
