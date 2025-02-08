use std::{
    collections::{HashMap, HashSet},
    ops::Range,
};

/// Static Single-Assignment representation of a program (and conversion from direct multiple
/// assignment).
///
/// The algorithms in this module are mostly derived from the descriptions in:
/// > Cooper, Keith & Harvey, Timothy & Kennedy, Ken. (2006).
/// > A Simple, Fast Dominance Algorithm.
/// > Rice University, CS Technical Report 06-33870.
use crate::{cfg, mil};

#[derive(Clone)]
pub struct Program {
    // an ssa::Program contains a mil::Program at its core, but never exposes it directly:
    //
    // - extra instructions are appended for various uses, although they do not belong
    //   (directly) to the program sequence (they're only referred-to by other in-sequence
    //   instruction)
    //
    // - registers are numerically equal to the index of the defining instruction.
    //   mil::Index values are just as good as mil::Reg for identifying both insns and
    //   values.
    inner: mil::Program,
    phis: cfg::BlockMap<PhiInfo>,
    cfg: cfg::Graph,

    rdr_count: ReaderCount,

    #[cfg(feature = "proto_typing")]
    ptr_regs: HashMap<mil::Reg, Ptr>,
}

#[cfg(feature = "proto_typing")]
slotmap::new_key_type! { pub struct TypeID; }

#[cfg(feature = "proto_typing")]
#[derive(Clone)]
pub struct Ptr {
    pub pointee_tyid: TypeID,
}

impl Program {
    pub fn cfg(&self) -> &cfg::Graph {
        &self.cfg
    }

    /// Get the defining instruction for the given register.
    ///
    /// (Note that it's not allowed to fetch instructions by position.)
    pub fn get(&self, reg: mil::Reg) -> Option<mil::InsnView> {
        // In SSA, Reg(ndx) happens to be located at index ndx.
        // But it's a detail we try to hide, as it's likely we're going to have
        // to transition to a more complex structure in the future.
        let iv = self.inner.get(reg.0)?;
        debug_assert_eq!(iv.dest.get(), reg);

        // "Get" instructions are never really needed in SSA.
        // If we have `y <- Get(x)`, then every usage of y could be replaced
        // with x. But performing the replacement is relatively costly (linearly
        // scanning the whole program). So, instead, we only "dereference"
        // lookups done at the SSA level if they hit a Get, and do another
        // trick in printing to make it look like the substitution is performed
        // "textually".
        if let mil::Insn::Get(x) = iv.insn.get() {
            return self.get(x);
        }

        Some(iv)
    }

    pub fn reg_count(&self) -> mil::Index {
        self.inner.len()
    }

    pub fn readers_count(&self, reg: mil::Reg) -> usize {
        self.rdr_count.get(reg)
    }

    pub fn block_phi(&self, bid: cfg::BlockID) -> &PhiInfo {
        &self.phis[bid]
    }

    pub fn is_alive(&self, reg: mil::Reg) -> bool {
        self.rdr_count.get(reg) > 0
    }

    pub fn live_regs(&self) -> impl '_ + Iterator<Item = mil::Reg> {
        (0..self.reg_count())
            .map(|ndx| mil::Reg(ndx))
            .filter(|reg| self.is_alive(*reg))
    }

    /// Iterate through the instructions in the program, in no particular order
    pub fn insns_unordered(&self) -> impl Iterator<Item = mil::InsnView> {
        self.live_regs().map(|reg| self.get(reg).unwrap())
    }

    pub fn block_normal_insns(&self, bid: cfg::BlockID) -> Option<mil::InsnSlice> {
        let ndx_range = self.cfg.insns_ndx_range(bid);
        self.inner.slice(ndx_range)
    }

    pub fn get_call_args(&self, reg: mil::Reg) -> impl '_ + Iterator<Item = mil::Reg> {
        self.inner.get_call_args(reg.0)
    }

    pub fn get_phi_args(&self, reg: mil::Reg) -> impl '_ + Iterator<Item = mil::Reg> {
        self.inner.get_phi_args(reg.0)
    }

    pub fn map_phi_args(&self, reg: mil::Reg, f: impl Fn(mil::Reg) -> mil::Reg) {
        self.inner.map_phi_args(reg.0, f)
    }

    pub fn value_type(&self, reg: mil::Reg) -> mil::RegType {
        use mil::{Insn, RegType};
        match self.inner.get(reg.0).unwrap().insn.get() {
            Insn::True => RegType::Bool,
            Insn::False => RegType::Bool,
            Insn::Const1(_) => RegType::Bytes(1),
            Insn::Const2(_) => RegType::Bytes(2),
            Insn::Const4(_) => RegType::Bytes(4),
            Insn::Const8(_) => RegType::Bytes(8),
            Insn::Part { size, .. } => RegType::Bytes(size as usize),
            Insn::Get(_) => panic!("mil::Program::get returned a Get"),
            Insn::Concat { lo, hi } => {
                let lo_size = self.value_type(lo).bytes_size().unwrap();
                let hi_size = self.value_type(hi).bytes_size().unwrap();
                RegType::Bytes(lo_size + hi_size)
            }
            Insn::Widen1_2(_) => RegType::Bytes(2),
            Insn::Widen1_4(_) => RegType::Bytes(4),
            Insn::Widen1_8(_) => RegType::Bytes(8),
            Insn::Widen2_4(_) => RegType::Bytes(4),
            Insn::Widen2_8(_) => RegType::Bytes(8),
            Insn::Widen4_8(_) => RegType::Bytes(8),
            Insn::Arith1(_, _, _) => RegType::Bytes(1),
            Insn::Arith2(_, _, _) => RegType::Bytes(2),
            Insn::Arith4(_, _, _) => RegType::Bytes(4),
            Insn::Arith8(_, _, _) => RegType::Bytes(8),
            Insn::ArithK1(_, _, _) => RegType::Bytes(1),
            Insn::ArithK2(_, _, _) => RegType::Bytes(2),
            Insn::ArithK4(_, _, _) => RegType::Bytes(4),
            Insn::ArithK8(_, _, _) => RegType::Bytes(8),
            Insn::Cmp(_, _, _) => RegType::Bool,
            Insn::Bool(_, _, _) => RegType::Bool,
            Insn::Not(_) => RegType::Bool,
            // TODO This might have to change based on the use of calling
            // convention and function type info
            Insn::Call(_) => RegType::Bytes(8),
            Insn::CArg(_) => RegType::Effect,
            Insn::Ret(_) => RegType::Effect,
            Insn::JmpInd(_) => RegType::Effect,
            Insn::Jmp(_) => RegType::Effect,
            Insn::JmpExt(_) => RegType::Effect,
            Insn::JmpIf { .. } => RegType::Effect,
            Insn::JmpExtIf { .. } => RegType::Effect,
            Insn::TODO(_) => RegType::Effect,
            Insn::LoadMem1(_) => RegType::Bytes(1),
            Insn::LoadMem2(_) => RegType::Bytes(2),
            Insn::LoadMem4(_) => RegType::Bytes(4),
            Insn::LoadMem8(_) => RegType::Bytes(8),
            Insn::StoreMem(_, _) => RegType::Effect,
            Insn::OverflowOf(_) => RegType::Effect,
            Insn::CarryOf(_) => RegType::Effect,
            Insn::SignOf(_) => RegType::Effect,
            Insn::IsZero(_) => RegType::Effect,
            Insn::Parity(_) => RegType::Effect,
            Insn::Undefined => RegType::Effect,
            Insn::Phi { size } => RegType::Bytes(size as usize),
            Insn::PhiBool => RegType::Bool,
            Insn::PhiArg(_) => RegType::Effect,
            Insn::Ancestral(anc_name) => self
                .inner
                .ancestor_type(anc_name)
                .expect("ancestor has no defined type"),
            Insn::StructGet8 { .. } => RegType::Bytes(8),
            Insn::StructGetMember { .. } => RegType::Effect,
        }
    }

    pub fn assert_invariants(&self) {
        self.assert_no_circular_refs();
        self.assert_inputs_alive();
        self.assert_phis_separated();
    }

    fn assert_phis_separated(&self) {
        for bid in self.cfg.block_ids() {
            for phi_reg in self.block_phi(bid).phi_regs() {
                let insn = self.get(phi_reg).unwrap().insn.get();
                assert!(insn.is_phi());
            }

            for (_, insn_cell) in self.block_normal_insns(bid).unwrap().iter() {
                assert!(!insn_cell.get().is_phi());
            }
        }
    }

    fn assert_inputs_alive(&self) {
        for bid in self.cfg.block_ids_rpo() {
            for phi_reg in self.block_phi(bid).phi_regs() {
                if !self.is_alive(phi_reg) {
                    continue;
                }

                for arg in self.get_phi_args(phi_reg) {
                    assert!(
                        self.is_alive(arg),
                        "{phi_reg:?}: phi input not alive: {arg:?}"
                    );
                }
            }

            for (dest, insn) in self.block_normal_insns(bid).unwrap().iter_copied() {
                if !self.is_alive(dest) {
                    continue;
                }

                insn.input_regs_iter().for_each(|reg| {
                    assert!(self.is_alive(reg), "{dest:?}: input not alive: {reg:?}");
                });
            }
        }
    }

    fn assert_no_circular_refs(&self) {
        let mut defined = HashSet::new();
        for bid in self.cfg.block_ids_rpo() {
            for phi_reg in self.block_phi(bid).phi_regs() {
                // phi arguments are not checked
                let is_first_def = defined.insert(phi_reg);
                assert!(is_first_def);
            }

            for (dest, insn) in self.block_normal_insns(bid).unwrap().iter_copied() {
                insn.input_regs_iter().for_each(|reg| {
                    assert!(defined.contains(&reg), "{dest:?}: undefined reg: {reg:?}")
                });

                let is_first_def = defined.insert(dest);
                assert!(is_first_def);
            }
        }
    }
}

#[cfg(test)]
#[test]
#[should_panic]
fn test_assert_no_circular_refs() {
    use mil::{ArithOp, Insn, Reg};

    let prog = {
        let mut pb = mil::ProgramBuilder::new();
        pb.set_input_addr(0xf0);
        pb.push(Reg(0), Insn::Const8(123));
        pb.push(Reg(1), Insn::Arith8(ArithOp::Add, Reg(0), Reg(2)));
        pb.push(Reg(2), Insn::Arith8(ArithOp::Add, Reg(0), Reg(1)));
        pb.build()
    };
    let prog = mil_to_ssa(ConversionParams::new(prog));
    prog.assert_no_circular_refs();
}

#[cfg(feature = "proto_typing")]
impl Program {
    pub fn set_ptr_type(&mut self, reg: mil::Reg, ptr_ty: Ptr) {
        self.ptr_regs.insert(reg, ptr_ty);
    }

    pub fn ptr_type(&self, reg: mil::Reg) -> Option<&Ptr> {
        self.ptr_regs.get(&reg)
    }
}

#[derive(Clone)]
pub struct PhiInfo {
    ndxs: Range<mil::Index>,
    // TODO smaller sizes?; these rarely go above, like 4-8
    phi_count: mil::Index,
    pred_count: mil::Index,
}

impl PhiInfo {
    fn empty() -> Self {
        PhiInfo {
            ndxs: 0..0,
            phi_count: 0,
            pred_count: 0,
        }
    }

    pub fn phi_count(&self) -> mil::Index {
        self.phi_count
    }

    pub fn phi_reg(&self, phi_ndx: mil::Index) -> mil::Reg {
        assert!(phi_ndx < self.phi_count);
        assert_eq!(
            self.ndxs.len(),
            (self.phi_count * (1 + self.pred_count)).into()
        );
        let value_ndx = self.ndxs.start + phi_ndx * (1 + self.pred_count);
        mil::Reg(value_ndx)
    }

    pub fn phi_regs(&self) -> impl '_ + DoubleEndedIterator<Item = mil::Reg> {
        (0..self.phi_count).map(|phi_ndx| self.phi_reg(phi_ndx))
    }

    pub fn arg<'p>(&self, ssa: &'p Program, phi_ndx: mil::Index, pred_ndx: mil::Index) -> mil::Reg {
        let item = ssa.get(self.arg_ndx(phi_ndx, pred_ndx)).unwrap();
        match item.insn.get() {
            mil::Insn::PhiArg(reg) => reg,
            other => panic!("expected phiarg, got {:?}", other),
        }
    }

    fn arg_ndx(&self, phi_ndx: mil::Index, pred_ndx: mil::Index) -> mil::Reg {
        assert!(pred_ndx < self.pred_count);
        mil::Reg(self.phi_reg(phi_ndx).0 + 1 + pred_ndx)
    }
}

impl std::fmt::Debug for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count: usize = self.live_regs().count();
        writeln!(f, "ssa program  {} instrs", count)?;

        for bid in self.cfg.block_ids_rpo() {
            let phis = self.block_phi(bid);
            let block_addr = {
                let nor_ndxs = self.cfg.insns_ndx_range(bid);
                self.inner.get(nor_ndxs.start).unwrap().addr
            };
            let insns = self.block_normal_insns(bid).unwrap();

            write!(f, ".B{}:    ;;  ", bid.as_usize())?;
            let preds = self.cfg.block_preds(bid);
            if preds.len() > 0 {
                write!(f, "preds:")?;
                for (ndx, pred) in preds.iter().enumerate() {
                    if ndx > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "B{}", pred.as_number())?;
                }
                write!(f, "  ")?;
            }
            writeln!(
                f,
                "addr:0x{:x}; {} insn {} phis",
                block_addr,
                insns.insns.len(),
                phis.phi_count(),
            )?;

            let print_rdr_count = |f: &mut std::fmt::Formatter, reg| -> std::fmt::Result {
                let rdr_count = self.rdr_count.get(reg);
                if rdr_count > 1 {
                    write!(f, "  ({:3})  ", rdr_count)?;
                } else {
                    write!(f, "         ")?;
                }
                Ok(())
            };

            if phis.phi_count() > 0 {
                write!(f, "                  ɸ  ")?;
                for pred in preds {
                    write!(f, "B{:<5} ", pred.as_number())?;
                }
                writeln!(f)?;
                for phi in phis.phi_regs() {
                    print_rdr_count(f, phi)?;
                    write!(f, "  r{:<5} <- ", phi.0)?;
                    for arg in self.get_phi_args(phi) {
                        write!(f, "r{:<5} ", arg.0)?;
                    }
                    writeln!(f)?;
                }
            }

            for (dest, mut insn) in insns.iter_copied() {
                if self.is_alive(dest) {
                    // modify insn (our copy) so that the registers skip/dereference any Get
                    for input in insn.input_regs_iter_mut() {
                        loop {
                            let input_def = self.get(*input).unwrap().insn.get();
                            if let mil::Insn::Get(x) = input_def {
                                *input = x;
                            } else {
                                break;
                            }
                        }
                    }

                    print_rdr_count(f, dest)?;
                    writeln!(f, "{:?} <- {:?}", dest, insn)?;
                }
            }
        }

        Ok(())
    }
}

// TODO Remove this. No longer needed in the current design. Already cut down to nothing.
pub struct ConversionParams {
    pub program: mil::Program,
}

impl ConversionParams {
    pub fn new(program: mil::Program) -> Self {
        ConversionParams { program }
    }
}

pub fn mil_to_ssa(input: ConversionParams) -> Program {
    let ConversionParams { mut program, .. } = input;

    let cfg = cfg::analyze_mil(&program);

    let dom_tree = cfg.dom_tree();
    let mut phis = place_phi_nodes(&mut program, &cfg, dom_tree);

    /*
    stack[v] is a stack of variable names (for every variable v)

    def rename(block):
      for instr in block:
        replace each argument to instr with stack[old name]

        replace instr's destination with a new name
        push that new name onto stack[old name]

      for s in block's successors:
        for p in s's ϕ-nodes:
          Assuming p is for a variable v, make it read from stack[v].

      for b in blocks immediately dominated by block:
        # That is, children in the dominance tree.
        rename(b)

      pop all the names we just pushed onto the stacks

    rename(entry)
    */

    // TODO rework this with a more efficient data structure
    struct VarMap {
        count: mil::Index,
        stack: Vec<Box<[Option<mil::Reg>]>>,
    }

    impl VarMap {
        fn new(count: mil::Index) -> Self {
            VarMap {
                count,
                stack: Vec::new(),
            }
        }

        fn push(&mut self) {
            if self.stack.is_empty() {
                self.stack
                    .push(vec![None; self.count.into()].into_boxed_slice());
            } else {
                let new_elem = self.stack.last().unwrap().clone();
                self.stack.push(new_elem);
            }

            self.check_invariants()
        }

        fn check_invariants(&self) {
            for elem in &self.stack {
                assert_eq!(elem.len(), self.count.into());
            }
        }

        fn pop(&mut self) {
            self.stack.pop();
            self.check_invariants();
        }

        fn current(&self) -> &[Option<mil::Reg>] {
            &self.stack.last().expect("no mappings!")[..]
        }
        fn current_mut(&mut self) -> &mut [Option<mil::Reg>] {
            &mut self.stack.last_mut().expect("no mappings!")[..]
        }
        fn get(&self, reg: mil::Reg) -> Option<mil::Reg> {
            let reg_num = reg.reg_index() as usize;
            self.current()[reg_num]
        }

        fn set(&mut self, old: mil::Reg, new: mil::Reg) {
            let reg_num = old.reg_index() as usize;
            self.current_mut()[reg_num] = Some(new);
        }
    }

    let var_count = program.reg_count();
    let mut var_map = VarMap::new(var_count);

    enum Cmd {
        Finish,
        Start(cfg::BlockID),
    }
    let mut queue = vec![Cmd::Finish, Cmd::Start(cfg.entry_block_id())];

    while let Some(cmd) = queue.pop() {
        match cmd {
            Cmd::Start(bid) => {
                var_map.push();

                // -- patch current block

                // phi nodes are *used* here for their destination; their arguments are fixed up
                // while processing the predecessors
                let block_phis = &phis[bid];
                for ndx in block_phis.ndxs.clone() {
                    let item = program.get(ndx).unwrap();
                    match item.insn.get() {
                        mil::Insn::Phi { size: 1 }
                        | mil::Insn::Phi { size: 2 }
                        | mil::Insn::Phi { size: 4 }
                        | mil::Insn::Phi { size: 8 } => {
                            var_map.set(item.dest.get(), mil::Reg(ndx));
                            item.dest.set(mil::Reg(ndx));
                        }
                        mil::Insn::PhiArg { .. } => {}
                        _ => panic!("non-phi node in block phi ndx range"),
                    }
                }

                for insn_ndx in cfg.insns_ndx_range(bid) {
                    let iv = program.get(insn_ndx).unwrap();

                    // TODO Re-establish this invariant
                    //  this will only make sense after I add abstract values representing initial
                    //  conditions to each function... (probably coming from the calling
                    //  convention)
                    //
                    // > if bid == cfg::ENTRY_BID && insn_ndx == 0 {
                    // >     assert_eq!(inputs, [None, None]);
                    // > }

                    let mut insn = iv.insn.get();
                    for reg in insn.input_regs_mut().into_iter().flatten() {
                        *reg = var_map.get(*reg).expect("value not initialized in pre-ssa");
                    }
                    iv.insn.set(insn);

                    // in the output SSA, each destination register corrsponds to the instruction's
                    // index. this way, the numeric value of a register can also be used as
                    // instruction ID, to locate a register/variable's defining instruction.
                    let new_dest = mil::Reg(insn_ndx);
                    let old_name = iv.dest.replace(new_dest);
                    let new_name = if let mil::Insn::Get(input_reg) = iv.insn.get() {
                        // exception: for Get(_) instructions, we just reuse the input reg for the
                        // output
                        input_reg
                    } else {
                        new_dest
                    };

                    var_map.set(old_name, new_name);
                }
                for insn_ndx in cfg.insns_ndx_range(bid) {
                    let item = program.get(insn_ndx).unwrap();
                    assert_eq!(item.dest.get(), mil::Reg(insn_ndx));
                }

                // -- patch successor's phi nodes
                // The algorithm is the following:
                // >  for s in block's successors:
                // >    for p in s's ϕ-nodes:
                // >      Assuming p is for a variable v, make it read from stack[v].
                //
                // ... in order to patch the correct operand in each phi instruction, we have
                // to use its "predecessor position", i.e. whether the current block is the
                // successor's 1st, 2nd, 3rd predecessor.

                for (my_pred_ndx, succ) in cfg.block_cont(bid).as_array().into_iter().flatten() {
                    let succ_phis = &mut phis[succ];

                    for phi_ndx in 0..succ_phis.phi_count {
                        let arg = succ_phis.arg_ndx(phi_ndx, my_pred_ndx.into());
                        let iv = program.get(arg.0).unwrap();
                        match iv.insn.get() {
                            // These are not supposed to exist yet!
                            // Only Phi8 is added by place_phi_nodes; the others
                            // are only placed by narrow_phi_nodes
                            mil::Insn::Phi { size: 1 }
                            | mil::Insn::Phi { size: 2 }
                            | mil::Insn::Phi { size: 4 } => {
                                panic!("unexpected narrow phi node at this phase of ssa conversion")
                            }
                            mil::Insn::Phi { size: 8 } => continue,
                            mil::Insn::PhiArg(arg) => {
                                // NOTE: the substitution of the *successor's* phi node's argument is
                                // done in the context of *this* node (its predecessor)
                                let arg_repl = var_map.get(arg).unwrap_or_else(|| {
                                    panic!(
                                        "value {:?} not initialized in pre-ssa (phi {:?}--[{}]-->{:?})",
                                        arg, bid, my_pred_ndx, succ
                                    )
                                });
                                iv.insn.set(mil::Insn::PhiArg(arg_repl));
                            }
                            _ => panic!("non-phi node in block phi ndx range"),
                        };
                    }
                }

                // TODO quadratic, but cfgs are generally pretty small
                let imm_dominated = dom_tree
                    .items()
                    .filter(|(_, parent)| **parent == Some(bid))
                    .map(|(bid, _)| bid);
                for child in imm_dominated {
                    queue.push(Cmd::Finish);
                    queue.push(Cmd::Start(child));
                }
            }

            Cmd::Finish => {
                var_map.pop();
            }
        }
    }
    let var_count = program.len();
    let rdr_count = ReaderCount::new(var_count);

    // establish SSA invariants
    // the returned Program will no longer change (just some instructions are going to be marked as
    // "dead" and ignored)
    for (ndx, insn) in program.iter().enumerate() {
        let ndx = ndx.try_into().unwrap();
        assert_eq!(
            insn.dest.get(),
            mil::Reg(ndx),
            "insn unvisited: {:?} <- {:?}",
            insn.dest,
            insn.insn
        );
    }

    let mut ssa = Program {
        inner: program,
        cfg,
        phis,
        rdr_count,

        #[cfg(feature = "proto_typing")]
        ptr_regs: HashMap::new(),
    };
    eliminate_dead_code(&mut ssa);
    narrow_phi_nodes(&mut ssa);
    ssa
}

/// Add phi nodes to a MIL program, where required.
///
/// Some notes:
///
/// - This function acts on MIL programs not yet in SSA form. Phi nodes are
/// added in preparation for an SSA conversion, but no part of the actual
/// conversion is performed at all.
///
/// - The added Phi nodes are all with the largest result width (Phi8). They
/// must be later swapped for the proper width variants (Phi1, Phi2, Phi4, ...)
/// by calling `narrow_phi_nodes` on the SSA-converted program.
fn place_phi_nodes(
    program: &mut mil::Program,
    cfg: &cfg::Graph,
    dom_tree: &cfg::DomTree,
) -> cfg::BlockMap<PhiInfo> {
    let block_count = cfg.block_count();

    if program.len() == 0 {
        return cfg::BlockMap::new(PhiInfo::empty(), block_count);
    }

    let phis_set = compute_phis_set(program, cfg, dom_tree);
    let var_count = phis_set.0.cols.try_into().unwrap();
    assert_eq!(var_count, program.reg_count());

    // translate `phis` into a representation such that inputs can be replaced with specific input
    // variables assigned in predecessors.
    // they are appended as Phi/PhiArg nodes to the program
    let mut phis = cfg::BlockMap::new(PhiInfo::empty(), block_count);

    for bid in cfg.block_ids() {
        let pred_count: u16 = cfg.block_preds(bid).len().try_into().unwrap();
        let start_ndx = program.len();
        let mut phi_count = 0;

        for var_ndx in 0..var_count {
            let reg = mil::Reg(var_ndx);
            if *phis_set.get(bid, reg) {
                // the largest-width Phi opcode is inserted here; will be
                // narrowed down in a later phase of ssa construction
                program.push(reg, mil::Insn::Phi { size: 8 });
                phi_count += 1;

                for _pred_ndx in 0..pred_count {
                    // implicitly corresponds to _pred_ndx
                    program.push(mil::Reg(program.len()), mil::Insn::PhiArg(reg));
                }
            }
        }

        let end_ndx = program.len();
        assert_eq!(end_ndx - start_ndx, phi_count * (1 + pred_count));

        let phi_info = PhiInfo {
            ndxs: start_ndx..end_ndx,
            phi_count,
            pred_count,
        };
        for phi_ndx in 0..phi_info.phi_count {
            for pred_ndx in 0..phi_info.pred_count {
                // check that it doesn't throw
                let ndx = phi_info.arg_ndx(phi_ndx, pred_ndx);
                let item = program.get(ndx.0).unwrap();
                assert!(matches!(item.insn.get(), mil::Insn::PhiArg(_)));
            }
        }

        phis[bid] = phi_info;
    }

    phis
}

fn compute_phis_set(
    program: &mut mil::Program,
    cfg: &cfg::Graph,
    dom_tree: &cfg::DomTree,
) -> RegMat<bool> {
    // matrix [B * B] where B = #blocks
    // matrix[i, j] = true iff block j is in block i's dominance frontier
    let is_dom_front = compute_dominance_frontier(cfg, dom_tree);
    let block_uses_var = find_received_vars(program, cfg);
    let mut phis_set = RegMat::for_program(program, cfg, false);

    // the rule is:
    //   if variable v is written in block b,
    //   then we have to add `v <- phi v, v, ...` on each block in b's dominance frontier
    // (phis_set is to avoid having multiple phis for the same var)
    for bid in cfg.block_ids() {
        let slice = program.slice(cfg.insns_ndx_range(bid)).unwrap();
        for dest in slice.dests.iter() {
            let dest = dest.get();
            for target_bid in cfg.block_ids() {
                if cfg.block_preds(target_bid).len() < 2 {
                    continue;
                }

                let is_blk_in_df = *is_dom_front.item(bid.as_usize(), target_bid.as_usize());
                let is_used = *block_uses_var.get(target_bid, dest);

                if is_blk_in_df && is_used {
                    phis_set.set(target_bid, dest, true);
                }
            }
        }
    }

    phis_set
}

fn narrow_phi_nodes(program: &mut Program) {
    for iv in program.insns_unordered() {
        let dest = iv.dest.get();
        let insn = iv.insn.get();

        match insn {
            mil::Insn::Phi { size } => {
                if size != 8 {
                    panic!("narrow_phi_nodes: unexpected narrow phi node")
                }
            }
            _ => continue,
        }

        let mut args = program.get_phi_args(dest);
        let arg0 = args.next().unwrap();
        let rt0 = program.value_type(arg0);

        // check: all args have the same type!
        for arg in args {
            assert_eq!(
                program.value_type(arg),
                rt0,
                "malformed ssa: phi node has different type args"
            );
        }

        let repl_insn = match rt0 {
            mil::RegType::Effect => panic!("malformed ssa: phi node can't have Effect inputs"),
            mil::RegType::Bytes(size) => mil::Insn::Phi { size },
            mil::RegType::Bool => mil::Insn::PhiBool,
        };
        iv.insn.set(repl_insn);
    }
}

/// Find the set of "received variables" for each block.
///
/// This is the set of variables which are read before any write in the block.
/// In other words, for these are the variables, the block observes the values
/// left there by other blocks.
fn find_received_vars(prog: &mil::Program, graph: &cfg::Graph) -> RegMat<bool> {
    let mut is_received = RegMat::for_program(prog, graph, false);
    for bid in graph.block_ids_postorder() {
        for (dest, insn) in prog
            .slice(graph.insns_ndx_range(bid))
            .unwrap()
            .iter_copied()
            .rev()
        {
            is_received.set(bid, dest, false);
            for input in insn.input_regs_iter() {
                is_received.set(bid, input, true);
            }
        }
    }
    is_received
}

struct RegMat<T>(Mat<T>);

impl<T: Clone> RegMat<T> {
    fn for_program(program: &mil::Program, graph: &cfg::Graph, value: T) -> Self {
        let var_count = program.reg_count() as usize;
        RegMat(Mat::new(value, graph.block_count() as usize, var_count))
    }

    fn get(&self, bid: cfg::BlockID, reg: mil::Reg) -> &T {
        self.0
            .item(bid.as_number() as usize, reg.reg_index() as usize)
    }

    fn set(&mut self, bid: cfg::BlockID, reg: mil::Reg, value: T) {
        *self
            .0
            .item_mut(bid.as_number() as usize, reg.reg_index() as usize) = value;
    }
}

struct Mat<T> {
    items: Box<[T]>,
    rows: usize,
    cols: usize,
}
impl<T> Mat<T> {
    fn ndx(&self, i: usize, j: usize) -> usize {
        assert!(i < self.rows);
        assert!(j < self.cols);
        self.cols * i + j
    }

    fn item(&self, i: usize, j: usize) -> &T {
        &self.items[self.ndx(i, j)]
    }
    fn item_mut(&mut self, i: usize, j: usize) -> &mut T {
        &mut self.items[self.ndx(i, j)]
    }
}
impl<T: Clone> Mat<T> {
    fn new(init: T, rows: usize, cols: usize) -> Self {
        let items = vec![init; rows * cols].into_boxed_slice();
        Mat { items, rows, cols }
    }
}

fn compute_dominance_frontier(graph: &cfg::Graph, dom_tree: &cfg::DomTree) -> Mat<bool> {
    let count = graph.block_count() as usize;
    let mut mat = Mat::new(false, count, count);

    for bid in graph.block_ids() {
        let preds = graph.block_preds(bid);
        if preds.len() < 2 {
            continue;
        }

        let runner_stop = dom_tree[bid].unwrap();
        for &pred in preds {
            // bid is in the dominance frontier of pred, and of all its dominators
            let mut runner = pred;
            while runner != runner_stop {
                *mat.item_mut(runner.as_usize(), bid.as_usize()) = true;
                runner = dom_tree[runner].unwrap();
            }
        }
    }

    mat
}

pub fn eliminate_dead_code(prog: &mut Program) {
    if prog.inner.len() == 0 {
        return;
    }

    let count = prog.reg_count();
    let mut rdr_count = ReaderCount::new(count);
    let mut visited = vec![false; count as usize];

    let mut queue: Vec<_> = prog
        .inner
        .iter()
        .filter(|iv| iv.insn.get().has_side_effects())
        .map(|iv| iv.dest.get())
        .collect();

    for &side_fx_reg in &queue {
        rdr_count.inc(side_fx_reg);
    }

    while let Some(reg) = queue.pop() {
        // we can assume no circular references
        visited[reg.reg_index() as usize] = true;
        let insn = prog.get(reg).unwrap().insn.get();
        match insn {
            mil::Insn::Phi { size: _ } | mil::Insn::PhiBool => {
                for input in prog.get_phi_args(reg) {
                    rdr_count.inc(input);
                    if !visited[input.reg_index() as usize] {
                        queue.push(input);
                    }
                }
            }
            _ => {
                for input in insn.input_regs_iter() {
                    rdr_count.inc(input);
                    if !visited[input.reg_index() as usize] {
                        queue.push(input);
                    }
                }
            }
        }
    }

    prog.rdr_count = rdr_count;
}

#[derive(Debug, Clone)]
struct ReaderCount(Vec<usize>);
impl ReaderCount {
    fn new(var_count: mil::Index) -> Self {
        ReaderCount(vec![0; var_count as usize])
    }
    fn reset(&mut self) {
        self.0.fill(0);
    }
    fn get(&self, reg: mil::Reg) -> usize {
        self.0[reg.0 as usize]
    }
    fn inc(&mut self, reg: mil::Reg) {
        if let Some(elm) = self.0.get_mut(reg.reg_index() as usize) {
            *elm += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::mil;

    #[test]
    fn test_phi_read() {
        use mil::{ArithOp, Insn, Reg};

        let prog = {
            let mut pb = mil::ProgramBuilder::new();

            pb.set_input_addr(0xf0);
            pb.push(Reg(0), Insn::Const8(123));
            pb.push(
                Reg(1),
                Insn::JmpExtIf {
                    cond: Reg(0),
                    addr: 0xf2,
                },
            );

            pb.set_input_addr(0xf1);
            pb.push(Reg(2), Insn::Const1(4));
            pb.push(Reg(3), Insn::JmpExt(0xf3));

            pb.set_input_addr(0xf2);
            pb.push(Reg(2), Insn::Const1(8));

            pb.set_input_addr(0xf3);
            pb.push(Reg(4), Insn::ArithK1(ArithOp::Add, Reg(2), 456));
            pb.push(Reg(5), Insn::Ret(Reg(4)));

            pb.build()
        };

        eprintln!("-- mil:");
        eprintln!("{:?}", prog);

        eprintln!("-- ssa:");
        let prog = super::mil_to_ssa(super::ConversionParams::new(prog));
        insta::assert_debug_snapshot!(prog);
    }
}
