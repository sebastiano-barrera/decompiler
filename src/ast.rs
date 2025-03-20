use crate::{
    pp::PP,
    ssa::{self, ControlNID, DataNID},
};

pub fn pretty_print<W: PP + ?Sized>(pp: &mut W, prog: &ssa::Program) -> std::io::Result<()> {
    Ast::new(pp, prog).pretty_print()
}

struct Ast<'a, W: PP + ?Sized> {
    pp: &'a mut W,
    ssa: &'a ssa::Program,
    data_uses: ssa::DataUseGraph,
    control_uses: ssa::ControlUseGraph,
    control_visited: slotmap::SecondaryMap<ControlNID, ()>,
}

type IoResult = std::io::Result<()>;

// TODO treat phi nodes specially; make them appear as block parameters

impl<'a, W: PP + ?Sized> Ast<'a, W> {
    pub fn new(pp: &'a mut W, ssa: &'a ssa::Program) -> Self {
        let data_uses = ssa.compute_data_use_graph();
        let control_uses = ssa.compute_control_use_graph();
        Ast {
            pp,
            ssa,
            data_uses,
            control_uses,
            control_visited: slotmap::SecondaryMap::new(),
        }
    }

    pub fn pretty_print(&mut self) -> IoResult {
        self.pp_control(self.ssa.start_cnid())
    }

    fn pp_control(&mut self, cnid: ControlNID) -> IoResult {
        if self.control_visited.contains_key(cnid) {
            return Ok(());
        }
        self.control_visited.insert(cnid, ());

        let multiple_preds = self.ssa.get_control(cnid).unwrap().predecessors().len() > 1;

        if multiple_preds {
            write!(self.pp, "{:?} {{\n  ", cnid)?;
            self.pp.open_box();
            self.pp_control_direct(cnid)?;
            self.pp.close_box();
            writeln!(self.pp, "}}")?;
        } else {
            self.pp_control_direct(cnid)?;
        }
        Ok(())
    }

    fn pp_control_direct(&mut self, cnid: ControlNID) -> IoResult {
        // TODO switch from recursion to iteration

        let cn = self.ssa.get_control(cnid).unwrap();

        let data_inputs = cn.data_inputs();

        for input_dnid in &data_inputs {
            if self.is_data_named(input_dnid) {
                self.pp_let(*input_dnid)?;
            }
        }

        write!(self.pp, "{} (", cn.implicit_repr())?;

        for (ndx, input_dnid) in data_inputs.into_iter().enumerate() {
            if ndx > 0 {
                write!(self.pp, ", ")?;
            }

            if self.is_data_named(input_dnid) {
                write!(self.pp, "{:?}", input_dnid)?;
            } else {
                self.pp_expr(*input_dnid)?;
            }
        }

        write!(self.pp, ")")?;

        // TODO remove this allocation
        let control_uses = self.control_uses.get(cnid).unwrap().control.clone();
        if cn.is_branch() {
            // TODO fix this mess
            // we're relying on this node having at most two successors, at most
            // one cons, and at most one alt
            for succ_cnid in control_uses {
                let succ = self.ssa.get_control(succ_cnid).unwrap();
                if succ.is_branch_cons_of(cnid) {
                    write!(self.pp, " {{\n  ")?;
                    self.pp.open_box();
                } else if succ.is_branch_alt_of(cnid) {
                    write!(self.pp, " else {{\n  ")?;
                } else {
                    assert!(false, "successor of branch is neither alt nor cons");
                }

                self.pp_control_direct(succ_cnid)?;
                self.pp.close_box();
                write!(self.pp, "}}")?;
            }
        } else {
            for succ_cnid in control_uses {
                writeln!(self.pp)?;
                self.pp_control(succ_cnid)?;
            }
        }

        writeln!(self.pp)
    }

    fn is_data_named(&mut self, input_dnid: &DataNID) -> bool {
        self.data_uses.get(*input_dnid).unwrap().total() > 1
    }

    fn pp_let(&mut self, dnid: DataNID) -> IoResult {
        write!(self.pp, "let {:?} = ", dnid)?;
        self.pp_expr(dnid)
    }
    fn pp_expr(&mut self, dnid: DataNID) -> IoResult {
        self.pp.open_box();

        let dn = self.ssa.get_data(dnid).unwrap();
        let data_inputs = dn.data_inputs();

        write!(self.pp, "{} (", dn.implicit_repr())?;

        for (ndx, input_dnid) in data_inputs.into_iter().enumerate() {
            if ndx > 0 {
                write!(self.pp, ", ")?;
            }

            if self.is_data_named(input_dnid) {
                write!(self.pp, "{:?}", input_dnid)?;
            } else {
                self.pp_expr(*input_dnid)?;
            }
        }

        write!(self.pp, ")")?;

        self.pp.close_box();
        Ok(())
    }
}
