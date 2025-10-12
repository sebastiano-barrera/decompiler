use std::collections::HashMap;

// these types are explicitly selected as "OK for the protocol":
// they're simple, it's fine to let them "leak" into the serialized structures.
// they're unlikely to change in unforeseen manners, and it saves some work to build
// out the conversion.

pub use crate::cfg::BlockID;
pub use crate::mil::RegType;
pub use crate::ty::TypeID;
pub use crate::ExpandedInsn;

#[derive(Debug, serde::Serialize)]
pub struct Document {
    pub mil: Option<MIL>,
    pub ssa: Option<SSA>,
}
impl From<&crate::DecompiledFunction> for Document {
    fn from(df: &crate::DecompiledFunction) -> Self {
        let mil: Option<MIL> = df.mil().map(Into::into);
        let ssa: Option<SSA> = df.ssa().map(Into::into);
        Document { mil, ssa }
    }
}

#[derive(Debug, serde::Serialize)]
pub struct MIL {
    pub body: Vec<Insn>,
}
impl From<&crate::mil::Program> for MIL {
    fn from(mil: &crate::mil::Program) -> Self {
        let mut body = Vec::new();
        for ndx in 0..mil.len() {
            let iv = mil.get(ndx).unwrap();
            body.push(Insn {
                addr: Some(iv.addr),
                dest: iv.dest.get().reg_index(),
                insn: crate::mil::to_expanded(&iv.insn.get()),
                tyid: None,
                reg_type: None,
            });
        }

        MIL { body }
    }
}

#[derive(Debug, serde::Serialize)]
pub struct SSA {
    pub blocks: HashMap<BlockID, SSABlock>,
}
impl From<&crate::ssa::Program> for SSA {
    fn from(ssa: &crate::ssa::Program) -> Self {
        let mut body_of_block = HashMap::new();

        for bid in ssa.cfg().block_ids() {
            let cont = ssa.cfg().block_cont(bid);

            let mut body = Vec::new();
            for reg in ssa.block_regs(bid) {
                let insn = crate::mil::to_expanded(&ssa.get(reg).unwrap());
                body.push(Insn {
                    addr: None,
                    dest: reg.reg_index(),
                    insn,
                    tyid: ssa.value_type(reg),
                    reg_type: Some(ssa.reg_type(reg)),
                });
            }

            body_of_block.insert(
                bid,
                SSABlock {
                    body,
                    cont: cont.into(),
                },
            );
        }

        SSA {
            blocks: body_of_block,
        }
    }
}

#[derive(Debug, serde::Serialize)]
pub struct SSABlock {
    pub body: Vec<Insn>,
    pub cont: BlockCont,
}

#[derive(Debug, serde::Serialize)]
pub struct Insn {
    pub addr: Option<u64>,
    pub dest: u16,
    pub insn: ExpandedInsn,
    pub tyid: Option<TypeID>,
    pub reg_type: Option<RegType>,
}

#[derive(Debug, serde::Serialize)]
pub enum BlockCont {
    Always(Dest),
    Conditional { pos: Dest, neg: Dest },
}
impl From<crate::cfg::BlockCont> for BlockCont {
    fn from(value: crate::cfg::BlockCont) -> Self {
        match value {
            crate::cfg::BlockCont::Always(dest) => BlockCont::Always(dest.into()),
            crate::cfg::BlockCont::Conditional { pos, neg } => BlockCont::Conditional {
                pos: pos.into(),
                neg: neg.into(),
            },
        }
    }
}

#[derive(Debug, serde::Serialize)]
pub enum Dest {
    Ext(u64),
    Block(BlockID),
    Indirect,
    Return,
    Undefined,
}

impl From<crate::cfg::Dest> for Dest {
    fn from(value: crate::cfg::Dest) -> Self {
        match value {
            crate::cfg::Dest::Ext(addr) => Dest::Ext(addr),
            crate::cfg::Dest::Block(block_id) => Dest::Block(block_id),
            crate::cfg::Dest::Indirect => Dest::Indirect,
            crate::cfg::Dest::Return => Dest::Return,
            crate::cfg::Dest::Undefined => Dest::Undefined,
        }
    }
}
