/// Control Flow Graph.
///
/// Routines and data types to extract and represent the control-flow graph (basic blocks and their
/// sequence relationships).
use std::{collections::HashMap, ops::Range};

use crate::mil;

/// Control Flow Graph
pub struct Graph {
    bounds: Vec<mil::Index>,
    // successors[bndx] = successors to block #bndx
    successors: Vec<BlockCont>,
    block_at: HashMap<mil::Index, BasicBlockID>,
}

#[derive(Debug)]
enum BlockCont {
    End,
    Jmp(BasicBlockID),
    Alt(BasicBlockID, BasicBlockID),
}

#[derive(Debug, Copy, Clone)]
struct BasicBlockID(u16);

pub fn analyze_mil(program: &mil::Program) -> Graph {
    let bounds = {
        let mut bounds = Vec::with_capacity(program.len() / 5);
        bounds.push(0);

        for ndx in 0..program.len() {
            match dest_of_insn(program, ndx) {
                // (straight_dest, side_dest)
                (None, None) => panic!("all instructions must go *somewhere*!"),
                (_, None) => {
                    // here we don't care about straight_dest, as it's the default and does not influence
                    // basic block structure
                }
                (None, Some(a)) => {
                    bounds.push(a);
                }
                (Some(a), Some(b)) => {
                    bounds.push(a);
                    bounds.push(b);
                }
            }
        }

        bounds.push(program.len());
        bounds.sort();
        bounds.dedup();
        bounds
    };

    let block_count = bounds.len() - 1;

    let block_at: HashMap<_, _> = bounds[..block_count]
        .iter()
        .enumerate()
        .map(|(bndx, start_ndx)| {
            let bndx = bndx.try_into().unwrap();
            (*start_ndx, BasicBlockID(bndx))
        })
        .collect();

    // if bounds == [b0, b1, b2, b3, ...]
    // then the basic blocks span instructions at indices [b0, b1), [b1, b2), [b2, b3), ...
    // in particular, basic block at index bndx spans [bounds[bndx], bounds[bndx+1])

    let successors = bounds[1..]
        .iter()
        .map(|end_ndx| {
            let last_ndx = end_ndx - 1;
            match dest_of_insn(program, last_ndx) {
                (None, None) => panic!("all instructions must lead *somewhere*!"),
                (None, Some(dest)) if dest == program.len() => BlockCont::End,
                (Some(dest), None) | (None, Some(dest)) => {
                    BlockCont::Jmp(*block_at.get(&dest).unwrap())
                }
                (Some(straight_dest), Some(side_dest)) => BlockCont::Alt(
                    *block_at.get(&straight_dest).unwrap(),
                    *block_at.get(&side_dest).unwrap(),
                ),
            }
        })
        .collect();

    #[cfg(debug_assertions)]
    {
        let is_covered = {
            let mut is = vec![false; program.len()];
            for (&start_ndx, &end_ndx) in bounds.iter().zip(bounds[1..].iter()) {
                for i in start_ndx..end_ndx {
                    is[i] = true;
                }
            }
            is
        };

        let uncovered: Vec<_> = is_covered
            .into_iter()
            .enumerate()
            .filter(|(_, is)| !*is)
            .map(|(ndx, _)| ndx)
            .collect();

        // this way if the assertion fails we can see which indices are uncovered
        debug_assert_eq!(&uncovered, &[]);

        let invalid: Vec<_> = bounds
            .iter()
            .copied()
            .filter(|&i| !(i <= program.len()))
            .collect();
        debug_assert_eq!(&invalid, &[]);

        // no duplicates (<=> no 0-length blocks)
        debug_assert!(bounds.iter().zip(bounds[1..].iter()).all(|(a, b)| a != b));
    }

    Graph {
        bounds,
        block_at,
        successors,
    }
}

fn dest_of_insn(
    program: &mil::Program,
    ndx: mil::Index,
) -> (Option<mil::Index>, Option<mil::Index>) {
    let insn = program.get(ndx).unwrap().insn;
    match insn {
        mil::Insn::JmpK(target) => {
            let index = program.index_of_addr(*target).unwrap();
            (None, Some(index))
        }
        mil::Insn::JmpIfK { target, .. } => {
            let index = program.index_of_addr(*target).unwrap();
            (Some(ndx + 1), Some(index))
        }
        mil::Insn::Ret(_) => {
            // one-past-the-end of the program is a valid index; signifies "exit the function"
            (None, Some(program.len()))
        }
        mil::Insn::Jmp(_) => todo!("indirect jmp"),
        _ => (Some(ndx + 1), None),
    }
}

impl Graph {
    #[inline(always)]
    pub fn block_count(&self) -> usize {
        self.bounds.len() - 1
    }

    pub fn dump(&self, program: &mil::Program) {
        let count = self.block_count();
        println!("{:4} blocks", count);
        for bndx in 0..count {
            let start_ndx = self.bounds[bndx];
            let start_addr = 0; //program.get(start_ndx).unwrap().addr;

            let end_ndx = self.bounds[bndx + 1];
            let end_addr = 0; // program.get(end_ndx).unwrap().addr;

            println!(
                "  #{}  {}(0x{:x}) ..= {}(0x{:x}) -> {:?}",
                bndx, start_ndx, start_addr, end_ndx, end_addr, self.successors[bndx],
            );
        }
    }
}
