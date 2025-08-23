#[derive(Default)]
pub struct Warnings(Vec<Box<dyn std::error::Error>>);

impl Warnings {
    pub fn add(&mut self, warn: Box<dyn std::error::Error>) {
        self.0.push(warn);
    }

    pub fn into_vec(self) -> Vec<Box<dyn std::error::Error>> {
        self.0
    }
}

impl std::fmt::Debug for Warnings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_empty() {
            writeln!(f, "0 warnings.")
        } else {
            writeln!(f, "{} warnings:", self.0.len())?;
            for (ndx, warn) in self.0.iter().enumerate() {
                writeln!(f, "  #{:4}: {}", ndx, warn)?;

                let mut source = warn.source();
                while let Some(cur) = source {
                    writeln!(f, "           <- {}", cur)?;
                    source = cur.source();
                }
            }
            Ok(())
        }
    }
}

pub trait ToWarnings {
    type Ok;
    fn or_warn(self, warnings: &mut Warnings) -> Option<Self::Ok>;
}

impl<T> ToWarnings for std::result::Result<T, anyhow::Error> {
    type Ok = T;

    fn or_warn(self, warnings: &mut Warnings) -> Option<T> {
        match self {
            Ok(value) => Some(value),
            Err(err) => {
                warnings.add(err.into());
                None
            }
        }
    }
}

#[macro_export]
macro_rules! match_get {
    ($expr:expr, $pat:pat, $extract:expr) => {
        match $expr {
            $pat => Some($extract),
            _ => None,
        }
    };
}

/// A generic representation of a tree where each node is logically numbered,
/// such that:
///
/// - there is exactly one root (it's not a forest or anything);
///
/// - a "less than" relationship is defined between keys;
///
/// - every node has a number strictly less than the number of all of
///   its children.
///
/// Maintaining the above laws/invariants is the impls' responsibility.
pub trait NumberedTree {
    type Key: Eq + Copy;
    fn parent_of(&self, k: &Self::Key) -> Option<Self::Key>;
    fn key_lt(&self, a: Self::Key, b: Self::Key) -> bool;
}

/// Find the common ancestor of two nodes in a tree.
///
/// The two nodes are identified by the `ka` and `kb` keys.
///
/// The tree is accessed through the NumberedTree trait. (See [`NumberedTree`]
/// for its basic law, which this algorithm relies on.)
pub fn common_ancestor<T: NumberedTree>(
    tree: &T,
    mut ka: T::Key,
    mut kb: T::Key,
) -> Option<T::Key> {
    // If there are multiple entry points, we will never have ndx_a == ndx_b.
    // Rather, we would walk through parent_of infinitely, so we have to add
    // specific termination conditions for those.

    while ka != kb {
        let is_root_a = tree.parent_of(&ka).is_none();
        let is_root_b = tree.parent_of(&kb).is_none();

        while !is_root_b && tree.key_lt(ka, kb) {
            kb = tree.parent_of(&kb).unwrap();
        }
        while !is_root_a && tree.key_lt(kb, ka) {
            ka = tree.parent_of(&ka).unwrap();
        }

        // When the graph has multiple entry nodes, each `e` of them has
        // tree.parent_of(e) == None. If ka and kb represent two such nodes,
        // we're never going to find a common ancestor, and we must stop (or we
        // go into an infinite loop).
        if ka != kb && is_root_a && is_root_b {
            return None;
        }
    }

    Some(ka)
}

#[cfg(feature = "trace")]
#[macro_export]
macro_rules! traceln {
    ($($toks:tt)*) => {
        println!($($toks)*);
    }
}

#[cfg(feature = "trace")]
#[macro_export]
macro_rules! trace {
    ($($toks:tt)*) => {
        print!($($toks)*);
    }
}

#[cfg(not(feature = "trace"))]
#[macro_export]
macro_rules! trace {
    ($($toks:tt)*) => {};
}

#[cfg(not(feature = "trace"))]
#[macro_export]
macro_rules! traceln {
    ($($toks:tt)*) => {};
}
