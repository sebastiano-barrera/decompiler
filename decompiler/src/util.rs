use crate::mil::Endianness;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, facet::Facet)]
pub struct Bytes {
    data: [u8; Bytes::CAPACITY],
    len: u8,
}

impl Bytes {
    pub const CAPACITY: usize = 8;

    pub fn empty() -> Self {
        Self {
            data: [0u8; Bytes::CAPACITY],
            len: 0,
        }
    }

    pub fn from_slice(slice: &[u8]) -> Option<Self> {
        if slice.len() > Self::CAPACITY {
            return None;
        }
        let mut data = [0u8; Self::CAPACITY];
        data[0..slice.len()].copy_from_slice(slice);
        Some(Self {
            data,
            len: slice.len().try_into().unwrap(),
        })
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data[0..self.len()]
    }
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        let len = self.len();
        &mut self.data[0..len]
    }
}

/// A data structure storing a f32 as raw bytes, so that it is Hash, PartialEq, Eq.
///
/// The endianness in which the f32 value is expressed is also tracked.
///
/// The stored f32 value can be retrieved via [Float32Bytes::value].
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug, facet::Facet)]
pub struct Float32Bytes {
    bytes: [u8; 4],
    endianness: Endianness,
}

impl Float32Bytes {
    pub fn from_bytes(bytes: [u8; 4], endianness: Endianness) -> Self {
        Self { bytes, endianness }
    }
    pub fn value(&self) -> f32 {
        match self.endianness {
            Endianness::Little => f32::from_le_bytes(self.bytes),
            Endianness::Big => f32::from_be_bytes(self.bytes),
        }
    }
}

/// A data structure storing a f64 as raw bytes, so that it is Hash, PartialEq, Eq.
///
/// The endianness in which the f64 value is expressed is also tracked.
///
/// The stored f64 value can be retrieved via [Float64Bytes::value].
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug, facet::Facet)]
pub struct Float64Bytes {
    bytes: [u8; 8],
    endianness: Endianness,
}

impl Float64Bytes {
    pub fn from_bytes(bytes: [u8; 8], endianness: Endianness) -> Self {
        Self { bytes, endianness }
    }
    pub fn value(&self) -> f64 {
        match self.endianness {
            Endianness::Little => f64::from_le_bytes(self.bytes),
            Endianness::Big => f64::from_be_bytes(self.bytes),
        }
    }
}
