#[derive(Debug)]
pub struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl DisjointSet {
    pub fn new(size: usize) -> Self {
        let mut parent = Vec::with_capacity(size);
        for i in 0..size {
            parent.push(i);
        }
        let rank = vec![0; size];
        Self { parent, rank }
    }

    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    pub fn union(&mut self, x: usize, y: usize) {
        let root_x = self.find(x);
        let root_y = self.find(y);
        if root_x != root_y {
            if self.rank[root_x] < self.rank[root_y] {
                self.parent[root_x] = root_y;
            } else if self.rank[root_x] > self.rank[root_y] {
                self.parent[root_y] = root_x;
            } else {
                self.parent[root_y] = root_x;
                self.rank[root_x] += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let ds = DisjointSet::new(5);
        assert_eq!(ds.parent.len(), 5);
        assert_eq!(ds.rank.len(), 5);
        for i in 0..5 {
            assert_eq!(ds.parent[i], i);
            assert_eq!(ds.rank[i], 0);
        }
    }

    #[test]
    fn test_find() {
        let mut ds = DisjointSet::new(5);
        assert_eq!(ds.find(0), 0);
        assert_eq!(ds.find(4), 4);
    }

    #[test]
    fn test_union() {
        let mut ds = DisjointSet::new(5);
        ds.union(0, 1);
        assert_eq!(ds.find(0), ds.find(1));

        ds.union(2, 3);
        assert_eq!(ds.find(2), ds.find(3));

        ds.union(0, 2);
        assert_eq!(ds.find(0), ds.find(2));
        assert_eq!(ds.find(1), ds.find(3));
    }

    #[test]
    fn test_no_union_same_set() {
        let mut ds = DisjointSet::new(3);
        ds.union(0, 1);
        let root_before = ds.find(0);
        ds.union(0, 1); // should not change
        assert_eq!(ds.find(0), root_before);
    }
}
