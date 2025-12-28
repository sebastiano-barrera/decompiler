use std::fmt::Debug;
use std::panic::UnwindSafe;

pub(crate) trait BisectState: Sized + Debug {
    fn split(&self) -> (Self, Self);
    fn is_terminal(&self) -> bool;
}

pub(crate) fn bisect_find_panic<S, F>(init_state: S, action: F) -> Vec<S>
where
    S: BisectState + UnwindSafe,
    for<'a> &'a S: UnwindSafe,
    F: Fn(&S) + UnwindSafe,
    for<'a> &'a F: UnwindSafe,
{
    let mut terminals = Vec::new();

    let mut work = vec![init_state];
    while let Some(state) = work.pop() {
        let result = std::panic::catch_unwind(|| {
            action(&state);
        });

        if result.is_err() {
            if state.is_terminal() {
                terminals.push(state);
            } else {
                let (left, right) = state.split();
                work.push(left);
                work.push(right);
            }
        }
    }

    terminals
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq)]
    struct Range {
        lo: usize,
        hi: usize,
    }

    impl std::fmt::Debug for Range {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}..{}", self.lo, self.hi)
        }
    }

    impl BisectState for Range {
        fn split(&self) -> (Self, Self) {
            let Range { lo, hi } = *self;
            let mid = lo + (hi - lo) / 2;
            let left = Range { lo, hi: mid };
            let right = Range { lo: mid, hi };
            (left, right)
        }

        fn is_terminal(&self) -> bool {
            self.hi - self.lo <= 1
        }
    }

    fn panick_on_3(range: &Range) {
        let &Range { lo, hi } = range;
        if lo <= 3 && hi > 3 {
            panic!("FOUND ME with {range:?}");
        }
    }

    #[test]
    fn single() {
        let bad_ranges = bisect_find_panic(Range { lo: 0, hi: 10 }, |range| panick_on_3(range));
        assert_eq!(bad_ranges.as_slice(), &[Range { lo: 3, hi: 4 }]);
    }

    fn panick_on_3_and_5(&Range { lo, hi }: &Range) {
        if (lo <= 3 && hi > 3) || (lo <= 5 && hi > 5) {
            panic!("FOUND ME");
        }
    }

    #[test]
    fn multiple() {
        let bad_range =
            bisect_find_panic(Range { lo: 0, hi: 10 }, |range| panick_on_3_and_5(range));
        assert!(bad_range.contains(&Range { lo: 3, hi: 4 }));
        assert!(bad_range.contains(&Range { lo: 5, hi: 6 }));
        assert_eq!(bad_range.len(), 2);
    }
}
