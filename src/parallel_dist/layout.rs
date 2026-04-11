use std::ops::Range;

/// Partition metadata for one rank/subdomain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionLayout {
    /// Global problem size.
    pub global_size: usize,
    /// Half-open owned range `[start, end)` in global indexing.
    pub owned_global_range: Range<usize>,
    /// Ghost global indices needed from neighbor partitions.
    pub ghost_globals: Vec<usize>,
}

impl PartitionLayout {
    /// Build a partition layout and validate basic invariants.
    pub fn new(
        global_size: usize,
        owned_global_range: Range<usize>,
        ghost_globals: Vec<usize>,
    ) -> Result<Self, String> {
        if owned_global_range.start > owned_global_range.end {
            return Err("owned range start must be <= end".into());
        }
        if owned_global_range.end > global_size {
            return Err("owned range end must be <= global size".into());
        }
        if ghost_globals.iter().any(|&g| g >= global_size) {
            return Err("ghost index out of bounds".into());
        }
        Ok(Self {
            global_size,
            owned_global_range,
            ghost_globals,
        })
    }

    /// Number of owned DOFs on this partition.
    pub fn local_size(&self) -> usize {
        self.owned_global_range.end - self.owned_global_range.start
    }

    /// Number of ghost DOFs referenced by this partition.
    pub fn ghost_size(&self) -> usize { self.ghost_globals.len() }

    /// Whether this partition owns `gidx`.
    pub fn owns_global(&self, gidx: usize) -> bool {
        self.owned_global_range.contains(&gidx)
    }
}

/// Block partition helper.
///
/// Returns `[start, end)` for `rank` in `[0, nranks)`.
pub fn block_partition(global_size: usize, nranks: usize, rank: usize) -> Range<usize> {
    assert!(nranks > 0, "nranks must be positive");
    assert!(rank < nranks, "rank out of range");

    let q = global_size / nranks;
    let r = global_size % nranks;

    let start = rank * q + rank.min(r);
    let len = q + usize::from(rank < r);
    start..(start + len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_partition_covers_whole_domain() {
        let n = 10;
        let p = 3;
        let mut marks = vec![0usize; n];
        for rank in 0..p {
            let rg = block_partition(n, p, rank);
            for i in rg {
                marks[i] += 1;
            }
        }
        assert!(marks.iter().all(|&m| m == 1));
    }

    #[test]
    fn layout_validates_bounds() {
        assert!(PartitionLayout::new(10, 3..7, vec![1, 2, 8]).is_ok());
        assert!(PartitionLayout::new(10, 8..12, vec![]).is_err());
    }
}
