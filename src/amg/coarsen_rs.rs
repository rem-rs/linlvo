//! Ruge–Stüben (RS) classical AMG coarsening.
//!
//! Assigns each DOF to either a C-point (coarse) or F-point (fine) using the
//! standard RS algorithm:
//!
//! 1. Build the strength-of-connection graph S and its transpose Sᵀ.
//! 2. Compute a "lambda" measure λ(i) = |{j : i ∈ S(j)}| (number of points
//!    that strongly depend on i).
//! 3. Mark as C-point the undecided node with largest λ; mark all its
//!    strongly-dependent F-neighbours as F.
//! 4. Repeat until all nodes are decided.
//!
//! **Reference**: Ruge & Stüben 1987; Stuben 2001 (survey).

use crate::core::scalar::Scalar;
use crate::sparse::CsrMatrix;

/// Node classification.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodeType {
    Undecided,
    Coarse,
    Fine,
}

/// Perform RS coarsening.
///
/// Returns a `Vec<NodeType>` of length `n` classifying each DOF.
pub fn rs_coarsen<T: Scalar>(s: &CsrMatrix<T>) -> Vec<NodeType> {
    let n   = s.nrows();
    let rp  = s.row_ptr();
    let ci  = s.col_idx();

    // Build Sᵀ (transpose of S) to get "who strongly influences i".
    let st  = s.transpose_csr();
    let trp = st.row_ptr();
    let tci = st.col_idx();

    // λ(i) = number of undecided nodes that strongly depend on i.
    let mut lam: Vec<i32> = (0..n)
        .map(|i| (trp[i + 1] - trp[i]) as i32)
        .collect();

    let mut status = vec![NodeType::Undecided; n];

    loop {
        // Find undecided node with maximum λ.
        let best = (0..n)
            .filter(|&i| status[i] == NodeType::Undecided)
            .max_by_key(|&i| lam[i]);

        let c = match best { Some(c) => c, None => break };

        status[c] = NodeType::Coarse;

        // All undecided nodes that strongly depend on c → Fine.
        for k in trp[c]..trp[c + 1] {
            let j = tci[k];
            if status[j] == NodeType::Undecided {
                status[j] = NodeType::Fine;
                // Increase λ for undecided nodes that strongly depend on j.
                for kk in trp[j]..trp[j + 1] {
                    let m = tci[kk];
                    if status[m] == NodeType::Undecided {
                        lam[m] += 1;
                    }
                }
            }
        }

        // Decrease λ for undecided nodes strongly depending on c.
        for k in rp[c]..rp[c + 1] {
            let j = ci[k];
            if status[j] == NodeType::Undecided {
                lam[j] -= 1;
            }
        }
    }

    status
}

/// Count C-points and build a mapping: fine_index → coarse_index (or usize::MAX).
pub fn coarse_index_map(status: &[NodeType]) -> (usize, Vec<usize>) {
    let mut map = vec![usize::MAX; status.len()];
    let mut nc  = 0;
    for (i, &s) in status.iter().enumerate() {
        if s == NodeType::Coarse {
            map[i] = nc;
            nc += 1;
        }
    }
    (nc, map)
}
