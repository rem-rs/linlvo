//! Matrix Market (.mtx) file reader.
//!
//! Supports the coordinate-format Matrix Market exchange format as used by the
//! SuiteSparse Matrix Collection (formerly University of Florida Sparse Matrix
//! Collection).
//!
//! Supported header variants:
//! - `%%MatrixMarket matrix coordinate real general`
//! - `%%MatrixMarket matrix coordinate real symmetric`
//! - `%%MatrixMarket matrix coordinate real skew-symmetric`
//! - `%%MatrixMarket matrix coordinate pattern general`
//! - `%%MatrixMarket matrix coordinate pattern symmetric`
//! - `%%MatrixMarket matrix coordinate integer general`
//! - `%%MatrixMarket matrix coordinate integer symmetric`
//!
//! All indices are 1-based in the file format; this reader converts them to
//! 0-based on output.

use std::io::{self, BufRead};
use std::path::Path;

use crate::sparse::{CooMatrix, CsrMatrix};

// ─── error type ───────────────────────────────────────────────────────────────

/// Errors that can occur while parsing a Matrix Market file.
#[derive(Debug, thiserror::Error)]
pub enum MmioError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("missing %%MatrixMarket header")]
    MissingHeader,

    #[error("unsupported matrix type: {0}")]
    UnsupportedType(String),

    #[error("unsupported field type: {0}")]
    UnsupportedField(String),

    #[error("unsupported symmetry: {0}")]
    UnsupportedSymmetry(String),

    #[error("expected object 'matrix', got: {0}")]
    NotMatrix(String),

    #[error("expected format 'coordinate', got: {0}")]
    NotCoordinate(String),

    #[error("malformed size line: {0}")]
    MalformedSizeLine(String),

    #[error("malformed data line: {0}")]
    MalformedDataLine(String),

    #[error("index out of bounds: row {row}, col {col}, size {nrows}x{ncols}")]
    IndexOutOfBounds { row: usize, col: usize, nrows: usize, ncols: usize },

    #[error("matrix must be square for symmetric/skew-symmetric format")]
    NonSquareSymmetric,
}

// ─── symmetry / field enums ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Symmetry {
    General,
    Symmetric,
    SkewSymmetric,
    Hermitian,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Field {
    Real,
    Integer,
    Pattern,
    Complex,
}

// ─── public API ───────────────────────────────────────────────────────────────

/// Read a Matrix Market file from disk and return a CSR matrix.
///
/// For symmetric/skew-symmetric matrices the lower triangle is stored in the
/// file; this function expands them to the full symmetric/skew-symmetric matrix.
///
/// Indices in the file are 1-based; the returned matrix uses 0-based indices.
pub fn read_matrix_market<P: AsRef<Path>>(path: P) -> Result<CsrMatrix<f64>, MmioError> {
    let file = std::fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    parse_matrix_market(reader)
}

/// Read a Matrix Market file from disk and return a COO matrix.
pub fn read_matrix_market_coo<P: AsRef<Path>>(path: P) -> Result<CooMatrix<f64>, MmioError> {
    let file = std::fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    parse_matrix_market_coo(reader)
}

/// Parse a Matrix Market document from a string slice.
///
/// Useful for testing without writing temporary files to disk.
pub fn read_matrix_market_str(s: &str) -> Result<CsrMatrix<f64>, MmioError> {
    parse_matrix_market(io::Cursor::new(s))
}

/// Parse a Matrix Market document from a string slice, returning COO.
pub fn read_matrix_market_coo_str(s: &str) -> Result<CooMatrix<f64>, MmioError> {
    parse_matrix_market_coo(io::Cursor::new(s))
}

// ─── internal parsing ─────────────────────────────────────────────────────────

fn parse_matrix_market<R: BufRead>(reader: R) -> Result<CsrMatrix<f64>, MmioError> {
    Ok(CsrMatrix::from_coo(&parse_matrix_market_coo(reader)?))
}

fn parse_matrix_market_coo<R: BufRead>(reader: R) -> Result<CooMatrix<f64>, MmioError> {
    let mut lines = reader.lines();

    // ── Parse the header line ──────────────────────────────────────────────
    let header = loop {
        match lines.next() {
            None => return Err(MmioError::MissingHeader),
            Some(l) => {
                let l = l?;
                if l.starts_with("%%MatrixMarket") || l.starts_with("%MatrixMarket") {
                    break l;
                }
                // A non-comment, non-header first line → error.
                if !l.starts_with('%') {
                    return Err(MmioError::MissingHeader);
                }
            }
        }
    };

    let (field, symmetry) = parse_header(&header)?;

    // ── Skip comment lines, find size line ────────────────────────────────
    let (nrows, ncols, nnz_declared) = loop {
        match lines.next() {
            None => return Err(MmioError::MalformedSizeLine("unexpected EOF".into())),
            Some(l) => {
                let l = l?;
                let trimmed = l.trim();
                if trimmed.starts_with('%') || trimmed.is_empty() {
                    continue;
                }
                break parse_size_line(trimmed)?;
            }
        }
    };

    if (symmetry == Symmetry::Symmetric || symmetry == Symmetry::SkewSymmetric || symmetry == Symmetry::Hermitian)
        && nrows != ncols
    {
        return Err(MmioError::NonSquareSymmetric);
    }

    // ── Read data lines ───────────────────────────────────────────────────
    let mut coo = CooMatrix::new(nrows, ncols);

    for line_result in lines {
        let line = line_result?;
        let trimmed = line.trim();
        if trimmed.starts_with('%') || trimmed.is_empty() {
            continue;
        }

        let (row0, col0, val) = parse_data_line(trimmed, field)?;

        // Validate bounds (1-based indices already decremented).
        if row0 >= nrows || col0 >= ncols {
            return Err(MmioError::IndexOutOfBounds {
                row: row0,
                col: col0,
                nrows,
                ncols,
            });
        }

        coo.push(row0, col0, val);

        match symmetry {
            Symmetry::Symmetric if row0 != col0 => {
                coo.push(col0, row0, val);
            }
            Symmetry::SkewSymmetric if row0 != col0 => {
                coo.push(col0, row0, -val);
            }
            Symmetry::Hermitian if row0 != col0 => {
                // For real matrices hermitian == symmetric.
                coo.push(col0, row0, val);
            }
            _ => {}
        }
    }

    let _ = nnz_declared; // declared nnz is informational; actual count may differ after expansion
    Ok(coo)
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Parse the `%%MatrixMarket matrix coordinate <field> <symmetry>` header line.
fn parse_header(line: &str) -> Result<(Field, Symmetry), MmioError> {
    // Tokens after the `%%MatrixMarket` banner.
    let tokens: Vec<&str> = line.split_whitespace().collect();
    // tokens[0] = "%%MatrixMarket"  (or "%MatrixMarket")
    // tokens[1] = object  (must be "matrix")
    // tokens[2] = format  (must be "coordinate")
    // tokens[3] = field
    // tokens[4] = symmetry
    if tokens.len() < 5 {
        return Err(MmioError::MissingHeader);
    }

    let object = tokens[1].to_ascii_lowercase();
    if object != "matrix" {
        return Err(MmioError::NotMatrix(object));
    }

    let format = tokens[2].to_ascii_lowercase();
    if format != "coordinate" {
        return Err(MmioError::NotCoordinate(format));
    }

    let field = match tokens[3].to_ascii_lowercase().as_str() {
        "real"    => Field::Real,
        "integer" => Field::Integer,
        "pattern" => Field::Pattern,
        "complex" => Field::Complex,
        other     => return Err(MmioError::UnsupportedField(other.into())),
    };

    if field == Field::Complex {
        return Err(MmioError::UnsupportedField("complex (not yet supported)".into()));
    }

    let symmetry = match tokens[4].to_ascii_lowercase().as_str() {
        "general"         => Symmetry::General,
        "symmetric"       => Symmetry::Symmetric,
        "skew-symmetric"  => Symmetry::SkewSymmetric,
        "hermitian"       => Symmetry::Hermitian,
        other             => return Err(MmioError::UnsupportedSymmetry(other.into())),
    };

    Ok((field, symmetry))
}

/// Parse the size line: `nrows ncols nnz`
fn parse_size_line(line: &str) -> Result<(usize, usize, usize), MmioError> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 3 {
        return Err(MmioError::MalformedSizeLine(line.into()));
    }
    let nrows = parts[0].parse::<usize>().map_err(|_| MmioError::MalformedSizeLine(line.into()))?;
    let ncols = parts[1].parse::<usize>().map_err(|_| MmioError::MalformedSizeLine(line.into()))?;
    let nnz   = parts[2].parse::<usize>().map_err(|_| MmioError::MalformedSizeLine(line.into()))?;
    Ok((nrows, ncols, nnz))
}

/// Parse a data line: `row col [value]` (1-indexed → 0-indexed).
fn parse_data_line(line: &str, field: Field) -> Result<(usize, usize, f64), MmioError> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(MmioError::MalformedDataLine(line.into()));
    }
    let row1 = parts[0].parse::<usize>().map_err(|_| MmioError::MalformedDataLine(line.into()))?;
    let col1 = parts[1].parse::<usize>().map_err(|_| MmioError::MalformedDataLine(line.into()))?;

    if row1 == 0 || col1 == 0 {
        return Err(MmioError::MalformedDataLine(format!("indices must be >= 1: {line}")));
    }

    let val = match field {
        Field::Pattern => 1.0,
        _ => {
            if parts.len() < 3 {
                return Err(MmioError::MalformedDataLine(line.into()));
            }
            parts[2].parse::<f64>().map_err(|_| MmioError::MalformedDataLine(line.into()))?
        }
    };

    Ok((row1 - 1, col1 - 1, val))
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn general_3x3() {
        let mtx = "\
%%MatrixMarket matrix coordinate real general
% comment
3 3 4
1 1 1.0
1 3 2.0
2 2 3.0
3 1 4.0
";
        let a = read_matrix_market_str(mtx).unwrap();
        assert_eq!(a.nrows(), 3);
        assert_eq!(a.ncols(), 3);
        assert_eq!(a.nnz(), 4);
    }

    #[test]
    fn symmetric_3x3_expands() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
3 3 3
1 1 4.0
2 1 -1.0
3 3 5.0
";
        let a = read_matrix_market_str(mtx).unwrap();
        // diagonal: (0,0)=4, (2,2)=5
        // off-diagonal: (1,0)=-1 AND (0,1)=-1 (symmetric expansion)
        assert_eq!(a.nrows(), 3);
        assert_eq!(a.nnz(), 4); // 2 diag + 2 off-diag
    }

    #[test]
    fn pattern_general() {
        let mtx = "\
%%MatrixMarket matrix coordinate pattern general
4 4 3
1 1
2 3
4 2
";
        let a = read_matrix_market_str(mtx).unwrap();
        assert_eq!(a.nrows(), 4);
        assert_eq!(a.nnz(), 3);
        // All structural entries should have value 1.0.
        for &v in a.values() {
            assert!((v - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn integer_general() {
        let mtx = "\
%%MatrixMarket matrix coordinate integer general
2 2 2
1 1 5
2 2 7
";
        let a = read_matrix_market_str(mtx).unwrap();
        assert_eq!(a.nnz(), 2);
    }

    #[test]
    fn skew_symmetric_expands_negated() {
        let mtx = "\
%%MatrixMarket matrix coordinate real skew-symmetric
3 3 2
2 1 3.0
3 2 -2.0
";
        let a = read_matrix_market_str(mtx).unwrap();
        // (1,0)=3 and (0,1)=-3; (2,1)=-2 and (1,2)=2
        assert_eq!(a.nnz(), 4);
    }

    #[test]
    fn comment_lines_ignored() {
        let mtx = "\
%%MatrixMarket matrix coordinate real general
% This is a comment
% Another comment
2 2 1
% inline comment is invalid data but we skip % lines
1 2 9.0
";
        let a = read_matrix_market_str(mtx).unwrap();
        assert_eq!(a.nnz(), 1);
    }

    #[test]
    fn missing_header_error() {
        let mtx = "2 2 1\n1 1 1.0\n";
        assert!(matches!(read_matrix_market_str(mtx), Err(MmioError::MissingHeader)));
    }

    #[test]
    fn not_coordinate_error() {
        let mtx = "%%MatrixMarket matrix array real general\n2 2\n1.0\n2.0\n3.0\n4.0\n";
        assert!(matches!(
            read_matrix_market_str(mtx),
            Err(MmioError::NotCoordinate(_))
        ));
    }

    #[test]
    fn index_out_of_bounds_error() {
        let mtx = "\
%%MatrixMarket matrix coordinate real general
2 2 1
5 1 1.0
";
        assert!(matches!(
            read_matrix_market_str(mtx),
            Err(MmioError::IndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn non_square_symmetric_error() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
3 4 1
1 1 1.0
";
        assert!(matches!(
            read_matrix_market_str(mtx),
            Err(MmioError::NonSquareSymmetric)
        ));
    }

    #[test]
    fn malformed_size_line_error() {
        let mtx = "\
%%MatrixMarket matrix coordinate real general
3 3
1 1 1.0
";
        assert!(matches!(
            read_matrix_market_str(mtx),
            Err(MmioError::MalformedSizeLine(_))
        ));
    }

    #[test]
    fn zero_index_error() {
        let mtx = "\
%%MatrixMarket matrix coordinate real general
3 3 1
0 1 1.0
";
        assert!(matches!(
            read_matrix_market_str(mtx),
            Err(MmioError::MalformedDataLine(_))
        ));
    }

    #[test]
    fn empty_matrix() {
        let mtx = "\
%%MatrixMarket matrix coordinate real general
0 0 0
";
        let a = read_matrix_market_str(mtx).unwrap();
        assert_eq!(a.nrows(), 0);
        assert_eq!(a.nnz(), 0);
    }

    #[test]
    fn pattern_symmetric_all_ones() {
        let mtx = "\
%%MatrixMarket matrix coordinate pattern symmetric
4 4 4
1 1
2 2
3 2
4 4
";
        let a = read_matrix_market_str(mtx).unwrap();
        // diagonal entries: 3 (not expanded since row==col)
        // off-diagonal: (2,1) and (1,2) → 2 entries
        assert_eq!(a.nnz(), 5);
    }

    #[test]
    fn coo_variant_returns_coo() {
        let mtx = "\
%%MatrixMarket matrix coordinate real general
3 3 2
1 1 1.0
2 2 2.0
";
        let coo = read_matrix_market_coo_str(mtx).unwrap();
        assert_eq!(coo.nrows(), 3);
        assert_eq!(coo.nnz(), 2);
    }
}
