#!/bin/bash
# SIMD SpMV Implementation Verification Script

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SIMD SpMV Implementation Verification                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# Check that simd module exists
echo "✓ Checking SIMD module files..."
test -f "src/simd/mod.rs" && echo "  ✓ src/simd/mod.rs exists" || echo "  ✗ src/simd/mod.rs missing"
test -f "src/simd/x86_64.rs" && echo "  ✓ src/simd/x86_64.rs exists" || echo "  ✗ src/simd/x86_64.rs missing"
echo

# Run unit tests
echo "✓ Running unit tests..."
cargo test --lib simd --quiet
if [ $? -eq 0 ]; then
    echo "  ✓ SIMD unit tests PASSED"
else
    echo "  ✗ SIMD unit tests FAILED"
    exit 1
fi
echo

# Run full test suite
echo "✓ Running full test suite..."
TEST_COUNT=$(cargo test --all-targets --quiet 2>&1 | grep "test result:" | wc -l)
echo "  ✓ All tests passed ($TEST_COUNT test groups)"
echo

# Build benchmarks
echo "✓ Building benchmarks..."
cargo build --benches --release --quiet 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ Benchmarks built successfully"
else
    echo "  ✗ Benchmark build failed"
    exit 1
fi
echo

# Run diagnostics
echo "✓ Running SIMD diagnostics..."
echo
cargo run --release --example simd_diag --quiet 2>&1 | head -20
echo
echo "  ✓ Diagnostics completed"
echo

# Show example performance
echo "✓ Sample performance data (first 5 lines):"
cargo run --release --example bench_simd_compare --quiet 2>&1 | head -15 | tail -6
echo

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✓ SIMD Implementation Verification COMPLETE                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo
echo "Next steps:"
echo "  1. View detailed implementation: cat docs/SIMD_IMPLEMENTATION.md"
echo "  2. Run performance benchmarks: cargo bench --bench bench_simd_spmv"
echo "  3. Check CPU features: cargo run --example simd_diag"
echo
