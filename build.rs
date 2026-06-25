fn main() {
    // Feature blas-openblas-system: link against pre-installed OpenBLAS
    // using OPENBLAS_LIB_DIR and OPENBLAS_INCLUDE_DIR environment variables.
    #[cfg(feature = "blas-openblas-system")]
    {
        let lib_dir = std::env::var("OPENBLAS_LIB_DIR")
            .expect("OPENBLAS_LIB_DIR must be set for blas-openblas-system feature");
        let inc_dir = std::env::var("OPENBLAS_INCLUDE_DIR")
            .expect("OPENBLAS_INCLUDE_DIR must be set for blas-openblas-system feature");

        println!("cargo:rustc-link-search=native={}", lib_dir);
        println!("cargo:rustc-link-lib=static=libopenblas");
        println!("cargo:include={}", inc_dir);
    }

    // Regular blas-openblas uses blas-src which handles linking internally.
}
