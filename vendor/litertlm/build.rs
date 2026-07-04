fn main() {
    // DEP_ variable names are derived from the `links` key in Cargo.toml,
    // not the package name.
    //   litert-lm-sys  links = "LiteRtLm"  → DEP_LITERTLM_LIB_DIR
    //   litert-sys     links = "LiteRt"    → DEP_LITERT_LIB_DIR

    // libLiteRtLmC.{so,dylib,dll} — from litert-lm-sys
    println!("cargo:rerun-if-env-changed=DEP_LITERTLM_LIB_DIR");
    if let Ok(dir) = std::env::var("DEP_LITERTLM_LIB_DIR") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        // Re-export for the app's build.rs to forward to the final binary.
        println!("cargo:lib_dir={dir}");
    }

    // libLiteRt.{so,dylib} + accelerator plugins — from litert-sys
    println!("cargo:rerun-if-env-changed=DEP_LITERT_LIB_DIR");
    if let Ok(dir) = std::env::var("DEP_LITERT_LIB_DIR") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        // Re-export for the app's build.rs to forward to the final binary.
        println!("cargo:litert_lib_dir={dir}");
    }
}
