const COMMANDS: &[&str] = &[
    // Inference / embedding
    "load_model",
    "unload_model",
    "list_models",
    "run_inference",
    "create_embedding",
    "get_model_info",
    // LLM generation
    "load_lm_model",
    "unload_lm_model",
    "list_lm_models",
    "generate",
    "generate_stream",
];

fn main() {
    tauri_plugin::Builder::new(COMMANDS)
        .android_path("android")
        .build();

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        if let Ok(dir) = std::env::var("DEP_LITERT_LIB_DIR") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        }
        if let Ok(dir) = std::env::var("DEP_LITERTLM_LIB_DIR") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        }
    }
}
