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
        .ios_path("ios")
        .build();

    // iOS: the plugin's Swift package depends on LiteRT-LM, whose C core ships
    // as the dynamic CLiteRTLM.framework (SwiftPM copies the correct slice into
    // swift-rs's products dir thanks to the patched --triple). The downstream
    // cdylib link needs the framework search path + link. The .app bundle gets
    // the framework embedded via the Xcode project (gen/apple/project.yml).
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("ios") {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let target = std::env::var("TARGET").unwrap();
        let simulator = target.ends_with("ios-sim")
            || (target.starts_with("x86_64") && target.ends_with("ios"));
        let arch = match std::env::var("CARGO_CFG_TARGET_ARCH").unwrap().as_str() {
            "aarch64" => "arm64".to_string(),
            other => other.to_string(),
        };
        let products_dir = format!(
            "{arch}-apple-ios{}",
            if simulator { "-simulator" } else { "" }
        );
        let config = if std::env::var("DEBUG").as_deref() == Ok("true") {
            "debug"
        } else {
            "release"
        };
        println!(
            "cargo:rustc-link-search=framework={out_dir}/swift-rs/tauri-plugin-litert/{products_dir}/{config}"
        );
        println!("cargo:rustc-link-lib=framework=CLiteRTLM");
    }

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        if let Ok(dir) = std::env::var("DEP_LITERT_LIB_DIR") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        }
        if let Ok(dir) = std::env::var("DEP_LITERTLM_LIB_DIR") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        }
    }
}
