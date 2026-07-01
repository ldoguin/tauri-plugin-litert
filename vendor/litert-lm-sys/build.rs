//! Build script for litert-lm-sys.
//!
//! Downloads the pinned `libLiteRtLmC.{so,dylib}` shared library from our
//! mirrored GitHub release, SHA-256-verifies it, caches it, and emits the
//! linker directives. Same pattern as litert-sys.
//!
//! Escape hatches:
//!   `LITERT_LM_LIB_DIR`     — directory containing the shared lib; skip download.
//!   `LITERT_NO_DOWNLOAD`    — fail hard if the cache is empty.
//!   `LITERT_CACHE_DIR`      — override the cache root.

use std::{
    env, fs,
    io::Read,
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};

const LITERT_LM_VERSION: &str = "0.13.1";

#[cfg(feature = "generate-bindings")]
const LITERT_LM_HEADERS_VERSION: &str = "0.13.1";

struct Prebuilt {
    /// Full URL to download. For `.whl` sources this is the wheel URL.
    url: &'static str,
    /// If Some, the file is a zip/wheel and this is the entry to extract.
    zip_entry: Option<&'static str>,
    local_name: &'static str,
    sha256: &'static str,
    size: u64,
}

fn prebuilt_for(target: &str) -> Option<Prebuilt> {
    Some(match target {
        "x86_64-unknown-linux-gnu" => Prebuilt {
            // v0.13.1 ships inside the litert-lm-api Python wheel for Linux x86_64.
            url: "https://files.pythonhosted.org/packages/36/27/bb0c2e084d59938bc12f960db50e7e8e056337ed286f31468aaa39bc9d9a/litert_lm_api-0.13.1-py3-none-manylinux_2_27_x86_64.whl",
            zip_entry: Some("litert_lm/liblitert-lm.so"),
            local_name: "libLiteRtLmC.so",
            sha256: "a500feef22dc6d1c7b1abc7c59284fe3206891072d5fac6d247a9534e3e39d6b",
            size: 126_248_352,
        },
        "aarch64-unknown-linux-gnu" => Prebuilt {
            // v0.13.1 ships inside the litert-lm-api Python wheel for Linux aarch64.
            url: "https://files.pythonhosted.org/packages/12/a7/c23db30dc19ad07a188eeeab48fd8d676554e8647dac6e9dbf62be0f82f1/litert_lm_api-0.13.1-py3-none-manylinux_2_27_aarch64.whl",
            zip_entry: Some("litert_lm/liblitert-lm.so"),
            local_name: "libLiteRtLmC.so",
            sha256: "5f2bd137b6ed4fd657b4e868e3f52f76134fb383f24df038342b4abb8d64e847",
            size: 122_397_712,
        },
        "aarch64-apple-darwin" => Prebuilt {
            url: "https://github.com/offbit-ai/LiteRT/releases/download/litert-lm-v0.10.2/libLiteRtLmC.dylib",
            zip_entry: None,
            local_name: "libLiteRtLmC.dylib",
            sha256: "616c71d3f52d7b6e7847cba2a3876890aeac993e27ec147dbf1c7de4fd786456",
            size: 28_862_192,
        },
        _ => return None,
    })
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=LITERT_LM_LIB_DIR");
    println!("cargo:rerun-if-env-changed=LITERT_NO_DOWNLOAD");
    println!("cargo:rerun-if-env-changed=LITERT_CACHE_DIR");

    let target = env::var("TARGET").expect("TARGET env var missing");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR env var missing"));

    emit_bindings(&target, &out_dir);
    let lib_dir = locate_library(&target);
    emit_link_directives(&target, lib_dir.as_deref());

    // On Linux, build a symbol-interposition shim that makes
    // LiteRtGetEnvironmentOptionsValue safe against unknown tag values.
    //
    // Root cause: libLiteRtWebGpuAccelerator.so (any version >= v0.10.2) calls
    // LiteRtGetEnvironmentOptionsValue with tag 0xffff88a0 (a new enum value added
    // after v0.10.2).  The v0.10.2 implementation crashes in std::_Hashtable::find()
    // because the env options map either doesn't contain the key or is empty/unset.
    //
    // The shim delegates to the real implementation via RTLD_NEXT.  For unknown tags
    // (high-bit set, i.e. tag > 0x7fff_ffff) it returns kLiteRtStatusErrorNotFound
    // so the accelerator can degrade gracefully instead of segfaulting.
    if target.contains("linux") {
        build_litert_env_shim(&out_dir);
    }
}

fn build_litert_env_shim(out_dir: &Path) {
    let shim_src = out_dir.join("litert_env_shim.c");
    fs::write(&shim_src, r#"
/* LiteRtGetEnvironmentOptionsValue interposition shim.
 *
 * libLiteRtWebGpuAccelerator.so >= v0.10.2 calls this function with tag
 * values added after libLiteRtLmC.so v0.10.2 was released (e.g. 0xffff88a0).
 * The old implementation crashes in std::_Hashtable::find() for unknown tags.
 *
 * This shim catches those calls via RTLD_NEXT and returns
 * kLiteRtStatusErrorNotFound instead of crashing.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>

typedef void*    LiteRtEnvironment;
typedef uint32_t LiteRtEnvOptionTag;
typedef struct { uint32_t type; uint64_t data; } LiteRtAny;
typedef int32_t  LiteRtStatus;

/* kLiteRtStatusErrorNotFound from litert/c/litert_common.h */
#define kLiteRtStatusErrorNotFound 9

typedef LiteRtStatus (*GetEnvOptFn)(LiteRtEnvironment, LiteRtEnvOptionTag, LiteRtAny*);

__attribute__((visibility("default")))
LiteRtStatus LiteRtGetEnvironmentOptionsValue(
    LiteRtEnvironment env,
    LiteRtEnvOptionTag tag,
    LiteRtAny* value)
{
    static GetEnvOptFn real_fn = (GetEnvOptFn)0;
    static int initialised = 0;

    if (!initialised) {
        real_fn = (GetEnvOptFn)dlsym(RTLD_NEXT, "LiteRtGetEnvironmentOptionsValue");
        initialised = 1;
    }

    /* Tags with high bit set are from versions newer than libLiteRtLmC.so v0.10.2.
     * Return "not found" so the accelerator degrades gracefully. */
    if ((uint32_t)tag > 0x7fffffffu) {
        fprintf(stderr,
            "[litert-shim] LiteRtGetEnvironmentOptionsValue: unknown tag 0x%08x, "
            "returning kLiteRtStatusErrorNotFound\n", (unsigned)tag);
        return kLiteRtStatusErrorNotFound;
    }

    if (!real_fn) return kLiteRtStatusErrorNotFound;
    return real_fn(env, tag, value);
}
"#).expect("write litert_env_shim.c");

    let shim_lib = out_dir.join("liblitert_env_shim.a");
    let obj = out_dir.join("litert_env_shim.o");

    // Resolve the C compiler: honour Cargo's cross-compilation env vars so
    // the shim is compiled for the target, not the host.
    // Priority: CC_<TARGET_UNDERSCORED> → CC → "aarch64-linux-gnu-gcc" (for
    // that target) → "cc".
    let target = env::var("TARGET").unwrap_or_default();
    let target_cc_var = format!("CC_{}", target.replace('-', "_"));
    let cc_cmd = env::var(&target_cc_var)
        .or_else(|_| env::var("CC"))
        .unwrap_or_else(|_| {
            if target == "aarch64-unknown-linux-gnu" {
                "aarch64-linux-gnu-gcc".to_string()
            } else {
                "cc".to_string()
            }
        });
    let ar_var = format!("AR_{}", target.replace('-', "_"));
    let ar_cmd = env::var(&ar_var)
        .or_else(|_| env::var("AR"))
        .unwrap_or_else(|_| {
            if target == "aarch64-unknown-linux-gnu" {
                "aarch64-linux-gnu-ar".to_string()
            } else {
                "ar".to_string()
            }
        });

    let cc = std::process::Command::new(&cc_cmd)
        .args([
            "-O2", "-fPIC",
            "-c", shim_src.to_str().unwrap(),
            "-o", obj.to_str().unwrap(),
        ])
        .status()
        .unwrap_or_else(|e| panic!("{cc_cmd} not found: {e}"));
    assert!(cc.success(), "failed to compile litert_env_shim.c with {cc_cmd}");

    let ar = std::process::Command::new(&ar_cmd)
        .args(["rcs", shim_lib.to_str().unwrap(), obj.to_str().unwrap()])
        .status()
        .unwrap_or_else(|e| panic!("{ar_cmd} not found: {e}"));
    assert!(ar.success(), "failed to archive litert_env_shim.o with {ar_cmd}");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    // Link shim BEFORE libLiteRtLmC so our symbol takes precedence at runtime
    // when libLiteRtWebGpuAccelerator.so calls LiteRtGetEnvironmentOptionsValue.
    println!("cargo:rustc-link-lib=static:+whole-archive=litert_env_shim");
    println!("cargo:rustc-link-arg=-Wl,-ldl");
    println!("cargo:warning=litert-lm-sys: compiled LiteRtGetEnvironmentOptionsValue shim");
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

#[cfg(feature = "generate-bindings")]
fn emit_bindings(_target: &str, out_dir: &Path) {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let headers_dir = manifest_dir
        .join("third_party")
        .join(format!("litert-lm-v{LITERT_LM_HEADERS_VERSION}"));
    assert!(
        headers_dir.join("c/engine.h").exists(),
        "vendored headers not found at {}",
        headers_dir.display()
    );

    bindgen::Builder::default()
        .header(manifest_dir.join("wrapper.h").to_string_lossy())
        .clang_arg(format!("-I{}", headers_dir.display()))
        .allowlist_function("litert_lm_.*")
        .allowlist_type("LiteRtLm.*")
        .allowlist_type("Type")
        .allowlist_var("kLiteRtLm.*")
        .allowlist_var("kType.*|kTopK|kTopP|kGreedy|kTypeUnspecified")
        .allowlist_var("kInput.*")
        .prepend_enum_name(false)
        .layout_tests(false)
        .derive_default(true)
        .derive_debug(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed")
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("write bindings.rs");
}

#[cfg(not(feature = "generate-bindings"))]
fn emit_bindings(target: &str, out_dir: &Path) {
    let pregenerated = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("bindings")
        .join(format!("{target}.rs"));

    if !pregenerated.exists() {
        panic!(
            "litert-lm-sys: no pre-generated bindings for target `{target}`.\n\
             Run with `--features generate-bindings` (needs libclang) or add \
             src/bindings/{target}.rs.",
        );
    }
    fs::copy(&pregenerated, out_dir.join("bindings.rs")).expect("copy pre-generated bindings");
    println!("cargo:rerun-if-changed={}", pregenerated.display());
}

// ---------------------------------------------------------------------------
// Library resolution
// ---------------------------------------------------------------------------

fn locate_library(target: &str) -> Option<PathBuf> {
    // 1) Explicit override.
    if let Ok(dir) = env::var("LITERT_LM_LIB_DIR") {
        let dir = PathBuf::from(dir);
        assert!(
            dir.is_dir(),
            "LITERT_LM_LIB_DIR={} is not a directory",
            dir.display()
        );
        return Some(dir);
    }

    // 2) Forward litert-sys lib dir so the linker can also find libLiteRt +
    //    its accelerator/plugin dylibs (libGemmaModelConstraintProvider etc.)
    //    which libLiteRtLmC depends on at runtime.
    if let Ok(litert_dir) = env::var("DEP_LITERT_LIB_DIR") {
        println!("cargo:rustc-link-search=native={litert_dir}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{litert_dir}");
    }

    // 3) Download the mirrored shared lib.
    let pb = match prebuilt_for(target) {
        Some(p) => p,
        None => {
            println!(
                "cargo:warning=litert-lm-sys: no prebuilt for target `{target}`. \
                 Set LITERT_LM_LIB_DIR to a directory containing {lib}.",
                lib = if target.contains("windows") {
                    "LiteRtLmC.dll"
                } else {
                    "libLiteRtLmC.*"
                }
            );
            return None;
        }
    };

    let cache_dir = cache_dir_for(target);
    ensure_prebuilt(&pb, &cache_dir);
    Some(cache_dir)
}

fn cache_dir_for(target: &str) -> PathBuf {
    if let Some(dir) = env::var_os("LITERT_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(dir) = dirs::cache_dir() {
        if fs::create_dir_all(&dir).is_ok() {
            return dir
                .join("litert-lm-sys")
                .join(format!("v{LITERT_LM_VERSION}"))
                .join(target);
        }
    }
    PathBuf::from(env::var("OUT_DIR").unwrap())
        .join("litert-lm-cache")
        .join(target)
}

fn ensure_prebuilt(pb: &Prebuilt, cache_dir: &Path) {
    fs::create_dir_all(cache_dir).expect("create cache dir");

    let dest = cache_dir.join(pb.local_name);
    let marker = cache_dir.join(format!("{}.verified", pb.local_name));
    if dest.exists() && marker.exists() {
        return;
    }

    if env::var_os("LITERT_NO_DOWNLOAD").is_some() {
        panic!(
            "litert-lm-sys: LITERT_NO_DOWNLOAD set but {} missing from {}",
            pb.local_name,
            cache_dir.display()
        );
    }

    println!(
        "cargo:warning=litert-lm-sys: downloading {} ({} bytes) (first build only)",
        pb.local_name, pb.size,
    );

    // Download the source (may be a direct .so or a Python wheel/.zip).
    let mut raw = Vec::new();
    ureq::get(pb.url)
        .call()
        .unwrap_or_else(|e| panic!("GET {}: {e}", pb.url))
        .into_reader()
        .read_to_end(&mut raw)
        .unwrap_or_else(|e| panic!("read {}: {e}", pb.local_name));

    // If this is a wheel/zip, extract the specific entry; otherwise use as-is.
    let buf = if let Some(entry) = pb.zip_entry {
        extract_zip_entry(&raw, entry, pb.local_name)
    } else {
        raw
    };

    if buf.len() as u64 != pb.size {
        panic!(
            "litert-lm-sys: size mismatch for {}: expected {}, got {}",
            pb.local_name,
            pb.size,
            buf.len()
        );
    }

    let hash = hex(&Sha256::digest(&buf));
    if hash != pb.sha256 {
        panic!(
            "litert-lm-sys: SHA-256 mismatch for {}: expected {}, got {hash}",
            pb.local_name, pb.sha256
        );
    }

    fs::write(&dest, &buf).unwrap_or_else(|e| panic!("write {}: {e}", dest.display()));

    // The Bazel-built dylib ships with install_name=bazel-out/.../libLiteRtLmC.*
    // which is a relative path the macOS loader can't resolve. Rewrite to
    // @rpath/ so our -rpath emission works. This is safe because @rpath/ is
    // shorter than the Bazel path, so install_name_tool fits in the existing
    // header padding.
    if dest.extension().is_some_and(|e| e == "dylib") {
        let _ = std::process::Command::new("install_name_tool")
            .args(["-id", &format!("@rpath/{}", pb.local_name)])
            .arg(&dest)
            .status();
    }

    fs::write(&marker, pb.sha256).expect("write verified marker");
}

/// Extract a single named entry from a zip/wheel archive held in memory.
/// Uses the `zip` crate which handles ZIP64, Deflate, Stored, and other
/// compression methods — including Python wheels that use ZIP64 by default.
fn extract_zip_entry(zip_bytes: &[u8], entry_name: &str, label: &str) -> Vec<u8> {
    use std::io::{Cursor, Read as _};
    let cursor = Cursor::new(zip_bytes);
    let mut archive = zip::ZipArchive::new(cursor)
        .unwrap_or_else(|e| panic!("litert-lm-sys: failed to open {label} as zip: {e}"));
    let mut entry = archive.by_name(entry_name)
        .unwrap_or_else(|e| panic!("litert-lm-sys: entry '{entry_name}' not found in {label}: {e}"));
    let mut buf = Vec::with_capacity(entry.size() as usize);
    entry.read_to_end(&mut buf)
        .unwrap_or_else(|e| panic!("litert-lm-sys: failed to read '{entry_name}' from {label}: {e}"));
    buf
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        write!(&mut s, "{b:02x}").unwrap();
    }
    s
}

// ---------------------------------------------------------------------------
// Linker directives
// ---------------------------------------------------------------------------

fn emit_link_directives(target: &str, lib_dir: Option<&Path>) {
    if let Some(dir) = lib_dir {
        let dir = dir.display();
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        println!("cargo:lib_dir={dir}");
    }

    println!("cargo:rustc-link-lib=dylib=LiteRtLmC");

    if target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
    if target.contains("android") {
        println!("cargo:rustc-link-lib=dylib=log");
    }
}
