#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    improper_ctypes,
    clippy::useless_transmute,
    clippy::transmute_int_to_bool,
    clippy::missing_safety_doc
)]
#![doc = "Raw FFI bindings to LiteRT-LM C API. Use the safe `litert-lm` crate for idiomatic Rust."]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// On Windows there is no prebuilt LiteRtLmC.dll, so the build script emits an
// empty stub archive to satisfy `-lLiteRtLmC`. The extern "C" declarations in
// the bindings still need concrete symbol definitions at link time, which we
// provide here. Every stub panics — the plugin guards all LLM calls behind a
// runtime capability check and falls back to the WASM backend on Windows.
#[cfg(target_os = "windows")]
mod windows_stubs {
    use super::*;
    use std::os::raw::{c_char, c_int, c_void};

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_session_config_create() -> *mut LiteRtLmSessionConfig {
        std::ptr::null_mut()
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_session_config_set_max_output_tokens(
        _config: *mut LiteRtLmSessionConfig,
        _max_output_tokens: c_int,
    ) {
        // LiteRtLmC not available on Windows — no-op
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_session_config_set_sampler_params(
        _config: *mut LiteRtLmSessionConfig,
        _sampler_params: *const LiteRtLmSamplerParams,
    ) {
        // LiteRtLmC not available on Windows — no-op
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_session_config_delete(
        _config: *mut LiteRtLmSessionConfig,
    ) {
        // LiteRtLmC not available on Windows — no-op
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_conversation_config_create(
        _engine: *mut LiteRtLmEngine,
        _session_config: *const LiteRtLmSessionConfig,
        _system_message_json: *const c_char,
        _tools_json: *const c_char,
        _messages_json: *const c_char,
        _enable_constrained_decoding: bool,
    ) -> *mut LiteRtLmConversationConfig {
        std::ptr::null_mut()
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_conversation_config_delete(
        _config: *mut LiteRtLmConversationConfig,
    ) {
        // LiteRtLmC not available on Windows — no-op
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_set_min_log_level(_level: c_int) {
        // LiteRtLmC not available on Windows — no-op
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_engine_settings_create(
        _model_path: *const c_char,
        _backend_str: *const c_char,
        _vision_backend_str: *const c_char,
        _audio_backend_str: *const c_char,
    ) -> *mut LiteRtLmEngineSettings {
        std::ptr::null_mut()
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_engine_settings_delete(
        _settings: *mut LiteRtLmEngineSettings,
    ) {
        // LiteRtLmC not available on Windows — no-op
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_engine_settings_set_max_num_tokens(
        _settings: *mut LiteRtLmEngineSettings,
        _max_num_tokens: c_int,
    ) {
        // LiteRtLmC not available on Windows — no-op
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_engine_settings_set_parallel_file_section_loading(
        _settings: *mut LiteRtLmEngineSettings,
        _parallel_file_section_loading: bool,
    ) {
        // LiteRtLmC not available on Windows — no-op
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_engine_settings_set_cache_dir(
        _settings: *mut LiteRtLmEngineSettings,
        _cache_dir: *const c_char,
    ) {
        // LiteRtLmC not available on Windows — no-op
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_engine_settings_set_activation_data_type(
        _settings: *mut LiteRtLmEngineSettings,
        _activation_data_type_int: c_int,
    ) {
        // LiteRtLmC not available on Windows — no-op
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_engine_settings_set_prefill_chunk_size(
        _settings: *mut LiteRtLmEngineSettings,
        _prefill_chunk_size: c_int,
    ) {
        // LiteRtLmC not available on Windows — no-op
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_engine_settings_enable_benchmark(
        _settings: *mut LiteRtLmEngineSettings,
    ) {
        // LiteRtLmC not available on Windows — no-op
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_engine_settings_set_num_prefill_tokens(
        _settings: *mut LiteRtLmEngineSettings,
        _num_prefill_tokens: c_int,
    ) {
        // LiteRtLmC not available on Windows — no-op
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_engine_settings_set_num_decode_tokens(
        _settings: *mut LiteRtLmEngineSettings,
        _num_decode_tokens: c_int,
    ) {
        // LiteRtLmC not available on Windows — no-op
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_engine_create(
        _settings: *const LiteRtLmEngineSettings,
    ) -> *mut LiteRtLmEngine {
        std::ptr::null_mut()
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_engine_delete(_engine: *mut LiteRtLmEngine) {
        // LiteRtLmC not available on Windows — no-op
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_engine_create_session(
        _engine: *mut LiteRtLmEngine,
        _config: *mut LiteRtLmSessionConfig,
    ) -> *mut LiteRtLmSession {
        std::ptr::null_mut()
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_session_delete(_session: *mut LiteRtLmSession) {
        // LiteRtLmC not available on Windows — no-op
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_session_generate_content(
        _session: *mut LiteRtLmSession,
        _inputs: *const InputData,
        _num_inputs: usize,
    ) -> *mut LiteRtLmResponses {
        std::ptr::null_mut()
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_responses_delete(_responses: *mut LiteRtLmResponses) {
        // LiteRtLmC not available on Windows — no-op
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_responses_get_num_candidates(
        _responses: *const LiteRtLmResponses,
    ) -> c_int {
        0
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_responses_get_response_text_at(
        _responses: *const LiteRtLmResponses,
        _index: c_int,
    ) -> *const c_char {
        std::ptr::null_mut()
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_session_get_benchmark_info(
        _session: *mut LiteRtLmSession,
    ) -> *mut LiteRtLmBenchmarkInfo {
        std::ptr::null_mut()
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_benchmark_info_delete(
        _benchmark_info: *mut LiteRtLmBenchmarkInfo,
    ) {
        // LiteRtLmC not available on Windows — no-op
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_benchmark_info_get_time_to_first_token(
        _benchmark_info: *const LiteRtLmBenchmarkInfo,
    ) -> f64 {
        0.0
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_benchmark_info_get_total_init_time_in_second(
        _benchmark_info: *const LiteRtLmBenchmarkInfo,
    ) -> f64 {
        0.0
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_benchmark_info_get_num_prefill_turns(
        _benchmark_info: *const LiteRtLmBenchmarkInfo,
    ) -> c_int {
        0
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_benchmark_info_get_num_decode_turns(
        _benchmark_info: *const LiteRtLmBenchmarkInfo,
    ) -> c_int {
        0
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_benchmark_info_get_prefill_token_count_at(
        _benchmark_info: *const LiteRtLmBenchmarkInfo,
        _index: c_int,
    ) -> c_int {
        0
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_benchmark_info_get_decode_token_count_at(
        _benchmark_info: *const LiteRtLmBenchmarkInfo,
        _index: c_int,
    ) -> c_int {
        0
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(
        _benchmark_info: *const LiteRtLmBenchmarkInfo,
        _index: c_int,
    ) -> f64 {
        0.0
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_benchmark_info_get_decode_tokens_per_sec_at(
        _benchmark_info: *const LiteRtLmBenchmarkInfo,
        _index: c_int,
    ) -> f64 {
        0.0
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_session_generate_content_stream(
        _session: *mut LiteRtLmSession,
        _inputs: *const InputData,
        _num_inputs: usize,
        _callback: LiteRtLmStreamCallback,
        _callback_data: *mut c_void,
    ) -> c_int {
        0
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_conversation_create(
        _engine: *mut LiteRtLmEngine,
        _config: *mut LiteRtLmConversationConfig,
    ) -> *mut LiteRtLmConversation {
        std::ptr::null_mut()
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_conversation_delete(
        _conversation: *mut LiteRtLmConversation,
    ) {
        // LiteRtLmC not available on Windows — no-op
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_conversation_send_message(
        _conversation: *mut LiteRtLmConversation,
        _message_json: *const c_char,
        _extra_context: *const c_char,
    ) -> *mut LiteRtLmJsonResponse {
        std::ptr::null_mut()
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_json_response_delete(
        _response: *mut LiteRtLmJsonResponse,
    ) {
        // LiteRtLmC not available on Windows — no-op
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_json_response_get_string(
        _response: *const LiteRtLmJsonResponse,
    ) -> *const c_char {
        std::ptr::null_mut()
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_conversation_send_message_stream(
        _conversation: *mut LiteRtLmConversation,
        _message_json: *const c_char,
        _extra_context: *const c_char,
        _callback: LiteRtLmStreamCallback,
        _callback_data: *mut c_void,
    ) -> c_int {
        0
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_conversation_cancel_process(
        _conversation: *mut LiteRtLmConversation,
    ) {
        // LiteRtLmC not available on Windows — no-op
    }

    #[no_mangle]
    pub unsafe extern "C" fn litert_lm_conversation_get_benchmark_info(
        _conversation: *mut LiteRtLmConversation,
    ) -> *mut LiteRtLmBenchmarkInfo {
        std::ptr::null_mut()
    }
}
