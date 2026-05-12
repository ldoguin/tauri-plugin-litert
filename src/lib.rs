use tauri::{
    plugin::{Builder, TauriPlugin},
    Manager, Runtime,
};

pub use lm_models::*;
pub use models::*;

mod commands;
mod error;
mod lm_commands;
mod lm_models;
mod models;

// Platform-specific implementations
#[cfg(desktop)]
mod desktop;
#[cfg(desktop)]
mod desktop_lm;
#[cfg(mobile)]
mod mobile;
#[cfg(mobile)]
mod mobile_lm;

pub use error::{Error, Result};

// ---------------------------------------------------------------------------
// LiteRT (inference / embedding) extension trait
// ---------------------------------------------------------------------------

pub trait LiteRtExt<R: Runtime> {
    fn litert(&self) -> &LiteRt<R>;
}

impl<R: Runtime, T: Manager<R>> LiteRtExt<R> for T {
    fn litert(&self) -> &LiteRt<R> {
        self.state::<LiteRt<R>>().inner()
    }
}

#[cfg(desktop)]
pub use desktop::LiteRt;
#[cfg(mobile)]
pub use mobile::LiteRt;

// ---------------------------------------------------------------------------
// LiteRT-LM (LLM generation) extension trait
// ---------------------------------------------------------------------------

pub trait LiteRtLmExt<R: Runtime> {
    fn litert_lm(&self) -> &LiteRtLm<R>;
}

impl<R: Runtime, T: Manager<R>> LiteRtLmExt<R> for T {
    fn litert_lm(&self) -> &LiteRtLm<R> {
        self.state::<LiteRtLm<R>>().inner()
    }
}

#[cfg(desktop)]
pub use desktop_lm::LiteRtLm;
#[cfg(mobile)]
pub use mobile_lm::LiteRtLm;

// ---------------------------------------------------------------------------
// Plugin init
// ---------------------------------------------------------------------------

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("litert")
        .invoke_handler(tauri::generate_handler![
            // Inference / embedding
            commands::load_model,
            commands::unload_model,
            commands::list_models,
            commands::get_model_info,
            commands::run_inference,
            commands::create_embedding,
            // LLM generation
            lm_commands::load_lm_model,
            lm_commands::unload_lm_model,
            lm_commands::list_lm_models,
            lm_commands::generate,
            lm_commands::generate_stream,
        ])
        .setup(|app, api| {
            #[cfg(mobile)]
            {
                let litert = mobile::init(app, &api)?;
                app.manage(litert);
                let litert_lm = mobile_lm::init(app, &api)?;
                app.manage(litert_lm);
            }
            #[cfg(desktop)]
            {
                let _ = api;
                app.manage(desktop::LiteRt::new(app.clone()));
                app.manage(desktop_lm::LiteRtLm::new(app.clone()));
            }
            Ok(())
        })
        .build()
}
