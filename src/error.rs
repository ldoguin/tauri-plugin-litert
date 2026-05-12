use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("model already loaded: {0}")]
    ModelAlreadyLoaded(String),

    #[error("inference failed: {0}")]
    InferenceFailed(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("backend error: {0}")]
    Backend(String),

    #[error(transparent)]
    Tauri(#[from] tauri::Error),

    #[error(transparent)]
    Serde(#[from] serde_json::Error),
}

// Tauri requires command errors to be serialisable so they can be sent to the
// frontend as JSON.
impl Serialize for Error {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.to_string().as_str())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
