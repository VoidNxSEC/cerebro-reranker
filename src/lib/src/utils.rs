use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Error, Debug)]
pub enum CerebroError {
    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("IPFS error: {0}")]
    Ipfs(String),

    #[error("cache error: {0}")]
    Cache(String),

    #[error("scoring error: {0}")]
    Scoring(String),

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("config error: {0}")]
    Config(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Reqwest(#[from] reqwest::Error),

    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, CerebroError>;

// ---------------------------------------------------------------------------
// Model registry (models.toml)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub size_mb: u32,
    pub latency_ms: u32,
    pub accuracy: f64,
    pub ipfs_cid: String,
    #[serde(default)]
    pub trained_on: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ModelRegistry {
    models: HashMap<String, ModelConfig>,
}

/// Load model registry from a TOML file.
///
/// Expected format:
/// ```toml
/// [models.minilm]
/// name = "ms-marco-MiniLM-L-6-v2"
/// size_mb = 80
/// ...
/// ```
pub fn load_model_registry(path: &Path) -> Result<HashMap<String, ModelConfig>> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        CerebroError::Config(format!("failed to read {}: {e}", path.display()))
    })?;

    let registry: ModelRegistry = toml::from_str(&content).map_err(|e| {
        CerebroError::Config(format!("failed to parse {}: {e}", path.display()))
    })?;

    Ok(registry.models)
}

/// Initialize tracing subscriber (call once at startup).
pub fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("cerebro_scorer=info"));

    fmt()
        .with_env_filter(filter)
        .with_target(true)
        .compact()
        .try_init()
        .ok(); // ignore if already initialized
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_load_model_registry() {
        let toml_content = r#"
[models.minilm]
name = "ms-marco-MiniLM-L-6-v2"
size_mb = 80
latency_ms = 15
accuracy = 0.89
ipfs_cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"

[models.custom]
name = "cerebro-security-reranker"
size_mb = 500
latency_ms = 60
accuracy = 0.97
ipfs_cid = "TBD"
trained_on = "cerebro-knowledge-base"
"#;

        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(toml_content.as_bytes()).unwrap();

        let models = load_model_registry(tmp.path()).unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models["minilm"].size_mb, 80);
        assert_eq!(
            models["custom"].trained_on.as_deref(),
            Some("cerebro-knowledge-base")
        );
    }

    #[test]
    fn test_load_model_registry_missing_file() {
        let result = load_model_registry(Path::new("/nonexistent/models.toml"));
        assert!(result.is_err());
    }
}
