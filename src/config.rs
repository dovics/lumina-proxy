use serde::Deserialize;
use std::fs;
use anyhow::Result;

/// Rotation strategy for log files
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RotationStrategy {
    /// Rotate daily
    Daily,
    /// Rotate when file reaches max size
    Size,
    /// Never rotate
    Never,
}

/// Provider type for LLM backend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum ProviderType {
    /// OpenAI API
    #[serde(rename = "openai")]
    OpenAi,
    /// Ollama local API
    #[serde(rename = "ollama")]
    Ollama,
    /// Anthropic API
    #[serde(rename = "anthropic")]
    Anthropic,
    /// Google Gemini API
    #[serde(rename = "gemini")]
    Gemini,
    /// OpenAI-compatible API (e.g., vLLM, llama-cpp-python, etc.)
    #[serde(rename = "openai-compatible")]
    OpenAiCompatible,
}

/// File logging configuration
#[derive(Debug, Clone, Deserialize)]
pub struct FileLoggingConfig {
    /// Whether file logging is enabled
    pub enabled: bool,
    /// Path to the log file
    pub path: Option<String>,
    /// Rotation strategy
    pub rotation: Option<RotationStrategy>,
    /// Maximum size in megabytes before rotation (only for Size strategy)
    pub max_size_mb: Option<u64>,
    /// Maximum number of rotated files to keep
    pub max_files: Option<u32>,
}

/// Logging configuration
#[derive(Debug, Clone, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    /// Whether to enable console logging (default: false)
    #[serde(default)]
    pub console: bool,
    /// File logging configuration
    pub file: Option<FileLoggingConfig>,
}

/// Server configuration
#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    /// Port to listen on
    pub port: u16,
    /// Host address to bind to
    pub host: String,
    /// Authentication token required for requests (optional)
    pub auth_token: Option<String>,
}

/// Statistics configuration
#[derive(Debug, Clone, Deserialize)]
pub struct StatisticsConfig {
    /// Whether statistics collection is enabled
    pub enabled: bool,
    /// Path to the statistics file (JSONL format)
    pub file_path: Option<String>,
    /// Buffer duration in seconds before writing to disk
    pub buffer_seconds: Option<f64>,
}

/// Route configuration for a model to backend provider
#[derive(Debug, Clone, Deserialize)]
pub struct RouteConfig {
    /// Model name pattern (exact match for now)
    pub model_name: String,
    /// Optional override: actual model name to send to upstream provider
    /// If empty or not present, uses `model_name`
    #[serde(default)]
    pub upstream_model: Option<String>,
    /// Type of provider
    pub provider_type: ProviderType,
    /// Base URL for the API (alternative to url)
    pub base_url: Option<String>,
    /// Full URL for the API (alternative to base_url)
    pub url: Option<String>,
    /// API key for authentication
    pub api_key: String,
    /// Whether this route is enabled
    pub enabled: bool,
}

impl RouteConfig {
    /// Get the effective upstream model name - returns upstream_model if set and non-empty, otherwise model_name
    pub fn upstream_model(&self) -> &str {
        self.upstream_model
            .as_ref()
            .filter(|s| !s.is_empty())
            .unwrap_or(&self.model_name)
    }
}

/// Main configuration structure
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Server configuration
    pub server: ServerConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
    /// Statistics configuration
    pub statistics: StatisticsConfig,
    /// List of routes mapping models to backends
    pub routes: Vec<RouteConfig>,
}

impl Config {
    /// Load configuration from a YAML file
    pub fn load_from_file(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let config = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Find the backend route for a given model name
    pub fn find_backend_for_model(&self, model: &str) -> Option<&RouteConfig> {
        self.routes
            .iter()
            .find(|route| route.enabled && route.model_name == model)
    }
}
