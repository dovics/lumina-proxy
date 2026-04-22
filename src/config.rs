use serde::Deserialize;
use std::fs;
use anyhow::Result;
use chrono::Utc;
use std::collections::HashSet;

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
#[derive(Debug, Clone, PartialEq, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ServerConfig {
    /// Port to listen on
    pub port: u16,
    /// Host address to bind to
    pub host: String,
    /// Authentication token required for requests (optional)
    pub auth_token: Option<String>,
}

/// Statistics configuration
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct StatisticsConfig {
    /// Whether statistics collection is enabled
    pub enabled: bool,
    /// Path to the statistics file (JSONL format)
    pub file_path: Option<String>,
    /// Buffer duration in seconds before writing to disk
    pub buffer_seconds: Option<f64>,
}

/// Route configuration for a model to backend provider
#[derive(Debug, Clone, PartialEq, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    /// Server configuration
    pub server: ServerConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
    /// Statistics configuration
    pub statistics: StatisticsConfig,
    /// List of routes mapping models to backends
    pub routes: Vec<RouteConfig>,
    /// Configuration version (incremented on each reload)
    #[serde(skip)]
    pub version: u64,
    /// Timestamp when this configuration was loaded
    #[serde(skip)]
    pub loaded_at: chrono::DateTime<chrono::Utc>,
}

impl Config {
    /// Load configuration from a YAML file
    pub fn load_from_file(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let mut config: Self = serde_yaml::from_str(&content)?;
        config.version = 1;
        config.loaded_at = Utc::now();
        Ok(config)
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate server port (u16 is already 0-65535)
        if self.server.port == 0 {
            return Err("Server port cannot be 0. Must be between 1 and 65535".to_string());
        }

        // Validate server host
        if self.server.host.is_empty() {
            return Err("Server host cannot be empty".to_string());
        }

        // Validate routes is not empty
        if self.routes.is_empty() {
            return Err("Routes list cannot be empty".to_string());
        }

        // Check for duplicate model names and validate each route
        let mut seen_models = HashSet::new();
        for route in &self.routes {
            if route.model_name.is_empty() {
                return Err("Route model_name cannot be empty".to_string());
            }
            if !seen_models.insert(&route.model_name) {
                return Err(format!("Duplicate model name: {}", route.model_name));
            }
            if route.api_key.is_empty() {
                return Err(format!("API key for model '{}' cannot be empty", route.model_name));
            }
            if let Some(upstream) = &route.upstream_model {
                if upstream.is_empty() {
                    return Err(format!("Upstream model for '{}' cannot be empty (remove the field if not used)", route.model_name));
                }
            }
        }

        // Validate logging level
        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_levels.contains(&self.logging.level.to_lowercase().as_str()) {
            return Err(format!(
                "Invalid log level: {}. Valid levels are: {}",
                self.logging.level,
                valid_levels.join(", ")
            ));
        }

        Ok(())
    }

    /// Load configuration from a YAML file and validate it
    pub fn load_and_validate(path: &str) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file: {}", e))?;

        let mut config: Self = serde_yaml::from_str(&content)
            .map_err(|e| format!("Failed to parse config file: {}", e))?;

        config.validate()?;
        config.version = 1;
        config.loaded_at = Utc::now();
        Ok(config)
    }

    /// Find the backend route for a given model name
    pub fn find_backend_for_model(&self, model: &str) -> Option<&RouteConfig> {
        self.routes
            .iter()
            .find(|route| route.enabled && route.model_name == model)
    }
}
