//! Logging initialization for the proxy
//!
//! This module handles initialization of the tracing subscriber
//! with configurable console and file output, and log rotation.

use crate::config::{LoggingConfig, RotationStrategy};
use crate::types::ProxyError;
use tracing::Level;
use tracing_appender::rolling::{Rotation, RollingFileAppender};
use tracing_subscriber::{
    fmt::{self},
    prelude::*,
    EnvFilter,
};

/// Initialize the logging system based on the provided configuration
///
/// Sets up the tracing subscriber with:
/// - Console output (human-readable) if enabled
/// - File output (JSON) with rotation if enabled
/// - Configurable log level
pub fn init_logging(config: &LoggingConfig) -> Result<(), ProxyError> {
    // Parse log level
    let log_level = match config.level.to_lowercase().as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => {
            return Err(ProxyError::ConfigError(format!(
                "Invalid log level: {}",
                config.level
            )));
        }
    };

    // Create a filter with the configured level
    let filter = EnvFilter::builder()
        .with_default_directive(log_level.into())
        .from_env_lossy();

    // Start building the subscriber
    let mut layers = Vec::new();

    // Add console layer if enabled (human-readable formatting)
    if config.console {
        let console_layer = fmt::layer()
            .with_target(true)
            .with_ansi(true)
            .with_writer(std::io::stdout);
        layers.push(console_layer.boxed());
    }

    // Add file layer if enabled
    if let Some(file_config) = &config.file {
        if file_config.enabled {
            // Get the log file path
            let path = file_config
                .path
                .as_deref()
                .ok_or_else(|| ProxyError::ConfigError("File logging enabled but no path provided".into()))?;

            // Get directory and filename prefix from path
            let (directory, filename_prefix) = split_path(path);

            // Create appender based on rotation strategy
            let max_log_files = file_config.max_files.unwrap_or(5);
            let builder = RollingFileAppender::builder()
                .filename_prefix(filename_prefix)
                .max_log_files(max_log_files as usize);

            let appender = match file_config.rotation.unwrap_or(RotationStrategy::Never) {
                RotationStrategy::Daily => {
                    builder
                        .rotation(Rotation::DAILY)
                        .build(directory)
                        .map_err(|e| ProxyError::ConfigError(format!("Failed to create daily log appender: {}", e)))?
                }
                // tracing-appender 0.2.5 primarily supports time-based rotation.
                // For size-based rotation, the PRD asks for size-based rotation so we keep the configuration
                // and use daily rotation with max files, which is the closest approximation
                // that tracing-appender supports in this version.
                RotationStrategy::Size => {
                    builder
                        .rotation(Rotation::DAILY)
                        .build(directory)
                        .map_err(|e| ProxyError::ConfigError(format!("Failed to create log appender: {}", e)))?
                }
                RotationStrategy::Never => {
                    builder
                        .rotation(Rotation::NEVER)
                        .build(directory)
                        .map_err(|e| ProxyError::ConfigError(format!("Failed to create log appender: {}", e)))?
                }
            };

            // Create JSON-formatted file layer
            let file_layer = fmt::layer()
                .json()
                .with_target(true)
                .with_writer(appender);
            layers.push(file_layer.boxed());
        }
    }

    // If neither console nor file is enabled, still add a minimal console output
    // to avoid silent failure
    if layers.is_empty() {
        let console_layer = fmt::layer()
            .with_target(true)
            .with_ansi(true)
            .with_writer(std::io::stdout);
        layers.push(console_layer.boxed());
    }

    // Build and set the global subscriber
    tracing_subscriber::Registry::default()
        .with(filter)
        .with(layers)
        .try_init()
        .map_err(|e| ProxyError::ConfigError(format!("Failed to initialize logging: {}", e)))?;

    Ok(())
}

/// Split a path into directory and filename
///
/// If the path has no directory component, returns "." as the directory
fn split_path(path: &str) -> (&str, &str) {
    match path.rsplit_once('/') {
        Some((dir, file)) => (dir, file),
        None => {
            // No directory separator, use current directory
            (".", path)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_path() {
        assert_eq!(split_path("/var/log/app.log"), ("/var/log", "app.log"));
        assert_eq!(split_path("logs/app.log"), ("logs", "app.log"));
        assert_eq!(split_path("app.log"), (".", "app.log"));
    }
}
