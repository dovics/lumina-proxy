//! Statistics persistence - writes token usage statistics to JSONL file

use chrono::{DateTime, Utc};
use serde::Serialize;
use tokio::fs::OpenOptions;
use tokio::io::{AsyncWriteExt, BufWriter};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{Duration, Instant};

use crate::config::StatisticsConfig;
use crate::types::ProxyError;

/// Token statistics for a single LLM request
#[derive(Debug, Clone, Serialize)]
pub struct TokenStats {
    /// Timestamp when the request completed
    pub timestamp: DateTime<Utc>,
    /// Model name used for the request
    pub model: String,
    /// Provider that handled the request
    pub provider: String,
    /// Number of tokens in the prompt
    pub prompt_tokens: usize,
    /// Number of tokens in the completion
    pub completion_tokens: usize,
    /// Total tokens used (prompt + completion)
    pub total_tokens: usize,
    /// Total request duration in milliseconds
    pub duration_ms: u64,
    /// Request status: "success" or "error"
    pub status: String,
    /// Error message if status is "error"
    pub error_message: Option<String>,
}

/// Statistics writer that buffers writes and periodically flushes to disk
///
/// Uses buffered async writing to reduce IO operations. Flush happens when:
/// 1. Buffer has pending writes and the configured buffer interval has elapsed
/// 2. Or when explicitly flushed (not currently used)
#[derive(Clone)]
pub struct StatsWriter {
    /// Buffered writer protected by mutex
    writer: Arc<Mutex<Option<BufWriter<tokio::fs::File>>>>,
    /// Buffer flush interval
    flush_interval: Duration,
    /// Last flush time
    last_flush: Arc<Mutex<Instant>>,
    /// Whether statistics are enabled
    enabled: bool,
}

impl StatsWriter {
    /// Create a new StatsWriter from configuration
    pub async fn new(config: &StatisticsConfig) -> Result<Self, ProxyError> {
        if !config.enabled {
            return Ok(Self {
                writer: Arc::new(Mutex::new(None)),
                flush_interval: Duration::from_secs_f64(1.0),
                last_flush: Arc::new(Mutex::new(Instant::now())),
                enabled: false,
            });
        }

        let file_path = config.file_path.as_ref()
            .ok_or_else(|| ProxyError::ConfigError("Statistics enabled but file_path not provided".to_string()))?;

        // Open file in append mode, create if it doesn't exist
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(file_path)
            .await
            .map_err(|e| ProxyError::ConfigError(format!("Failed to open statistics file: {}", e)))?;

        let buf_writer = BufWriter::new(file);
        let buffer_seconds = config.buffer_seconds.unwrap_or(1.0);
        let flush_interval = Duration::from_secs_f64(buffer_seconds);

        Ok(Self {
            writer: Arc::new(Mutex::new(Some(buf_writer))),
            flush_interval,
            last_flush: Arc::new(Mutex::new(Instant::now())),
            enabled: true,
        })
    }

    /// Write statistics to the buffer
    ///
    /// If the flush interval has elapsed since the last flush, flushes to disk.
    pub async fn write_stat(&self, stats: TokenStats) -> Result<(), ProxyError> {
        if !self.enabled {
            return Ok(());
        }

        // Serialize to JSON line
        let mut json = serde_json::to_string(&stats)
            .map_err(|e| ProxyError::BackendRequestError(format!("Failed to serialize stats: {}", e)))?;
        json.push('\n');

        // Get writer lock and write
        let mut writer_guard = self.writer.lock().await;
        if let Some(writer) = writer_guard.as_mut() {
            writer.write_all(json.as_bytes())
                .await
                .map_err(|e| ProxyError::BackendRequestError(format!("Failed to write stats: {}", e)))?;
        }

        // Check if we need to flush
        let mut last_flush_guard = self.last_flush.lock().await;
        let now = Instant::now();
        if now.duration_since(*last_flush_guard) >= self.flush_interval {
            if let Some(writer) = writer_guard.as_mut() {
                writer.flush()
                    .await
                    .map_err(|e| ProxyError::BackendRequestError(format!("Failed to flush stats: {}", e)))?;
                *last_flush_guard = now;
            }
        }

        Ok(())
    }

    /// Force flush any buffered writes to disk
    ///
    /// Useful for shutdown to ensure all data is written.
    pub async fn flush(&self) -> Result<(), ProxyError> {
        if !self.enabled {
            return Ok(());
        }

        let mut writer_guard = self.writer.lock().await;
        if let Some(writer) = writer_guard.as_mut() {
            writer.flush()
                .await
                .map_err(|e| ProxyError::BackendRequestError(format!("Failed to flush stats: {}", e)))?;
        }

        let mut last_flush_guard = self.last_flush.lock().await;
        *last_flush_guard = Instant::now();

        Ok(())
    }
}
