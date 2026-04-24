//! Statistics persistence - aggregates and writes performance metrics to stats.jsonl

use chrono::{DateTime, Utc};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{Duration, Instant};
use tokio::fs::OpenOptions;
use tokio::io::{AsyncWriteExt, BufWriter};

use crate::config::StatisticsConfig;
use crate::types::ProxyError;

/// In-memory request metrics for aggregation buffering
#[derive(Debug, Clone)]
pub struct RequestMetrics {
    pub model: String,
    pub provider: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub duration_ms: u64,
    pub ttft_ms: Option<u64>,
    pub tpot_ms: Option<f64>,
    pub status: String,
}

/// Aggregated statistics written to stats.jsonl
#[derive(Debug, Clone, Serialize)]
pub struct AggregatedStats {
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub model: String,
    pub provider: String,
    pub request_count: u64,
    pub success_count: u64,
    pub error_count: u64,
    pub avg_duration_ms: f64,
    pub p50_duration_ms: f64,
    pub p95_duration_ms: f64,
    pub p99_duration_ms: u64,
    pub avg_ttft_ms: Option<f64>,
    pub avg_tpot_ms: Option<f64>,
    pub total_prompt_tokens: u64,
    pub total_completion_tokens: u64,
}

/// Statistics writer with in-memory buffering and periodic aggregation
#[derive(Clone)]
pub struct StatsWriter {
    buffer: Arc<Mutex<HashMap<String, Vec<RequestMetrics>>>>,
    flush_interval: Duration,
    last_flush: Arc<Mutex<Instant>>,
    stats_file: Option<String>,
    enabled: bool,
}

impl StatsWriter {
    pub async fn new(config: &StatisticsConfig) -> Result<Self, ProxyError> {
        if !config.enabled {
            return Ok(Self {
                buffer: Arc::new(Mutex::new(HashMap::new())),
                flush_interval: Duration::from_secs(60),
                last_flush: Arc::new(Mutex::new(Instant::now())),
                stats_file: None,
                enabled: false,
            });
        }

        let stats_file = config.stats_file.as_ref()
            .ok_or_else(|| ProxyError::ConfigError("Statistics enabled but stats_file not provided".to_string()))?;

        let interval_secs = config.aggregation_interval_secs.unwrap_or(60);

        Ok(Self {
            buffer: Arc::new(Mutex::new(HashMap::new())),
            flush_interval: Duration::from_secs(interval_secs),
            last_flush: Arc::new(Mutex::new(Instant::now())),
            stats_file: Some(stats_file.clone()),
            enabled: true,
        })
    }

    pub fn start_aggregation_task(&self) {
        if !self.enabled {
            return;
        }

        let buffer = self.buffer.clone();
        let flush_interval = self.flush_interval;
        let last_flush = self.last_flush.clone();
        let stats_file = self.stats_file.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(flush_interval);

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let mut buf = buffer.lock().await;
                        if buf.is_empty() {
                            continue;
                        }

                        let now = Utc::now();
                        let period_end = now;
                        let period_start = now - chrono::Duration::seconds(flush_interval.as_secs() as i64);

                        let mut aggregated: HashMap<String, AggregatedStats> = HashMap::new();

                        for (key, metrics) in buf.drain() {
                            if metrics.is_empty() {
                                continue;
                            }

                            let parts: Vec<&str> = key.split(':').collect();
                            let model = parts.get(0).unwrap_or(&"").to_string();
                            let provider = parts.get(1).unwrap_or(&"").to_string();

                            let request_count = metrics.len() as u64;
                            let success_count = metrics.iter().filter(|m| m.status == "success").count() as u64;
                            let error_count = metrics.iter().filter(|m| m.status == "error").count() as u64;

                            let total_duration: u64 = metrics.iter().map(|m| m.duration_ms).sum();
                            let avg_duration_ms = total_duration as f64 / request_count as f64;

                            let mut durations: Vec<u64> = metrics.iter().map(|m| m.duration_ms).collect();
                            durations.sort();
                            let p50_idx = ((request_count as f64 * 0.50) as usize).min(durations.len().saturating_sub(1));
                            let p95_idx = ((request_count as f64 * 0.95) as usize).min(durations.len().saturating_sub(1));
                            let p99_idx = ((request_count as f64 * 0.99) as usize).min(durations.len().saturating_sub(1));
                            let p50_duration_ms = durations.get(p50_idx).copied().unwrap_or(0) as f64;
                            let p95_duration_ms = durations.get(p95_idx).copied().unwrap_or(0) as f64;
                            let p99_duration_ms = durations.get(p99_idx).copied().unwrap_or(0);

                            let ttft_values: Vec<u64> = metrics.iter().filter_map(|m| m.ttft_ms).collect();
                            let avg_ttft_ms = if ttft_values.is_empty() {
                                None
                            } else {
                                Some(ttft_values.iter().sum::<u64>() as f64 / ttft_values.len() as f64)
                            };

                            let tpot_values: Vec<f64> = metrics.iter().filter_map(|m| m.tpot_ms).collect();
                            let avg_tpot_ms = if tpot_values.is_empty() {
                                None
                            } else {
                                Some(tpot_values.iter().sum::<f64>() / tpot_values.len() as f64)
                            };

                            let total_prompt: u64 = metrics.iter().map(|m| m.prompt_tokens).sum();
                            let total_completion: u64 = metrics.iter().map(|m| m.completion_tokens).sum();

                            let agg = AggregatedStats {
                                period_start,
                                period_end,
                                model,
                                provider,
                                request_count,
                                success_count,
                                error_count,
                                avg_duration_ms,
                                p50_duration_ms,
                                p95_duration_ms,
                                p99_duration_ms,
                                avg_ttft_ms,
                                avg_tpot_ms,
                                total_prompt_tokens: total_prompt,
                                total_completion_tokens: total_completion,
                            };

                            aggregated.insert(key, agg);
                        }

                        if let Some(ref path) = stats_file {
                            let file = OpenOptions::new()
                                .create(true)
                                .append(true)
                                .open(path)
                                .await;
                            if let Ok(file) = file {
                                let mut buf_writer = BufWriter::new(file);
                                for agg in aggregated.values() {
                                    if let Ok(json) = serde_json::to_string(agg) {
                                        let _ = buf_writer.write_all((json + "\n").as_bytes()).await;
                                    }
                                }
                                let _ = buf_writer.flush().await;
                            }
                        }

                        *last_flush.lock().await = Instant::now();
                    }
                }
            }
        });
    }

    pub async fn write_metric(&self, metric: RequestMetrics) -> Result<(), ProxyError> {
        if !self.enabled {
            return Ok(());
        }

        let key = format!("{}:{}", metric.model, metric.provider);
        let mut buffer = self.buffer.lock().await;
        buffer.entry(key).or_insert_with(Vec::new).push(metric);

        Ok(())
    }
}