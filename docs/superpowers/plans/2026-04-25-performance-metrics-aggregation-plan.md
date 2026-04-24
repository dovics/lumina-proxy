# Performance Metrics Aggregation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-request token_stats.jsonl with in-memory buffering and 60s aggregated stats written to stats.jsonl, including TTFT and TPOT metrics for streaming requests.

**Architecture:**
- `StatsWriter` holds per-model in-memory buffers of `RequestMetrics`
- A background Tokio task runs every 60s to aggregate and flush
- TTFT tracked in streaming handler via chunk arrival timestamps
- Non-streaming requests don't have TTFT/TPOT (only duration_ms)

**Tech Stack:** Rust, Tokio, serde, chrono

---

## Task 1: Update StatisticsConfig in config.rs

**Files:**
- Modify: `src/config.rs:89-97`
- Modify: `src/proxy.rs:1109-1113` (reload_config_handler comparison)

- [ ] **Step 1: Modify StatisticsConfig struct**

Replace:
```rust
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
```

With:
```rust
/// Statistics configuration
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct StatisticsConfig {
    /// Whether statistics collection is enabled
    pub enabled: bool,
    /// Path to aggregated stats file (JSONL format)
    pub stats_file: Option<String>,
    /// Aggregation interval in seconds (default: 60)
    pub aggregation_interval_secs: Option<u64>,
}
```

- [ ] **Step 2: Update reload_config_handler comparison**

Find and update the comparison in `reload_config_handler` (around line 1109) that references `file_path` and `buffer_seconds`. Replace with:

```rust
if new_config.statistics.stats_file != current_config.statistics.stats_file
    || new_config.statistics.aggregation_interval_secs != current_config.statistics.aggregation_interval_secs
{
    warnings.push("Statistics config changes require server restart to take effect.");
}
```

- [ ] **Step 3: Commit**

```bash
git add src/config.rs src/proxy.rs
git commit -m "refactor: rename stats config fields for aggregation"
```

---

## Task 2: Rewrite stats.rs - replace TokenStats with aggregation logic

**Files:**
- Modify: `src/stats.rs` (complete rewrite)

- [ ] **Step 1: Write new stats.rs**

```rust
//! Statistics persistence - aggregates and writes performance metrics to stats.jsonl

use chrono::{DateTime, Utc};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{Duration, Instant, Interval};
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
    /// Buffered request metrics per model
    buffer: Arc<Mutex<HashMap<String, Vec<RequestMetrics>>>>,
    /// Flush interval
    flush_interval: Duration,
    /// Last flush time
    last_flush: Arc<Mutex<Instant>>,
    /// Stats file path
    stats_file: Option<String>,
    /// Writer for stats file
    writer: Arc<Mutex<Option<BufWriter<tokio::fs::File>>>>,
    /// Whether statistics are enabled
    enabled: bool,
    /// Shutdown signal receiver
    shutdown_rx: Arc<Mutex<Option<tokio::sync::watch::Receiver<bool>>>>,
}

impl StatsWriter {
    /// Create a new StatsWriter from configuration
    pub async fn new(config: &StatisticsConfig) -> Result<Self, ProxyError> {
        if !config.enabled {
            return Ok(Self {
                buffer: Arc::new(Mutex::new(HashMap::new())),
                flush_interval: Duration::from_secs(60),
                last_flush: Arc::new(Mutex::new(Instant::now())),
                stats_file: None,
                writer: Arc::new(Mutex::new(None)),
                enabled: false,
                shutdown_rx: Arc::new(Mutex::new(None)),
            });
        }

        let stats_file = config.stats_file.as_ref()
            .ok_or_else(|| ProxyError::ConfigError("Statistics enabled but stats_file not provided".to_string()))?;

        // Open file in append mode, create if it doesn't exist
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(stats_file)
            .await
            .map_err(|e| ProxyError::ConfigError(format!("Failed to open stats file: {}", e)))?;

        let buf_writer = BufWriter::new(file);
        let interval_secs = config.aggregation_interval_secs.unwrap_or(60);

        Ok(Self {
            buffer: Arc::new(Mutex::new(HashMap::new())),
            flush_interval: Duration::from_secs(interval_secs),
            last_flush: Arc::new(Mutex::new(Instant::now())),
            stats_file: Some(stats_file.clone()),
            writer: Arc::new(Mutex::new(Some(buf_writer))),
            enabled: true,
            shutdown_rx: Arc::new(Mutex::new(None)),
        })
    }

    /// Start the background aggregation task
    pub fn start_aggregation_task(&self) {
        if !self.enabled {
            return;
        }

        let buffer = self.buffer.clone();
        let flush_interval = self.flush_interval;
        let last_flush = self.last_flush.clone();
        let writer = self.writer.clone();
        let stats_file = self.stats_file.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(flush_interval);

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        // Flush all buffered metrics
                        let mut buf = buffer.lock().await;
                        if buf.is_empty() {
                            continue;
                        }

                        let now = Utc::now();
                        let period_end = now;
                        let period_start = now - chrono::Duration::seconds(flush_interval.as_secs() as i64);

                        // Aggregate by model:provider
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

                            // Percentiles
                            let mut durations: Vec<u64> = metrics.iter().map(|m| m.duration_ms).collect();
                            durations.sort();
                            let p50_idx = ((request_count as f64 * 0.50) as usize).min(durations.len() - 1);
                            let p95_idx = ((request_count as f64 * 0.95) as usize).min(durations.len() - 1);
                            let p99_idx = ((request_count as f64 * 0.99) as usize).min(durations.len() - 1);
                            let p50_duration_ms = durations.get(p50_idx).copied().unwrap_or(0) as f64;
                            let p95_duration_ms = durations.get(p95_idx).copied().unwrap_or(0) as f64;
                            let p99_duration_ms = durations.get(p99_idx).copied().unwrap_or(0);

                            // TTFT and TPOT (streaming only)
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

                        // Write to file
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

    /// Write a request metric to the buffer
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
```

- [ ] **Step 2: Run build to check for errors**

```bash
cargo build 2>&1
```

Expected: compilation errors (missing types, ProxyState references StatsWriter) - we'll fix in next tasks

- [ ] **Step 3: Commit**

```bash
git add src/stats.rs
git commit -m "feat: rewrite stats with aggregation buffering and background task"
```

---

## Task 3: Update proxy.rs imports and streaming TTFT tracking

**Files:**
- Modify: `src/proxy.rs:22` (import change)
- Modify: `src/proxy.rs:320-869` (handle_streaming - add TTFT tracking)
- Modify: `src/proxy.rs:876-942` (proxy_handler - call start_aggregation_task)

- [ ] **Step 1: Change import in proxy.rs**

Replace line 22:
```rust
use crate::stats::TokenStats;
```

With:
```rust
use crate::stats::{StatsWriter, RequestMetrics};
```

- [ ] **Step 2: Add TTFT tracking in handle_streaming**

In `handle_streaming`, after `let start_time = std::time::Instant::now();` (line 333), add:

```rust
let first_bytes_time = Arc::new(Mutex::new(None::<Instant>));
```

In the inner stream unfold (the `unfold` that processes chunks), record `first_bytes_time` when bytes arrive from upstream (NOT when yielding). This happens at:

```rust
match bytes_stream.next().await {
    Some(bytes_result) => {
        let bytes = match bytes_result {
            Ok(b) => b,
            ...
        };
        // Record TTFT clock: first bytes received from upstream
        {
            let mut first_time = first_bytes_time.lock().await;
            if first_time.is_none() {
                *first_time = Some(Instant::now());
            }
        }
        ...
    }
    ...
}
```

TTFT = `first_bytes_time - start_time`. This is the true time-to-first-token measuring when upstream data actually arrived, not when we finished processing it.

- [ ] **Step 3: Calculate TTFT and TPOT at stream completion**

In the final_stream unfold (the one that wraps the transformed stream and writes stats), after getting completion_tokens:

```rust
let first_bytes_guard = first_bytes_time.lock().await;
let ttft_ms = first_bytes_guard.map(|t| t.elapsed().as_millis() as u64);
let tpot_ms = if completion_tokens > 0 {
    Some(duration_ms as f64 / completion_tokens as f64)
} else {
    None
};
```

- [ ] **Step 4: Change write_stat calls to write_metric**

Replace `TokenStats` creation and `stats_writer.write_stat(stats)` with:

```rust
let metric = RequestMetrics {
    model: model.clone(),
    provider: format!("{:?}", route.provider_type).to_lowercase(),
    prompt_tokens: prompt_tokens as u64,
    completion_tokens: completion_tokens as u64,
    duration_ms,
    ttft_ms,
    tpot_ms,
    status: "success".to_string(),
};
if let Err(e) = stats_writer.write_metric(metric).await { ... }
```

For error case: `ttft_ms: None`, `tpot_ms: None`, `status: "error"`.

- [ ] **Step 5: Start aggregation task in proxy_handler**

In `proxy_handler`, after `StatsWriter::new()` succeeds, call:

```rust
stats_writer.start_aggregation_task();
```

- [ ] **Step 6: Run build to check**

```bash
cargo build 2>&1
```

Expected: errors about StatsWriter not having write_metric method - fixed in task 2

- [ ] **Step 7: Commit**

```bash
git add src/proxy.rs
git commit -m "feat: track TTFT/TPOT in streaming handler and use RequestMetrics"
```

---

## Task 4: Update config.yaml

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Update statistics config section**

Replace:
```yaml
# Token 统计持久化配置
statistics:
  enabled: true
  file_path: "./token_stats.jsonl"  # JSONL 文件路径
  buffer_seconds: 1.0              # 缓冲写入多少秒后刷盘，减少 IO
```

With:
```yaml
# 统计持久化配置
statistics:
  enabled: true
  stats_file: "./stats.jsonl"      # 聚合统计文件路径
  aggregation_interval_secs: 60   # 聚合间隔（秒）
```

- [ ] **Step 2: Commit**

```bash
git add config.yaml
git commit -m "feat: update statistics config for aggregated stats"
```

---

## Task 5: Run tests and verify

**Files:**
- Test: `tests/config_tests.rs`
- Test: `tests/conversion_tests.rs`

- [ ] **Step 1: Run all tests**

```bash
cargo test 2>&1
```

Expected: all tests pass

- [ ] **Step 2: Run clippy**

```bash
cargo clippy 2>&1
```

Expected: no warnings

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: implement performance metrics aggregation with TTFT/TPOT"
```

---
