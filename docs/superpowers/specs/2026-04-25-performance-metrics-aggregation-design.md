# Performance Metrics Aggregation Design

## Overview

Replace per-request token statistics writing (`token_stats.jsonl`) with in-memory buffering and periodic aggregated statistics writing (`stats.jsonl`).

## Changes

### 1. Configuration

**File:** `src/config.rs`

Add to `StatisticsConfig`:
```rust
pub struct StatisticsConfig {
    pub enabled: bool,
    pub buffer_seconds: Option<f64>,
    pub stats_file: String,  // NEW: aggregated stats file path
}
```

### 2. Data Structures

**File:** `src/stats.rs`

In-memory request record (replaces `TokenStats` for buffering):
```rust
struct RequestMetrics {
    model: String,
    provider: String,
    prompt_tokens: u64,
    completion_tokens: u64,
    duration_ms: u64,
    ttft_ms: Option<u64>,  // streaming only
    tpot_ms: Option<f64>,  // streaming only, calculated as duration_ms / completion_tokens
    status: String,
}
```

Aggregated output record:
```rust
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
```

### 3. TTFT Tracking

**File:** `src/proxy.rs` - `handle_streaming`

- Record `first_chunk_time` when the first SSE chunk is received
- Calculate `ttft_ms = first_chunk_time - request_start_time`
- Pass TTFT through the stream processing to `write_stat`

### 4. StatsWriter Changes

**File:** `src/stats.rs`

**Removed:**
- `TokenStats` struct (replaced by `RequestMetrics`)
- `stats.jsonl` per-request writes

**Added:**
- `RequestMetrics` struct
- Per-model buffering: `HashMap<String, Vec<RequestMetrics>>`
- 60-second aggregation interval
- `AggregatedStats` writing to `stats.jsonl`
- Percentile calculation (p50, p95, p99)

**Aggregation logic:**
1. Every 60s, iterate through all buffered requests
2. Group by model + provider
3. Calculate aggregates per group
4. Write one `AggregatedStats` line per model/provider
5. Clear the buffer for aggregated periods

### 5. Flow

```
Request arrives
    │
    ▼
handle_streaming / handle_non_streaming
    │  Record: model, provider, duration_ms, ttft_ms
    ▼
stats_writer.write_metric(RequestMetrics)
    │
    ▼ (every 60s)
aggregate_by_model_provider()
    │
    ▼
write AggregatedStats to stats.jsonl
```

## File Summary

| File | Change |
|------|--------|
| `src/config.rs` | Add `stats_file` to `StatisticsConfig` |
| `src/stats.rs` | Full rewrite: remove TokenStats, add RequestMetrics, aggregation logic |
| `src/proxy.rs` | Track TTFT in streaming handler |
| `config.yaml` | Add `stats_file` field |
