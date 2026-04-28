//! Proxy state - shared application state

use crate::config::Config;
use crate::stats::StatsWriter;
use arc_swap::ArcSwap;
use std::sync::Arc;

/// Shared proxy state that is held by the Axum server
#[derive(Clone)]
pub struct ProxyState {
    /// Application configuration - atomically updatable
    pub config: Arc<ArcSwap<Config>>,
    /// Path to configuration file for reloading
    pub config_path: String,
    /// reqwest HTTP client for backend requests
    pub client: reqwest::Client,
    /// Optional statistics writer for token usage logging - atomically updatable
    pub stats_writer: Arc<ArcSwap<Option<StatsWriter>>>,
}
