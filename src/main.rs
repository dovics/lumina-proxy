//! Lumina-Proxy - LLM Routing Proxy
//! Main entry point

use std::sync::Arc;
use anyhow::Result;
use axum::Router;
use axum::routing::{get, post};

use lumina_proxy::config::Config;
use lumina_proxy::logging::init_logging;
use lumina_proxy::stats::StatsWriter;
use lumina_proxy::proxy::{ProxyState, models_handler, proxy_handler};
use lumina_proxy::auth::auth_middleware;

#[tokio::main]
async fn main() -> Result<()> {
    // Get config path from command line argument or use default
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./config.yaml".to_string());

    // Load configuration
    tracing::info!("Loading configuration from {}", config_path);
    let config = Config::load_from_file(&config_path)?;

    // Initialize logging
    init_logging(&config.logging)?;

    // Initialize HTTP client
    let client = reqwest::Client::new();

    // Initialize stats writer if statistics enabled
    let stats_writer = if config.statistics.enabled {
        Some(StatsWriter::new(&config.statistics).await?)
    } else {
        None
    };

    // Create proxy state
    let proxy_state = Arc::new(ProxyState {
        config: config.clone(),
        client,
        stats_writer,
    });

    // Build Axum router
    let mut router = Router::new()
        .route("/v1/chat/completions", post(proxy_handler))
        .route("/v1/models", get(models_handler))
        .with_state(proxy_state);

    // Add authentication middleware if auth token is configured
    if config.server.auth_token.is_some() {
        router = router.layer(axum::middleware::from_fn_with_state(config.clone(), auth_middleware));
    }

    // Bind to configured host/port and serve
    let addr = format!("{}:{}", config.server.host, config.server.port);
    tracing::info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}
