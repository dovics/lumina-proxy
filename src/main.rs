//! Lumina-Proxy - LLM Routing Proxy
//! Main entry point

#![cfg_attr(windows, windows_subsystem = "windows")]

use std::sync::Arc;
use anyhow::Result;
use axum::Router;
use axum::routing::{get, post};
use tokio::sync::mpsc;

use lumina::config::Config;
use lumina::logging::init_logging;
use lumina::stats::StatsWriter;
use lumina::proxy::{ProxyState, models_handler, proxy_handler};
use lumina::auth::auth_middleware;
#[cfg(windows)]
use lumina::tray::TrayManager;

#[cfg(windows)]
fn main() -> Result<()> {
    // Windows: Tao event loop must run on main thread
    // So we run Axum on a background thread

    // Get config path from command line argument or use default
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./config.yaml".to_string());

    // Load configuration
    let config = Config::load_from_file(&config_path)?;

    // Initialize logging
    init_logging(&config.logging)?;

    // Create shutdown channel
    let (shutdown_tx, shutdown_rx) = mpsc::channel::<()>(1);

    // Spawn Axum server on a background thread
    let server_addr = format!("{}:{}", config.server.host, config.server.port);
    let server_config = config.clone();
    let server_handle = std::thread::spawn(move || {
        // Create a tokio runtime for the server thread
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(run_server(server_config, shutdown_rx))
    });

    // Run tray on main thread (Windows only)
    tracing::info!("Starting tray on main thread");
    if let Err(e) = TrayManager::new(shutdown_tx)?.run(server_addr) {
        tracing::warn!("Tray failed to start: {}, running without tray", e);
    }

    // Wait for server to finish
    server_handle.join().unwrap()?;

    Ok(())
}

#[cfg(not(windows))]
#[tokio::main]
async fn main() -> Result<()> {
    // Non-Windows: Simple, just run Axum on main thread
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./config.yaml".to_string());

    let config = Config::load_from_file(&config_path)?;
    init_logging(&config.logging)?;

    let (_shutdown_tx, shutdown_rx) = mpsc::channel::<()>(1);
    run_server(config, shutdown_rx).await
}

/// Run the Axum server with graceful shutdown support
async fn run_server(config: Config, mut shutdown_rx: mpsc::Receiver<()>) -> Result<()> {
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

    // Bind to configured host/port and serve with graceful shutdown
    let addr = format!("{}:{}", config.server.host, config.server.port);
    tracing::info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    // Use graceful shutdown
    axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            // Wait for shutdown signal
            let _ = shutdown_rx.recv().await;
            tracing::info!("Received shutdown signal, initiating graceful shutdown");
        })
        .await?;

    tracing::info!("Server shutdown complete");
    Ok(())
}
