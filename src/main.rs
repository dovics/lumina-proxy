//! Lumina-Proxy - LLM Routing Proxy
//! Main entry point

#![cfg_attr(windows, windows_subsystem = "windows")]

use std::sync::Arc;
use anyhow::Result;
use arc_swap::ArcSwap;
use axum::Router;
use axum::routing::{get, post};
use tokio::sync::mpsc;

use lumina::config::Config;
use lumina::logging::init_logging;
use lumina::stats::StatsWriter;
use lumina::proxy::{ProxyState, models_handler, proxy_handler, reload_config_handler, get_config_handler};
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

    // Create shared configuration - accessible from both server and tray
    let shared_config = Arc::new(ArcSwap::from_pointee(config));

    // Spawn Axum server on a background thread
    let server_addr = format!("{}:{}", shared_config.load().server.host, shared_config.load().server.port);
    let server_config = shared_config.clone();
    let server_config_path = config_path.clone();
    let server_handle = std::thread::spawn(move || {
        // Create a tokio runtime for the server thread
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(run_server_with_shared_config(server_config, server_config_path, shutdown_rx))
    });

    // Run tray on main thread (Windows only) - pass shared config for reload functionality
    tracing::info!("Starting tray on main thread");
    if let Err(e) = TrayManager::new(shutdown_tx, shared_config, config_path)?.run(server_addr) {
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
    run_server(config, config_path, shutdown_rx).await
}

/// Build reqwest Client with proxy configuration
fn build_client(config: &Config) -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder();

    // Apply proxy configuration if configured
    if let Some(proxy_config) = &config.server.proxy {
        // HTTP proxy
        if let Some(http_proxy) = &proxy_config.http {
            let proxy = reqwest::Proxy::http(http_proxy)?;
            builder = builder.proxy(proxy);
        }

        // HTTPS proxy
        if let Some(https_proxy) = &proxy_config.https {
            let proxy = reqwest::Proxy::https(https_proxy)?;
            builder = builder.proxy(proxy);
        }

        // No proxy (bypass list) - handled via reqwest Proxy::custom or environment variables
        // For no_proxy support, users can also set NO_PROXY environment variable
    }

    Ok(builder.build()?)
}

async fn run_server_with_shared_config(
    shared_config: Arc<ArcSwap<Config>>,
    config_path: String,
    mut shutdown_rx: mpsc::Receiver<()>,
) -> Result<()> {
    // Get a snapshot of current config for initialization
    let config = shared_config.load();

    // Initialize HTTP client with proxy support
    let client = build_client(&config)?;

    // Initialize stats writer if statistics enabled
    let stats_writer = if config.statistics.enabled {
        Some(StatsWriter::new(&config.statistics).await?)
    } else {
        None
    };

    // Create proxy state - use the shared config from Windows main thread
    let proxy_state = Arc::new(ProxyState {
        config: shared_config,
        config_path: config_path.clone(),
        client,
        stats_writer: Arc::new(ArcSwap::from_pointee(stats_writer)),
    });

    // Build Axum router
    let mut router = Router::new()
        .route("/v1/chat/completions", post(proxy_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/admin/reload-config", post(reload_config_handler))
        .route("/v1/admin/config", get(get_config_handler));

    // Add authentication middleware if auth token is configured
    if config.server.auth_token.is_some() {
        router = router.layer(axum::middleware::from_fn_with_state(proxy_state.clone(), auth_middleware));
    }

    // Add proxy state to router
    let router = router.with_state(proxy_state);

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

/// Simple wrapper for non-Windows platforms - creates a new config instance
#[cfg(not(windows))]
async fn run_server(config: Config, config_path: String, shutdown_rx: mpsc::Receiver<()>) -> Result<()> {
    let shared_config = Arc::new(ArcSwap::from_pointee(config));
    run_server_with_shared_config(shared_config, config_path, shutdown_rx).await
}
