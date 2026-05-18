//! HTTP forward proxy server implementation

use super::auth::{ProxyAuthState, proxy_auth_middleware};
use super::handlers::{ProxyHandlerState, proxy_dispatch_handler};
use anyhow::Result;
use arc_swap::ArcSwap;
use axum::extract::ConnectInfo;
use axum::extract::DefaultBodyLimit;
use axum::Router;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use reqwest::Client;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::net::TcpStream;
use tokio::sync::broadcast;
use tower::Service;
use tower_http::trace::TraceLayer;

use crate::config::Config;

/// HTTP Forward Proxy Server
pub struct HttpForwardProxyServer {
    shared_config: Arc<ArcSwap<Config>>,
    shutdown: broadcast::Receiver<()>,
    client: Client,
}

impl HttpForwardProxyServer {
    /// Create a new HTTP forward proxy server
    pub fn new(shared_config: Arc<ArcSwap<Config>>) -> Result<Self> {
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(60))
            .pool_max_idle_per_host(10)
            .tcp_nodelay(true)
            .no_proxy()
            .build()?;

        Ok(Self {
            shared_config,
            shutdown: broadcast::channel(1).1, // Dummy receiver, replaced by with_shutdown
            client,
        })
    }

    /// Attach shutdown receiver
    pub fn with_shutdown(mut self, shutdown: broadcast::Receiver<()>) -> Self {
        self.shutdown = shutdown;
        self
    }

    /// Run the proxy server
    pub async fn run(self) -> Result<()> {
        let config = self.shared_config.load();
        let proxy_config = match &config.http_forward_proxy {
            Some(c) if c.enabled => c,
            _ => {
                tracing::info!("HTTP forward proxy is disabled in config");
                return Ok(());
            }
        };

        let bind_addr = format!("{}:{}", proxy_config.host, proxy_config.port);
        tracing::info!("Starting HTTP forward proxy on {}", bind_addr);

        // Build router with fallback handler for all requests
        let handler_state = ProxyHandlerState {
            shared_config: self.shared_config.clone(),
            client: self.client,
        };

        let auth_state = ProxyAuthState {
            shared_config: self.shared_config.clone(),
        };

        // Get max connections from config (default 1024)
        let max_conns = proxy_config.max_connections.unwrap_or(1024);
        tracing::debug!("HTTP proxy max connections: {}", max_conns);

        // Get max request body size (default 100MB)
        let max_body_size = proxy_config
            .max_request_body_size
            .unwrap_or(100 * 1024 * 1024);
        tracing::debug!("HTTP proxy max request body: {} bytes", max_body_size);

        let app = Router::new()
            .fallback(proxy_dispatch_handler)
            .with_state(handler_state)
            .layer(axum::middleware::from_fn_with_state(
                auth_state,
                proxy_auth_middleware,
            ))
            // Limit concurrent connections
            .layer(tower::limit::ConcurrencyLimitLayer::new(max_conns))
            // Limit request body size
            .layer(DefaultBodyLimit::max(max_body_size))
            .layer(TraceLayer::new_for_http());

        // Bind TCP listener
        let listener = TcpListener::bind(&bind_addr).await?;
        tracing::info!("HTTP forward proxy listening on {}", bind_addr);

        let mut shutdown = self.shutdown;

        loop {
            tokio::select! {
                result = listener.accept() => {
                    match result {
                        Ok((stream, remote_addr)) => {
                            let app = app.clone();

                            tokio::spawn(async move {
                                if let Err(e) = handle_connection(stream, app, remote_addr).await {
                                    tracing::debug!("Connection error: {}", e);
                                }
                            });
                        }
                        Err(e) => {
                            tracing::error!("Accept error: {}", e);
                        }
                    }
                }
                _ = shutdown.recv() => {
                    tracing::info!("HTTP forward proxy initiating graceful shutdown");
                    break;
                }
            }
        }

        tracing::info!("HTTP forward proxy shutdown complete");
        Ok(())
    }
}

/// Handle a single connection - detect CONNECT and handle tunnel directly
async fn handle_connection(
    stream: TcpStream,
    app: Router,
    remote_addr: SocketAddr,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Peek at the first bytes to detect CONNECT request
    let mut buffer = [0u8; 4096];
    let n = match stream.peek(&mut buffer).await {
        Ok(n) => n,
        Err(e) => {
            tracing::debug!("Failed to peek connection: {}", e);
            return Ok(());
        }
    };

    if n == 0 {
        return Ok(());
    }

    let request_str = String::from_utf8_lossy(&buffer[..n]);

    if request_str.starts_with("CONNECT ") {
        // Handle HTTPS tunnel directly - bypass hyper's HTTP parser
        handle_https_tunnel(stream, app, remote_addr).await
    } else {
        // Handle normal HTTP request through Axum
        let io = TokioIo::new(stream);
        let service = service_fn(move |req| {
            let app = app.clone();
            async move {
                let mut req = req;
                // Add remote_addr to extensions for ConnectInfo extractor
                req.extensions_mut().insert(ConnectInfo(remote_addr));
                app.clone().call(req).await
            }
        });

        if let Err(e) = http1::Builder::new().serve_connection(io, service).await {
            tracing::debug!("Failed to serve HTTP connection: {}", e);
        }
        Ok(())
    }
}

/// Handle HTTPS tunnel - detect CONNECT, establish tunnel, proxy data
async fn handle_https_tunnel(
    mut client_stream: TcpStream,
    _app: Router,
    remote_addr: SocketAddr,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Read the CONNECT request
    let mut buffer = [0u8; 4096];
    let n = match client_stream.read(&mut buffer).await {
        Ok(n) => n,
        Err(e) => {
            tracing::debug!("Failed to read CONNECT request: {}", e);
            return Ok(());
        }
    };

    if n == 0 {
        return Ok(());
    }

    let request_str = String::from_utf8_lossy(&buffer[..n]);
    let lines: Vec<&str> = request_str.lines().collect();

    if lines.is_empty() {
        return Ok(());
    }

    let connect_line = lines[0];
    if !connect_line.starts_with("CONNECT ") {
        return Ok(());
    }

    let parts: Vec<&str> = connect_line.split_whitespace().collect();
    if parts.len() < 3 {
        return Ok(());
    }

    let authority = parts[1];

    tracing::debug!(
        "HTTPS tunnel request from {} to {}",
        remote_addr,
        authority
    );

    // Parse authority (host:port)
    let (host, port) = match parse_authority(authority) {
        Some(hp) => hp,
        None => {
            tracing::warn!("Invalid authority: {}", authority);
            let response = "HTTP/1.1 400 Bad Request\r\n\r\n";
            client_stream.write_all(response.as_bytes()).await?;
            return Ok(());
        }
    };

    // Connect to target server
    let target_addr = format!("{}:{}", host, port);
    let target_stream = match tokio::net::TcpStream::connect(&target_addr).await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("Failed to connect to target {}: {}", target_addr, e);
            let response = "HTTP/1.1 502 Bad Gateway\r\n\r\n";
            client_stream.write_all(response.as_bytes()).await?;
            return Ok(());
        }
    };

    // Enable TCP_NODELAY for lower latency
    if let Err(e) = target_stream.set_nodelay(true) {
        tracing::debug!("Failed to set TCP_NODELAY: {}", e);
    }

    tracing::debug!("HTTPS tunnel established to {}", authority);

    // Send 200 Connection Established
    let response = "HTTP/1.1 200 Connection Established\r\n\r\n";
    client_stream.write_all(response.as_bytes()).await?;
    client_stream.flush().await?;

    // Bidirectional copy using split
    let (mut client_read, mut client_write) = tokio::io::split(client_stream);
    let (mut target_read, mut target_write) = tokio::io::split(target_stream);

    let client_to_target = tokio::spawn(async move {
        let mut buffer = vec![0u8; 8192];
        loop {
            match client_read.read(&mut buffer).await {
                Ok(0) => break,
                Ok(n) => {
                    if let Err(e) = target_write.write_all(&buffer[..n]).await {
                        tracing::debug!("Client to target write error: {}", e);
                        break;
                    }
                    if let Err(e) = target_write.flush().await {
                        tracing::debug!("Client to target flush error: {}", e);
                        break;
                    }
                }
                Err(e) => {
                    tracing::debug!("Client to target read error: {}", e);
                    break;
                }
            }
        }
    });

    let target_to_client = tokio::spawn(async move {
        let mut buffer = vec![0u8; 8192];
        loop {
            match target_read.read(&mut buffer).await {
                Ok(0) => break,
                Ok(n) => {
                    if let Err(e) = client_write.write_all(&buffer[..n]).await {
                        tracing::debug!("Target to client write error: {}", e);
                        break;
                    }
                    if let Err(e) = client_write.flush().await {
                        tracing::debug!("Target to client flush error: {}", e);
                        break;
                    }
                }
                Err(e) => {
                    tracing::debug!("Target to client read error: {}", e);
                    break;
                }
            }
        }
    });

    let _ = tokio::join!(client_to_target, target_to_client);

    tracing::debug!("HTTPS tunnel closed for {}", authority);

    Ok(())
}

/// Parse authority string (host:port or [host]:port)
fn parse_authority(authority: &str) -> Option<(String, u16)> {
    if authority.starts_with('[') {
        // IPv6 format: [::1]:8080
        let closing_bracket = authority.find(']')?;
        let host = &authority[1..closing_bracket];
        let rest = &authority[closing_bracket + 1..];
        if let Some(port_str) = rest.strip_prefix(':') {
            let port = port_str.parse().ok()?;
            Some((host.to_string(), port))
        } else if rest.is_empty() {
            Some((host.to_string(), 443))
        } else {
            None
        }
    } else if authority.contains(':') && authority.split(':').count() > 2 {
        // Multiple colons but not IPv6 - take last :port
        let port = authority.rsplit(':').next()?.parse().ok()?;
        let colon = authority.rfind(':')?;
        let host = &authority[..colon];
        Some((host.to_string(), port))
    } else {
        // Simple host:port
        let parts: Vec<&str> = authority.rsplitn(2, ':').collect();
        if parts.len() == 2 {
            let port = parts[0].parse().ok()?;
            Some((parts[1].to_string(), port))
        } else {
            None
        }
    }
}
