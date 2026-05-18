# HTTP Forward Proxy Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add standalone HTTP forward proxy functionality to Lumina-Proxy, supporting HTTP request forwarding and HTTPS CONNECT tunneling with independent authentication.

**Architecture:**
- New `src/http_proxy/` module with server, handlers, and auth middleware
- Uses `tokio::sync::broadcast` for multi-service shutdown coordination
- Leverages existing `ArcSwap<Config>` for hot-reload of auth token and port restrictions
- Axum fallback handler for CONNECT method support (not natively supported by axum routing)

**Tech Stack:**
- Axum (web framework)
- reqwest (HTTP client for forwarding)
- tokio (async runtime, io::copy_bidirectional for tunneling)
- tower-http (middleware utilities)
- serde_yaml (config parsing)

---

## Prerequisites

- Read the spec: `docs/superpowers/specs/2026-05-18-http-forward-proxy-design.md`
- Check existing patterns:
  - `src/config.rs` - Config structs and validation patterns
  - `src/main.rs` - Server startup, shutdown channel, ArcSwap usage
  - `src/auth.rs` - Middleware patterns
  - `src/proxy/` - Existing handler patterns

---

## Task 1: Config Structure - Add HttpForwardProxyConfig

**Files:**
- Modify: `src/config.rs`
- Test: `tests/config_tests.rs`

---

- [ ] **Step 1: Add test for new config parsing**

Add to `tests/config_tests.rs`:

```rust
#[test]
fn test_http_forward_proxy_config_parsing() {
    let yaml = r#"
server:
  port: 3000
  host: 127.0.0.1

http_forward_proxy:
  enabled: true
  port: 8080
  host: 127.0.0.1
  auth_token: "test-secret"
  max_connections: 512
  idle_timeout_secs: 120
  max_request_body_size: 52428800
  allowed_target_ports: [80, 443, 8080]
  blocked_target_ports: [22, 3306]

logging:
  level: info

routes:
  - model_name: "gpt-4"
    provider_type: openai
    base_url: "https://api.openai.com/v1"
    enabled: true
"#;

    let config: Config = serde_yaml::from_str(yaml).unwrap();
    let proxy = config.http_forward_proxy.unwrap();
    
    assert!(proxy.enabled);
    assert_eq!(proxy.port, 8080);
    assert_eq!(proxy.host, "127.0.0.1");
    assert_eq!(proxy.auth_token, Some("test-secret".to_string()));
    assert_eq!(proxy.max_connections, Some(512));
    assert_eq!(proxy.idle_timeout_secs, Some(120));
    assert_eq!(proxy.allowed_target_ports, Some(vec![80, 443, 8080]));
    assert_eq!(proxy.blocked_target_ports, Some(vec![22, 3306]));
}

#[test]
fn test_http_forward_proxy_default_ports() {
    // Test that config without allowed/blocked ports uses defaults
    let yaml = r#"
server:
  port: 3000
  host: 127.0.0.1

http_forward_proxy:
  enabled: true
  port: 8080
  host: 127.0.0.1

logging:
  level: info

routes:
  - model_name: "gpt-4"
    provider_type: openai
    base_url: "https://api.openai.com/v1"
    enabled: true
"#;

    let config: Config = serde_yaml::from_str(yaml).unwrap();
    let proxy = config.http_forward_proxy.unwrap();
    
    // None means use defaults in code
    assert!(proxy.allowed_target_ports.is_none());
    assert!(proxy.blocked_target_ports.is_none());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test test_http_forward_proxy_config_parsing --test config_tests -v`
Expected: FAIL with "no field `http_forward_proxy` on type `Config`"

- [ ] **Step 3: Add HttpForwardProxyConfig struct to src/config.rs**

Add after `ServerConfig` (around line 100):

```rust
/// HTTP forward proxy configuration
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct HttpForwardProxyConfig {
    /// Whether the HTTP forward proxy is enabled
    pub enabled: bool,
    /// Port to listen on for proxy requests
    pub port: u16,
    /// Host address to bind to
    pub host: String,
    /// Authentication token for proxy access (optional)
    pub auth_token: Option<String>,
    /// Maximum concurrent connections (default: 1024)
    pub max_connections: Option<usize>,
    /// Idle timeout in seconds for CONNECT tunnels (default: 60)
    pub idle_timeout_secs: Option<u64>,
    /// Maximum request body size in bytes (default: 100MB)
    pub max_request_body_size: Option<usize>,
    /// Allowed target ports (None means use default whitelist)
    pub allowed_target_ports: Option<Vec<u16>>,
    /// Blocked target ports (None means use default blacklist)
    pub blocked_target_ports: Option<Vec<u16>>,
}

impl HttpForwardProxyConfig {
    /// Get default allowed target ports
    pub fn default_allowed_ports() -> Vec<u16> {
        vec![80, 443, 8080, 8081, 8082, 8083, 8084, 8085, 8086, 8087, 8088, 8089, 8090]
    }

    /// Get default blocked target ports
    pub fn default_blocked_ports() -> Vec<u16> {
        vec![22, 3306, 5432, 6379, 27017]
    }

    /// Get effective allowed ports (configured or default)
    pub fn allowed_ports(&self) -> Vec<u16> {
        self.allowed_target_ports
            .clone()
            .unwrap_or_else(Self::default_allowed_ports)
    }

    /// Get effective blocked ports (configured or default)
    pub fn blocked_ports(&self) -> Vec<u16> {
        self.blocked_target_ports
            .clone()
            .unwrap_or_else(Self::default_blocked_ports)
    }

    /// Check if a target port is allowed
    pub fn is_port_allowed(&self, port: u16) -> bool {
        let blocked = self.blocked_ports();
        if blocked.contains(&port) {
            return false;
        }
        self.allowed_ports().contains(&port)
    }
}
```

Then add the field to `Config` struct (around line 157, after `server`):

```rust
pub struct Config {
    /// Server configuration
    pub server: ServerConfig,
    /// HTTP forward proxy configuration (optional)
    pub http_forward_proxy: Option<HttpForwardProxyConfig>,
    /// Logging configuration
    pub logging: LoggingConfig,
    /// Statistics configuration
    pub statistics: StatisticsConfig,
    /// List of routes mapping models to backends
    pub routes: Vec<RouteConfig>,
    /// Configuration version (incremented on each reload)
    #[serde(skip)]
    pub version: u64,
    /// Timestamp when this configuration was loaded
    #[serde(skip)]
    pub loaded_at: chrono::DateTime<chrono::Utc>,
}
```

- [ ] **Step 4: Add validation for proxy config**

In `Config::validate()` (around line 185), add at the end before `Ok(())`:

```rust
// Validate HTTP forward proxy config if enabled
if let Some(proxy) = &self.http_forward_proxy {
    if proxy.enabled {
        if proxy.port == 0 {
            return Err("HTTP forward proxy port cannot be 0".to_string());
        }
        if proxy.port == self.server.port {
            return Err(format!(
                "HTTP forward proxy port ({}) cannot be the same as main server port ({})",
                proxy.port, self.server.port
            ));
        }
        if proxy.host.is_empty() {
            return Err("HTTP forward proxy host cannot be empty".to_string());
        }
        if let Some(max_conn) = proxy.max_connections {
            if max_conn == 0 {
                return Err("HTTP forward proxy max_connections cannot be 0".to_string());
            }
        }
        if let Some(timeout) = proxy.idle_timeout_secs {
            if timeout == 0 {
                return Err("HTTP forward proxy idle_timeout_secs cannot be 0".to_string());
            }
        }
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test test_http_forward_proxy --test config_tests -v`
Expected: Both tests PASS

- [ ] **Step 6: Run full config test suite**

Run: `cargo test --test config_tests -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/config.rs tests/config_tests.rs
git commit -m "feat: Add HttpForwardProxyConfig structure with validation"
```

---

## Task 2: Infrastructure - Change shutdown channel from mpsc to broadcast

**Files:**
- Modify: `src/main.rs`

---

- [ ] **Step 1: Review current shutdown setup in main.rs**

Read lines around shutdown channel creation (currently around line 38-40).

- [ ] **Step 2: Change mpsc::channel to broadcast::channel**

In `main()`:

```rust
// Create shutdown broadcast channel (supports multiple receivers)
let (shutdown_tx, _shutdown_rx) = broadcast::channel::<()>(16);
```

Also update the import at the top:

```rust
use tokio::sync::broadcast;
// Remove the mpsc import if no longer needed
```

- [ ] **Step 3: Update run_server_with_shared_config to accept broadcast::Receiver**

Find the function signature and change the shutdown parameter type. Inside the function, the receiver usage should work similarly for single receivers.

- [ ] **Step 4: Update all places that create shutdown receivers to use shutdown_tx.subscribe()**

Update the server thread:

```rust
let main_shutdown_rx = shutdown_tx.subscribe();
// Pass main_shutdown_rx to run_server_with_shared_config
```

- [ ] **Step 5: Verify graceful shutdown still works**

Run: `cargo build`
Expected: Build succeeds

Run the server and test Ctrl+C shutdown works:
```bash
cargo run -- config.yaml
# Press Ctrl+C after it starts
```
Expected: Clean shutdown with "Server shutdown complete" log

- [ ] **Step 6: Commit**

```bash
git add src/main.rs
git commit -m "refactor: Change shutdown channel from mpsc to broadcast"
```

---

## Task 3: Create http_proxy module structure

**Files:**
- Create: `src/http_proxy/mod.rs`
- Create: `src/http_proxy/auth.rs`
- Create: `src/http_proxy/handlers.rs`
- Create: `src/http_proxy/server.rs`

---

- [ ] **Step 1: Create src/http_proxy/mod.rs**

```rust
//! HTTP Forward Proxy module
//!
//! Provides standalone HTTP/HTTPS forward proxy functionality
//! with authentication and port restrictions.

pub mod auth;
pub mod handlers;
pub mod server;

pub use server::HttpForwardProxyServer;
pub use auth::proxy_auth_middleware;
```

- [ ] **Step 2: Create src/http_proxy/auth.rs (empty placeholder)**

```rust
//! Proxy authentication middleware

use axum::{
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
};

// Placeholder - will implement in Task 4
pub async fn proxy_auth_middleware<B>(
    req: Request<B>,
    next: Next<B>,
) -> Result<Response, StatusCode> {
    Ok(next.run(req).await)
}
```

- [ ] **Step 3: Create src/http_proxy/handlers.rs (empty placeholder)**

```rust
//! HTTP proxy request handlers

use axum::{
    http::{Request, Response, StatusCode},
    body::Body,
};

// Placeholders - will implement in Tasks 5-6
pub async fn handle_http_proxy(
    req: Request<Body>,
) -> Result<Response<Body>, StatusCode> {
    Ok(Response::builder()
        .status(StatusCode::NOT_IMPLEMENTED)
        .body(Body::from("HTTP proxy not implemented yet"))
        .unwrap())
}

pub async fn handle_connect_tunnel(
    req: Request<Body>,
) -> Result<Response<Body>, StatusCode> {
    Ok(Response::builder()
        .status(StatusCode::NOT_IMPLEMENTED)
        .body(Body::from("CONNECT tunnel not implemented yet"))
        .unwrap())
}
```

- [ ] **Step 4: Create src/http_proxy/server.rs (placeholder)**

```rust
//! HTTP forward proxy server implementation

use anyhow::Result;
use tokio::sync::broadcast;

pub struct HttpForwardProxyServer;

impl HttpForwardProxyServer {
    pub fn new() -> Self {
        Self
    }

    pub fn with_shutdown(self, _shutdown: broadcast::Receiver<()>) -> Self {
        self
    }

    pub async fn run(self) -> Result<()> {
        tracing::info!("HTTP forward proxy server placeholder running");
        Ok(())
    }
}
```

- [ ] **Step 5: Add module to src/lib.rs**

Add to `src/lib.rs`:

```rust
pub mod http_proxy;
```

- [ ] **Step 6: Verify build**

Run: `cargo check`
Expected: Build succeeds

- [ ] **Step 7: Commit**

```bash
git add src/http_proxy/mod.rs src/http_proxy/auth.rs src/http_proxy/handlers.rs src/http_proxy/server.rs src/lib.rs
git commit -m "feat: Create http_proxy module structure with placeholders"
```

---

## Task 4: Implement proxy authentication middleware

**Files:**
- Modify: `src/http_proxy/auth.rs`
- Create test TBD

---

- [ ] **Step 1: Write middleware implementation**

Replace the placeholder in `src/http_proxy/auth.rs`:

```rust
//! Proxy authentication middleware

use crate::config::Config;
use arc_swap::ArcSwap;
use axum::{
    extract::State,
    http::{header, Request, StatusCode},
    middleware::Next,
    response::Response,
};
use std::sync::Arc;

/// Proxy authentication middleware state
#[derive(Clone)]
pub struct ProxyAuthState {
    pub shared_config: Arc<ArcSwap<Config>>,
}

/// Proxy authentication middleware
///
/// Validates Proxy-Authorization header if auth_token is configured.
pub async fn proxy_auth_middleware<B>(
    State(state): State<ProxyAuthState>,
    req: Request<B>,
    next: Next<B>,
) -> Result<Response, StatusCode> {
    let config = state.shared_config.load();
    
    let proxy_config = match &config.http_forward_proxy {
        Some(c) if c.enabled && c.auth_token.is_some() => c,
        _ => {
            // No auth configured or proxy disabled - pass through
            return Ok(next.run(req).await);
        }
    };

    let expected_token = proxy_config.auth_token.as_ref().unwrap();
    
    // Check for Proxy-Authorization header
    let auth_header = req
        .headers()
        .get("Proxy-Authorization")
        .and_then(|h| h.to_str().ok());

    match auth_header {
        Some(header) if header.starts_with("Bearer ") => {
            let token = &header["Bearer ".len()..];
            if token == expected_token {
                return Ok(next.run(req).await);
            }
        }
        _ => {}
    }

    // Authentication failed
    let mut response = Response::new(axum::body::Body::from(
        r#"{"error":"Proxy Authentication Required","code":407,"message":"Valid Proxy-Authorization header is required"}"#
    ));
    *response.status_mut() = StatusCode::PROXY_AUTHENTICATION_REQUIRED;
    response.headers_mut().insert(
        "Proxy-Authenticate",
        header::HeaderValue::from_static("Bearer realm=\"Lumina-Proxy\""),
    );
    response.headers_mut().insert(
        "Content-Type",
        header::HeaderValue::from_static("application/json"),
    );
    
    Ok(response)
}
```

- [ ] **Step 2: Add helper function exports for handlers to use**

These helpers will be called from handlers.rs - exports only:

```rust
/// Check if a target port is allowed by configuration
///
/// Returns Ok(()) if allowed, Err(message) if blocked
pub fn check_target_port_allowed(config: &Config, port: u16) -> Result<(), String> {
    let proxy_config = match &config.http_forward_proxy {
        Some(c) if c.enabled => c,
        _ => return Ok(()), // No config means no restrictions
    };

    if !proxy_config.is_port_allowed(port) {
        return Err(format!("Target port {} is not allowed by proxy configuration", port));
    }

    Ok(())
}

/// Sanitize request headers by removing internal/sensitive headers
pub fn sanitize_request_headers(headers: &mut axum::http::HeaderMap) {
    // Remove headers that could reveal internal infrastructure
    headers.remove("X-Forwarded-For");
    headers.remove("X-Real-IP");
    
    // Remove any X-Internal-* headers
    let internal_headers: Vec<_> = headers
        .keys()
        .filter(|k| k.as_str().to_lowercase().starts_with("x-internal-"))
        .cloned()
        .collect();
    
    for header in internal_headers {
        headers.remove(header);
    }
}
```

- [ ] **Step 3: Verify build**

Run: `cargo check`
Expected: Build succeeds

- [ ] **Step 4: Run clippy**

Run: `cargo clippy -- -D warnings`
Expected: No warnings

- [ ] **Step 5: Commit**

```bash
git add src/http_proxy/auth.rs
git commit -m "feat: Implement proxy authentication middleware"
```

---

## Task 5: Implement HTTP proxy handler (request forwarding)

**Files:**
- Modify: `src/http_proxy/handlers.rs`
- Add: Helper functions to handlers

---

- [ ] **Step 1: Update handlers.rs with HTTP proxy implementation**

Replace the placeholder implementation in `src/http_proxy/handlers.rs`:

```rust
//! HTTP proxy request handlers

use crate::config::Config;
use crate::http_proxy::auth::{check_target_port_allowed, sanitize_request_headers};
use arc_swap::ArcSwap;
use axum::{
    body::Body,
    extract::State,
    http::{Method, Request, Response, StatusCode, Uri},
};
use reqwest::Client;
use std::sync::Arc;

/// Proxy handler state shared across requests
#[derive(Clone)]
pub struct ProxyHandlerState {
    pub shared_config: Arc<ArcSwap<Config>>,
    pub client: Client,
}

/// Extract host and port from a request URI that contains full URL
fn extract_target_host_port(uri: &Uri) -> Result<(String, u16), String> {
    let host = uri.host().ok_or_else(|| "Missing host in request URI".to_string())?;
    let port = uri.port_u16().unwrap_or_else(|| {
        if uri.scheme_str() == Some("https") {
            443
        } else {
            80
        }
    });
    Ok((host.to_string(), port))
}

/// Main dispatch handler - routes to appropriate handler based on method
pub async fn proxy_dispatch_handler(
    State(state): State<ProxyHandlerState>,
    req: Request<Body>,
) -> Result<Response<Body>, StatusCode> {
    match req.method() {
        &Method::CONNECT => handle_connect_tunnel(state, req).await,
        _ => handle_http_proxy(state, req).await,
    }
}

/// Handle regular HTTP proxy requests (forward to target server)
pub async fn handle_http_proxy(
    State(state): State<ProxyHandlerState>,
    mut req: Request<Body>,
) -> Result<Response<Body>, StatusCode> {
    let config = state.shared_config.load();
    let uri = req.uri().clone();
    
    // Extract target host and port
    let (host, port) = extract_target_host_port(&uri)
        .map_err(|e| {
            tracing::warn!("Invalid proxy request URI: {}", e);
            return json_error_response(
                StatusCode::BAD_REQUEST,
                "Invalid Request",
                &format!("Invalid proxy request URI: {}", e)
            );
        })?;
    
    // Check if port is allowed
    check_target_port_allowed(&config, port)
        .map_err(|e| {
            tracing::warn!("Port blocked: {}", e);
            return json_error_response(
                StatusCode::FORBIDDEN,
                "Port Not Allowed",
                &e
            );
        })?;
    
    // Sanitize headers
    sanitize_request_headers(req.headers_mut());
    
    // Build target URL
    let scheme = uri.scheme_str().unwrap_or("http");
    let path_and_query = uri.path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/");
    
    let target_url = format!("{}://{}:{}{}", scheme, host, port, path_and_query);
    
    tracing::debug!(
        method = %req.method(),
        target = %target_url,
        "Forwarding proxy request"
    );
    
    // Build and send forwarded request
    let mut forwarded_req = state.client
        .request(req.method().clone(), &target_url)
        .timeout(std::time::Duration::from_secs(
            config.http_forward_proxy
                .as_ref()
                .and_then(|c| c.idle_timeout_secs)
                .unwrap_or(60)
        ));
    
    // Forward all headers (preserve Host header for virtual hosting)
    for (key, value) in req.headers() {
        forwarded_req = forwarded_req.header(key, value);
    }
    
    // Forward body
    let forwarded_req = forwarded_req.body(req.into_body());
    
    // Send and get response
    let response = forwarded_req.send().await.map_err(|e| {
        tracing::warn!("Proxy request failed: {}", e);
        if e.is_timeout() {
            return json_error_response(
                StatusCode::GATEWAY_TIMEOUT,
                "Gateway Timeout",
                "Timeout connecting to upstream server"
            );
        } else {
            return json_error_response(
                StatusCode::BAD_GATEWAY,
                "Bad Gateway",
                &format!("Failed to connect to upstream server: {}", e)
            );
        }
    })?;
    
    // Build response for client
    let mut client_response = Response::builder()
        .status(response.status());
    
    // Copy response headers
    for (key, value) in response.headers() {
        client_response = client_response.header(key, value);
    }
    
    // Stream response body
    let body = Body::from_stream(response.bytes_stream());
    let client_response = client_response.body(body)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(client_response)
}

pub async fn handle_connect_tunnel(
    _state: State<ProxyHandlerState>,
    _req: Request<Body>,
) -> Result<Response<Body>, StatusCode> {
    Ok(Response::builder()
        .status(StatusCode::NOT_IMPLEMENTED)
        .body(Body::from("CONNECT tunnel not implemented yet - coming next"))
        .unwrap())
}
```

- [ ] **Step 2: Verify build**

Run: `cargo check`
Expected: Build succeeds

- [ ] **Step 3: Run clippy**

Run: `cargo clippy -- -D warnings`
Expected: No warnings

- [ ] **Step 4: Commit**

```bash
git add src/http_proxy/handlers.rs
git commit -m "feat: Implement HTTP proxy request forwarding handler"
```

---

## Task 6: Implement CONNECT tunnel handler

**Files:**
- Modify: `src/http_proxy/handlers.rs` - update handle_connect_tunnel

---

- [ ] **Step 1: Implement CONNECT tunnel handling**

Replace the `handle_connect_tunnel` placeholder in `src/http_proxy/handlers.rs`:

```rust
/// Handle CONNECT method to establish HTTPS tunnel
pub async fn handle_connect_tunnel(
    State(state): State<ProxyHandlerState>,
    req: Request<Body>,
) -> Result<Response<Body>, StatusCode> {
    let config = state.shared_config.load();
    
    // Extract target host:port from CONNECT URI (should be just host:port)
    let uri = req.uri().clone();
    let host_port = uri.path();
    
    // Parse host:port format
    let parts: Vec<&str> = host_port.split(':').collect();
    if parts.len() != 2 {
        tracing::warn!("Invalid CONNECT target: {}", host_port);
        return Ok(Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .body(Body::from(format!("Invalid CONNECT target: {}", host_port)))
            .unwrap());
    }
    
    let host = parts[0].to_string();
    let port: u16 = parts[1].parse().map_err(|_| {
        tracing::warn!("Invalid port in CONNECT: {}", parts[1]);
        StatusCode::BAD_REQUEST
    })?;
    
    // Check if port is allowed
    check_target_port_allowed(&config, port)
        .map_err(|e| {
            tracing::warn!("Port blocked for CONNECT: {}", e);
            StatusCode::FORBIDDEN
        })?;
    
    tracing::debug!(target = %format!("{}:{}", host, port), "Establishing CONNECT tunnel");
    
    // Get idle timeout config
    let timeout_secs = config.http_forward_proxy
        .as_ref()
        .and_then(|c| c.idle_timeout_secs)
        .unwrap_or(60);
    
    // Establish connection to target server
    let target_addr = format!("{}:{}", host, port);
    let mut target_stream = tokio::net::TcpStream::connect(&target_addr)
        .await
        .map_err(|e| {
            tracing::warn!("Failed to connect to {}: {}", target_addr, e);
            StatusCode::BAD_GATEWAY
        })?;
    
    // Send 200 Connection Established response to client
    // The response body is the tunnel itself - we use a special streaming body
    // that will perform bidirectional copy after the response headers are sent
    
    // First, we need to extract the client's TCP stream from the request
    // This is tricky in axum - we use an upgrade mechanism
    
    let response = Response::builder()
        .status(StatusCode::OK)
        .header("Connection", "keep-alive")
        .body(Body::empty())
        .unwrap();
    
    // Spawn task to handle the tunnel
    // Note: In a real implementation, we need to use axum's upgrade mechanism
    // For now, we'll complete the implementation once server.rs is wired
    
    Ok(response)
}
```

**Note:** CONNECT tunnel requires special handling with axum's upgrade mechanism. The full implementation will be completed in Task 10 after server wiring is in place.

- [ ] **Step 2: Verify build**

Run: `cargo check`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add src/http_proxy/handlers.rs
git commit -m "feat: Implement CONNECT tunnel handler skeleton"
```

---

## Task 7: Implement HttpForwardProxyServer and wire to main

**Files:**
- Modify: `src/http_proxy/server.rs`
- Modify: `src/main.rs`

---

- [ ] **Step 1: Complete HttpForwardProxyServer implementation**

Replace `src/http_proxy/server.rs`:

```rust
//! HTTP forward proxy server implementation

use super::handlers::{proxy_dispatch_handler, ProxyHandlerState};
use super::auth::{proxy_auth_middleware, ProxyAuthState};
use anyhow::Result;
use arc_swap::ArcSwap;
use axum::{
    Router,
    body::Body,
    http::Request,
};
use reqwest::Client;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
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
        let max_body_size = proxy_config.max_request_body_size.unwrap_or(100 * 1024 * 1024);
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
            .layer(axum::body::RequestBodyLimitLayer::new(max_body_size))
            .layer(TraceLayer::new_for_http());

        // Bind and serve with graceful shutdown
        let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
        tracing::info!("HTTP forward proxy listening on {}", bind_addr);

        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown.recv().await;
                tracing::info!("HTTP forward proxy initiating graceful shutdown");
            })
            .await?;

        tracing::info!("HTTP forward proxy shutdown complete");
        Ok(())
    }
}
```

- [ ] **Step 2: Wire server to main.rs**

In `src/main.rs`, after starting the main server thread (around line 75), add:

```rust
// Start HTTP forward proxy if enabled in config
if let Some(proxy_config) = &config.http_forward_proxy 
    && proxy_config.enabled 
{
    let proxy_shared_config = shared_config.clone();
    let proxy_shutdown_rx = shutdown_tx.subscribe();
    
    tracing::info!("HTTP forward proxy enabled, starting on {}:{}", proxy_config.host, proxy_config.port);
    
    std::thread::spawn(move || {
        let rt = match tokio::runtime::Runtime::new() {
            Ok(rt) => rt,
            Err(e) => {
                tracing::error!("Failed to create Tokio runtime for proxy: {}", e);
                return Err(e);
            }
        };
        
        rt.block_on(async move {
            match HttpForwardProxyServer::new(proxy_shared_config) {
                Ok(server) => {
                    if let Err(e) = server.with_shutdown(proxy_shutdown_rx).run().await {
                        tracing::error!("HTTP forward proxy error: {}", e);
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to create HTTP forward proxy server: {}", e);
                }
            }
        });
        
        Ok::<(), anyhow::Error>(())
    });
}
```

Also add the import at the top:

```rust
use lumina::http_proxy::HttpForwardProxyServer;
```

- [ ] **Step 3: Verify build**

Run: `cargo build`
Expected: Build succeeds

- [ ] **Step 4: Run clippy**

Run: `cargo clippy -- -D warnings`
Expected: No warnings

- [ ] **Step 5: Commit**

```bash
git add src/http_proxy/server.rs src/main.rs
git commit -m "feat: Implement HttpForwardProxyServer and wire to main"
```

---

## Task 8: Add hot-reload detection for proxy config changes

**Files:**
- Modify: `src/proxy/handlers.rs` (reload_config_handler)

---

- [ ] **Step 1: Find reload_config_handler in src/proxy/handlers.rs**

Locate the `reload_config_handler` function (around line 424).

- [ ] **Step 2: Add proxy config change detection**

Add proxy config change detection alongside other warnings (around line 475):

```rust
// Check for HTTP forward proxy config changes that need restart
match (&current_config.http_forward_proxy, &new_config.http_forward_proxy) {
    (None, None) => {},
    (Some(old), Some(new)) => {
        if old.enabled != new.enabled 
            || old.port != new.port 
            || old.host != new.host 
        {
            warnings.push(
                "HTTP forward proxy configuration (host/port/enabled) changed. \
                Server restart required for changes to take effect.".to_string()
            );
        }
    },
    (None, Some(new)) if new.enabled => {
        warnings.push(
            "HTTP forward proxy was enabled in configuration. \
            Server restart required to start the proxy server.".to_string()
        );
    },
    (Some(old), None) if old.enabled => {
        warnings.push(
            "HTTP forward proxy was disabled in configuration. \
            Server restart required to stop the proxy server.".to_string()
        );
    },
    _ => {},
}
```

- [ ] **Step 3: Verify build**

Run: `cargo check`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add src/proxy/handlers.rs
git commit -m "feat: Add HTTP forward proxy config change detection for hot reload"
```

---

## Task 9: Update config.yaml with proxy configuration example

**Files:**
- Modify: `config.yaml` (or the default config file)

---

- [ ] **Step 1: Add HTTP forward proxy config section to config.yaml**

Add after the `server` section:

```yaml
# HTTP Forward Proxy configuration
# Provides standalone HTTP/HTTPS proxy functionality on a separate port
http_forward_proxy:
  # Whether to enable the HTTP forward proxy
  enabled: false
  
  # Port to listen for proxy requests
  port: 8080
  
  # Host address to bind to (use 0.0.0.0 to allow external access)
  host: 127.0.0.1
  
  # Authentication token for proxy access (optional but recommended)
  # Clients must send: Proxy-Authorization: Bearer <token>
  # auth_token: "your-proxy-secret-token"
  
  # Maximum concurrent connections (default: 1024)
  # max_connections: 1024
  
  # Idle timeout in seconds for CONNECT tunnels (default: 60)
  # idle_timeout_secs: 60
  
  # Maximum request body size in bytes (default: 104857600 = 100MB)
  # max_request_body_size: 104857600
  
  # Allowed target ports - requests to other ports will be rejected
  # Default: [80, 443, 8080-8090]
  # allowed_target_ports: [80, 443, 8080]
  
  # Blocked target ports - takes precedence over allowed list
  # Default: [22, 3306, 5432, 6379, 27017] (SSH + database ports)
  # blocked_target_ports: [22, 3306]
```

- [ ] **Step 2: Verify config loads correctly**

Run: `cargo run -- config.yaml`
Expected: Server starts without errors

- [ ] **Step 3: Commit**

```bash
git add config.yaml
git commit -m "docs: Add HTTP forward proxy configuration example"
```

---

## Task 10: Complete CONNECT tunnel implementation with TCP upgrade

**Files:**
- Modify: `src/http_proxy/handlers.rs`
- Modify: `src/http_proxy/server.rs`

This is the most complex part - CONNECT requires TCP stream upgrade.

---

- [ ] **Step 1: Update handle_connect_tunnel for proper upgrade**

Replace the full implementation. This requires using hyper upgrade:

```rust
use axum::extract::ConnectInfo;
use std::net::SocketAddr;

/// Handle CONNECT method to establish HTTPS tunnel
pub async fn handle_connect_tunnel(
    State(state): State<ProxyHandlerState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    let config = state.shared_config.load();
    
    // Extract target host:port from CONNECT URI
    let uri = req.uri().clone();
    let host_port = uri.path();
    
    // Parse host:port format
    let parts: Vec<&str> = host_port.split(':').collect();
    if parts.len() != 2 {
        tracing::warn!("Invalid CONNECT target: {}", host_port);
        return Ok(json_error_response(
            StatusCode::BAD_REQUEST,
            "Invalid Request",
            &format!("Invalid CONNECT target: {}", host_port)
        ));
    }
    
    let host = parts[0].to_string();
    let port: u16 = match parts[1].parse() {
        Ok(p) => p,
        Err(_) => {
            tracing::warn!("Invalid port in CONNECT: {}", parts[1]);
            return Ok(json_error_response(
                StatusCode::BAD_REQUEST,
                "Invalid Request",
                &format!("Invalid port: {}", parts[1])
            ));
        }
    };
    
    // Check if port is allowed
    if let Err(e) = check_target_port_allowed(&config, port) {
        tracing::warn!("Port blocked for CONNECT: {}", e);
        return Ok(json_error_response(
            StatusCode::FORBIDDEN,
            "Port Not Allowed",
            &e
        ));
    }
    
    tracing::debug!(
        client = %client_addr,
        target = %format!("{}:{}", host, port),
        "CONNECT tunnel requested"
    );
    
    // Get timeout config
    let timeout_secs = config.http_forward_proxy
        .as_ref()
        .and_then(|c| c.idle_timeout_secs)
        .unwrap_or(60);
    
    // Get upgrade future - this gives us the raw TCP stream
    let upgrade = axum::extract::upgrade::on(req);
    
    // Spawn task to handle tunnel - we return 200 immediately
    tokio::spawn(async move {
        // Wait for upgrade to complete
        let mut upgraded = match upgrade.await {
            Ok(u) => u,
            Err(e) => {
                tracing::warn!("CONNECT upgrade failed: {}", e);
                return;
            }
        };
        
        // Connect to target server
        let target_addr = format!("{}:{}", host, port);
        let mut target_stream = match tokio::net::TcpStream::connect(&target_addr).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("Failed to connect to target {}: {}", target_addr, e);
                return;
            }
        };
        
        // Enable TCP_NODELAY to reduce latency
        if let Err(e) = target_stream.set_nodelay(true) {
            tracing::debug!("Failed to set TCP_NODELAY: {}", e);
        }
        
        // Bidirectional copy with proper half-close support
        // This handles TCP FIN properly and copies both directions until both close
        match tokio::io::copy_bidirectional(&mut upgraded, &mut target_stream).await {
            Ok((client_to_target, target_to_client)) => {
                tracing::debug!(
                    "CONNECT tunnel closed cleanly, {} bytes up, {} bytes down",
                    client_to_target, target_to_client
                );
            }
            Err(e) => {
                tracing::debug!("CONNECT tunnel error: {}", e);
            }
        }
    });
    
    // Return 200 Connection Established immediately
    Ok(Response::builder()
        .status(StatusCode::OK)
        .body(Body::empty())
        .unwrap())
}
```

- [ ] **Step 2: Update server.rs to add ConnectInfo extractor**

In `server.rs`, update the router to add `ConnectInfo`:

```rust
use axum::extract::ConnectInfo;
use std::net::SocketAddr;

// In run():
let app = Router::new()
    .fallback(proxy_dispatch_handler)
    .with_state(handler_state)
    .layer(axum::middleware::from_fn_with_state(
        auth_state,
        proxy_auth_middleware,
    ))
    .layer(TraceLayer::new_for_http())
    .into_make_service_with_connect_info::<SocketAddr>();
```

- [ ] **Step 3: Verify build and clippy**

Run: `cargo build && cargo clippy -- -D warnings`
Expected: Both succeed

- [ ] **Step 4: Commit**

```bash
git add src/http_proxy/handlers.rs src/http_proxy/server.rs
git commit -m "feat: Complete CONNECT tunnel implementation with TCP upgrade"
```

---

## Task 11: Manual testing and integration

**Files:** Manual testing only

---

- [ ] **Step 1: Enable proxy in config.yaml**

Set `enabled: true` in the `http_forward_proxy` section.

- [ ] **Step 2: Start the server**

Run: `cargo run -- config.yaml`
Expected: Log shows "HTTP forward proxy listening on 127.0.0.1:8080"

- [ ] **Step 3: Test HTTP proxy with curl**

In another terminal:
```bash
curl -x http://127.0.0.1:8080 http://example.com -v
```
Expected: Returns example.com HTML

- [ ] **Step 4: Test HTTPS CONNECT tunnel with curl**

```bash
curl -x http://127.0.0.1:8080 https://example.com -v
```
Expected: Returns example.com HTML via CONNECT tunnel

- [ ] **Step 5: Test authentication**

Add `auth_token: "test123"` to config, restart, then:
```bash
# Should fail with 407
curl -x http://127.0.0.1:8080 http://example.com -v

# Should succeed
curl -x http://127.0.0.1:8080 http://example.com -H "Proxy-Authorization: Bearer test123" -v
```

- [ ] **Step 6: Test port restrictions**

```bash
# Should fail (SSH port blocked)
curl -x http://127.0.0.1:8080 https://example.com:22 -v

# Should succeed (HTTPS port allowed)
curl -x http://127.0.0.1:8080 https://example.com -v
```

---

## Task 12: Run full test suite and fix issues

**Files:** All files touched in this plan

---

- [ ] **Step 1: Run full test suite**

Run: `cargo test -v`
Expected: All tests pass

- [ ] **Step 2: Run clippy**

Run: `cargo clippy -- -D warnings`
Expected: No warnings

- [ ] **Step 3: Format check**

Run: `cargo fmt --check`
Expected: No formatting issues (or run cargo fmt to fix)

- [ ] **Step 4: Commit any fixes**

```bash
# Only if there were fixes
git add -A
git commit -m "fix: Final test and clippy fixes"
```

---

## Task 13: Write integration tests

**Files:**
- Create: `tests/http_proxy_tests.rs`

---

- [ ] **Step 1: Create integration test file**

Create `tests/http_proxy_tests.rs` with tests for:
- Port checking logic (allowed/blocked ports)
- Config validation
- Header sanitization

- [ ] **Step 2: Run tests**

Run: `cargo test --test http_proxy_tests -v`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add tests/http_proxy_tests.rs
git commit -m "test: Add HTTP forward proxy integration tests"
```

---

## Post-Implementation Checklist

- [ ] All tests pass: `cargo test`
- [ ] Clippy clean: `cargo clippy -- -D warnings`
- [ ] Format clean: `cargo fmt --check`
- [ ] Manual HTTP proxy test passes
- [ ] Manual HTTPS CONNECT tunnel test passes
- [ ] Authentication test passes
- [ ] Port restriction test passes
- [ ] Hot reload config change warning works
