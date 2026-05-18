//! HTTP proxy request handlers

use crate::config::Config;
use crate::http_proxy::auth::{check_target_port_allowed, sanitize_request_headers};
use arc_swap::ArcSwap;
use axum::{
    body::Body,
    extract::{ConnectInfo, State},
    http::{Method, Request, Response, StatusCode, Uri},
};
use hyper::upgrade;
use hyper_util::rt::TokioIo;
use reqwest::Client;
use std::net::SocketAddr;
use std::sync::Arc;

/// Proxy handler state shared across requests
#[derive(Clone)]
pub struct ProxyHandlerState {
    pub shared_config: Arc<ArcSwap<Config>>,
    pub client: Client,
}

/// Helper: Build JSON error response with consistent format
fn json_error_response(status: StatusCode, error: &str, message: &str) -> Response<Body> {
    let body = serde_json::json!({
        "error": error,
        "code": status.as_u16(),
        "message": message
    });
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// Extract host and port from a request URI that contains full URL
fn extract_target_host_port(uri: &Uri) -> Result<(String, u16), String> {
    let host = uri
        .host()
        .ok_or_else(|| "Missing host in request URI".to_string())?;
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
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    match req.method() {
        &Method::CONNECT => {
            handle_connect_tunnel(State(state), ConnectInfo(client_addr), req).await
        }
        _ => handle_http_proxy(State(state), req).await,
    }
}

/// Handle regular HTTP proxy requests (forward to target server)
pub async fn handle_http_proxy(
    State(state): State<ProxyHandlerState>,
    mut req: Request<Body>,
) -> Response<Body> {
    let config = state.shared_config.load();
    let uri = req.uri().clone();

    // Extract target host and port
    let (host, port) = match extract_target_host_port(&uri) {
        Ok(hp) => hp,
        Err(e) => {
            tracing::warn!("Invalid proxy request URI: {}", e);
            return json_error_response(
                StatusCode::BAD_REQUEST,
                "Invalid Request",
                &format!("Invalid proxy request URI: {}", e),
            );
        }
    };

    // Check if port is allowed
    if let Err(e) = check_target_port_allowed(&config, port) {
        tracing::warn!("Port blocked: {}", e);
        return json_error_response(StatusCode::FORBIDDEN, "Port Not Allowed", &e);
    }

    // Sanitize headers
    sanitize_request_headers(req.headers_mut());

    // Build target URL
    let scheme = uri.scheme_str().unwrap_or("http");
    let path_and_query = uri.path_and_query().map(|pq| pq.as_str()).unwrap_or("/");

    let target_url = format!("{}://{}:{}{}", scheme, host, port, path_and_query);

    tracing::debug!(
        method = %req.method(),
        target = %target_url,
        "Forwarding proxy request"
    );

    // Build and send forwarded request
    let mut forwarded_req = state
        .client
        .request(req.method().clone(), &target_url)
        .timeout(std::time::Duration::from_secs(
            config
                .http_forward_proxy
                .as_ref()
                .and_then(|c| c.idle_timeout_secs)
                .unwrap_or(60),
        ));

    // Forward all headers (preserve Host header for virtual hosting)
    for (key, value) in req.headers() {
        forwarded_req = forwarded_req.header(key, value);
    }

    // Forward body (convert axum Body to reqwest Body via stream)
    let forwarded_req = forwarded_req.body(reqwest::Body::wrap_stream(
        req.into_body().into_data_stream(),
    ));

    // Send and get response
    let response = match forwarded_req.send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("Proxy request failed: {}", e);
            if e.is_timeout() {
                return json_error_response(
                    StatusCode::GATEWAY_TIMEOUT,
                    "Gateway Timeout",
                    "Timeout connecting to upstream server",
                );
            } else {
                return json_error_response(
                    StatusCode::BAD_GATEWAY,
                    "Bad Gateway",
                    &format!("Failed to connect to upstream server: {}", e),
                );
            }
        }
    };

    // Build response for client
    let mut client_response = Response::builder().status(response.status());

    // Copy response headers
    for (key, value) in response.headers() {
        client_response = client_response.header(key, value);
    }

    // Stream response body
    let body = Body::from_stream(response.bytes_stream());
    client_response.body(body).unwrap_or_else(|_| {
        json_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Internal Error",
            "Failed to build response body",
        )
    })
}

/// Handle CONNECT method to establish HTTPS tunnel
pub async fn handle_connect_tunnel(
    State(state): State<ProxyHandlerState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    let config = state.shared_config.load();

    // Extract target host:port from CONNECT URI
    // For CONNECT requests like "CONNECT ipconfig.io:443 HTTP/1.1", the target
    // is in the authority part, not the path. Use authority().unwrap().as_str()
    // to get the host:port format directly from the request URI.
    let uri = req.uri().clone();
    let host_port = uri
        .authority()
        .map(|a| a.as_str())
        .unwrap_or_default();

    // Parse host:port format
    let parts: Vec<&str> = host_port.split(':').collect();
    if parts.len() != 2 {
        tracing::warn!("Invalid CONNECT target: {}", host_port);
        return json_error_response(
            StatusCode::BAD_REQUEST,
            "Invalid Request",
            &format!("Invalid CONNECT target: {}", host_port),
        );
    }

    let host = parts[0].to_string();
    let port: u16 = match parts[1].parse() {
        Ok(p) => p,
        Err(_) => {
            tracing::warn!("Invalid port in CONNECT: {}", parts[1]);
            return json_error_response(
                StatusCode::BAD_REQUEST,
                "Invalid Request",
                &format!("Invalid port: {}", parts[1]),
            );
        }
    };

    // Check if port is allowed
    if let Err(e) = check_target_port_allowed(&config, port) {
        tracing::warn!("Port blocked for CONNECT: {}", e);
        return json_error_response(StatusCode::FORBIDDEN, "Port Not Allowed", &e);
    }

    tracing::debug!(
        client = %client_addr,
        target = %format!("{}:{}", host, port),
        "CONNECT tunnel requested"
    );

    // Get timeout config
    let _timeout_secs = config
        .http_forward_proxy
        .as_ref()
        .and_then(|c| c.idle_timeout_secs)
        .unwrap_or(60);

    // Get upgrade future - this gives us the raw TCP stream
    let upgrade = upgrade::on(req);

    // Spawn task to handle tunnel - we return 200 immediately
    tokio::spawn(async move {
        // Wait for upgrade to complete
        let upgraded = match upgrade.await {
            Ok(u) => u,
            Err(e) => {
                tracing::warn!("CONNECT upgrade failed: {}", e);
                return;
            }
        };

        // Convert hyper::Upgraded to tokio-compatible IO
        let mut upgraded = TokioIo::new(upgraded);

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
                    client_to_target,
                    target_to_client
                );
            }
            Err(e) => {
                tracing::debug!("CONNECT tunnel error: {}", e);
            }
        }
    });

    // Return 200 Connection Established immediately
    Response::builder()
        .status(StatusCode::OK)
        .body(Body::empty())
        .unwrap()
}
