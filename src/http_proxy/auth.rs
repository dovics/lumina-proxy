//! Proxy authentication middleware

use crate::config::Config;
use arc_swap::ArcSwap;
use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode, header},
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
pub async fn proxy_auth_middleware(
    State(state): State<ProxyAuthState>,
    req: Request<Body>,
    next: Next,
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
    let error_json = serde_json::json!({
        "error": "Proxy Authentication Required",
        "code": 407,
        "message": "Valid Proxy-Authorization header is required"
    });

    let mut response = Response::new(Body::from(error_json.to_string()));
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

/// Check if a target port is allowed by configuration
///
/// Returns Ok(()) if allowed, Err(message) if blocked
pub fn check_target_port_allowed(config: &Config, port: u16) -> Result<(), String> {
    let proxy_config = match &config.http_forward_proxy {
        Some(c) if c.enabled => c,
        _ => return Ok(()), // No config means no restrictions
    };

    if !proxy_config.is_port_allowed(port) {
        return Err(format!(
            "Target port {} is not allowed by proxy configuration",
            port
        ));
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
