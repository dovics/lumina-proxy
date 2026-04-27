use crate::proxy::ProxyState;
use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
};
use std::sync::Arc;

/// Authentication middleware that validates Bearer tokens
/// If no auth token is configured in the config, all requests are allowed
pub async fn auth_middleware(
    State(state): State<Arc<ProxyState>>,
    req: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    // If no auth token is configured, allow all requests
    let config = state.config.load();
    let Some(expected_token) = &config.server.auth_token else {
        return Ok(next.run(req).await);
    };

    // Extract the Authorization header
    let auth_header = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .ok_or(StatusCode::UNAUTHORIZED)?;

    // Parse as a Bearer token
    let auth_str = auth_header.to_str().map_err(|_| StatusCode::UNAUTHORIZED)?;

    if !auth_str.starts_with("Bearer ") {
        return Err(StatusCode::UNAUTHORIZED);
    }

    let token = &auth_str["Bearer ".len()..];

    // Compare the token with the expected token
    if token == expected_token {
        Ok(next.run(req).await)
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}
