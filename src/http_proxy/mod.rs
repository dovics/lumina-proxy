//! HTTP Forward Proxy module
//!
//! Provides standalone HTTP/HTTPS forward proxy functionality
//! with authentication and port restrictions.

pub mod auth;
pub mod handlers;
pub mod server;

pub use auth::proxy_auth_middleware;
pub use server::HttpForwardProxyServer;
