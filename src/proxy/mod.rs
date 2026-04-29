//! Core proxy implementation - handles request routing, conversion, and streaming
//!
//! This module is organized into submodules:
//! - `state`: ProxyState shared application state
//! - `url`: Backend URL construction utilities
//! - `tools`: Tool call parsing and aggregation utilities
//! - `non_streaming`: Non-streaming request handling
//! - `streaming`: Streaming request handling
//! - `responses_convert`: Responses API conversion utilities
//! - `handlers`: HTTP handlers for proxy endpoints

pub mod handlers;
pub mod non_streaming;
pub mod responses_convert;
pub mod state;
pub mod streaming;
pub mod tools;
pub mod url;

// Re-export commonly used items
pub use handlers::{
    get_config_handler, models_handler, proxy_handler, reload_config_handler, responses_handler,
};
pub use non_streaming::handle_non_streaming;
pub use state::ProxyState;
pub use streaming::handle_streaming;
pub use tools::aggregate_tool_calls;
pub use url::{build_backend_url, build_backend_url_for_endpoint};
