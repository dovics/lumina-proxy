//! Backend URL construction utilities

use crate::config::{ProviderType, RouteConfig};
use crate::types::ProxyError;

// =============================================================================
// Backend URL Construction
// =============================================================================

/// Build the appropriate backend URL based on provider type and route config
pub fn build_backend_url(route: &RouteConfig, model: &str) -> Result<String, ProxyError> {
    build_backend_url_for_endpoint(route, model, false)
}

/// Build backend URL with option to use Responses API endpoint
pub fn build_backend_url_for_endpoint(
    route: &RouteConfig,
    model: &str,
    use_responses_api: bool,
) -> Result<String, ProxyError> {
    match route.provider_type {
        ProviderType::Ollama => {
            let base_url = route.base_url.as_ref().ok_or_else(|| {
                ProxyError::ConfigError(format!(
                    "Ollama route for model '{}' missing base_url",
                    model
                ))
            })?;
            Ok(format!("{}/api/chat", base_url.trim_end_matches('/')))
        }

        ProviderType::Anthropic => {
            let base_url = route.base_url.as_ref().ok_or_else(|| {
                ProxyError::ConfigError(format!(
                    "Anthropic route for model '{}' missing base_url",
                    model
                ))
            })?;
            Ok(format!("{}/v1/messages", base_url.trim_end_matches('/')))
        }

        ProviderType::Gemini => {
            let base_url = route.base_url.as_ref().ok_or_else(|| {
                ProxyError::ConfigError(format!(
                    "Gemini route for model '{}' missing base_url",
                    model
                ))
            })?;
            Ok(format!(
                "{}/v1beta/models/{}:streamGenerateContent?alt=sse",
                base_url.trim_end_matches('/'),
                route.upstream_model()
            ))
        }

        ProviderType::OpenAi => {
            if let Some(url) = &route.url {
                Ok(url.clone())
            } else if let Some(base_url) = &route.base_url {
                let endpoint = if use_responses_api {
                    "/v1/responses"
                } else {
                    "/v1/chat/completions"
                };
                Ok(format!("{}{}", base_url.trim_end_matches('/'), endpoint))
            } else {
                Err(ProxyError::ConfigError(format!(
                    "OpenAI route for model '{}' missing either url or base_url",
                    model
                )))
            }
        }

        ProviderType::OpenAiCompatible => {
            if use_responses_api {
                if let Some(base_url) = &route.base_url {
                    Ok(format!("{}/v1/responses", base_url.trim_end_matches('/')))
                } else if let Some(url) = &route.url {
                    Ok(url.clone())
                } else {
                    Err(ProxyError::ConfigError(format!(
                        "OpenAI-compatible route for model '{}' missing base_url or url",
                        model
                    )))
                }
            } else {
                if let Some(url) = &route.url {
                    Ok(url.clone())
                } else if let Some(base_url) = &route.base_url {
                    Ok(format!(
                        "{}/v1/chat/completions",
                        base_url.trim_end_matches('/')
                    ))
                } else {
                    Err(ProxyError::ConfigError(format!(
                        "OpenAI-compatible route for model '{}' missing url or base_url",
                        model
                    )))
                }
            }
        }

        ProviderType::Moonlight => {
            if let Some(url) = &route.url {
                Ok(url.clone())
            } else if let Some(base_url) = &route.base_url {
                let endpoint = if use_responses_api {
                    "/v1/responses"
                } else {
                    "/v1/chat/completions"
                };
                Ok(format!("{}{}", base_url.trim_end_matches('/'), endpoint))
            } else {
                Err(ProxyError::ConfigError(format!(
                    "Moonlight route for model '{}' missing either url or base_url",
                    model
                )))
            }
        }
    }
}
