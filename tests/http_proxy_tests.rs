//! HTTP Forward Proxy integration tests

use lumina::config::*;

#[test]
fn test_proxy_config_default_ports() {
    // Test default allowed ports
    let allowed = HttpForwardProxyConfig::default_allowed_ports();
    assert!(allowed.contains(&80));
    assert!(allowed.contains(&443));
    assert!(allowed.contains(&8080));
}

#[test]
fn test_proxy_config_default_blocked() {
    // Test default blocked ports
    let blocked = HttpForwardProxyConfig::default_blocked_ports();
    assert!(blocked.contains(&22));    // SSH
    assert!(blocked.contains(&3306)); // MySQL
    assert!(blocked.contains(&6379)); // Redis
}

#[test]
fn test_proxy_is_port_allowed_with_defaults() {
    // Create a proxy config with None for allowed and blocked
    let yaml = r#"
enabled: true
port: 8080
host: 127.0.0.1
"#;
    let config: HttpForwardProxyConfig = serde_yaml::from_str(yaml).unwrap();

    // Default allowed ports should be used
    assert!(config.is_port_allowed(80));
    assert!(config.is_port_allowed(443));
    assert!(config.is_port_allowed(8080));

    // Default blocked ports should take precedence
    assert!(!config.is_port_allowed(22));
    assert!(!config.is_port_allowed(3306));
}

#[test]
fn test_proxy_is_port_allowed_with_custom_allowed() {
    // Test with custom allowed ports
    let yaml = r#"
enabled: true
port: 8080
host: 127.0.0.1
allowed_target_ports: [80, 443, 3000]
"#;
    let config: HttpForwardProxyConfig = serde_yaml::from_str(yaml).unwrap();

    // Custom allowed ports should be used
    assert!(config.is_port_allowed(80));
    assert!(config.is_port_allowed(443));
    assert!(config.is_port_allowed(3000));
    assert!(!config.is_port_allowed(8081)); // Not in custom list

    // Default blocked should still apply
    assert!(!config.is_port_allowed(22));
}

#[test]
fn test_proxy_is_port_allowed_with_custom_blocked() {
    // Test with custom blocked ports
    let yaml = r#"
enabled: true
port: 8080
host: 127.0.0.1
blocked_target_ports: [8080, 9000]
"#;
    let config: HttpForwardProxyConfig = serde_yaml::from_str(yaml).unwrap();

    // Custom blocked ports should override default
    // 8080 should be blocked even though it's in default allowed
    assert!(!config.is_port_allowed(8080));
    assert!(!config.is_port_allowed(9000));

    // Default blocked ports should NOT apply (since we set custom blocked)
    // So 22 should be allowed if it's in the default allowed list
    // Note: The implementation is that if blocked_target_ports is set,
    // it uses that instead of default_blocked_ports
}

#[test]
fn test_proxy_sanitize_headers() {
    use axum::http::HeaderMap;
    use lumina::http_proxy::auth::sanitize_request_headers;

    let mut headers = HeaderMap::new();
    headers.insert("X-Forwarded-For", "1.2.3.4".parse().unwrap());
    headers.insert("X-Real-IP", "5.6.7.8".parse().unwrap());
    headers.insert("X-Internal-Secret", "secret".parse().unwrap());
    headers.insert("Content-Type", "application/json".parse().unwrap());

    sanitize_request_headers(&mut headers);

    // Internal headers should be removed
    assert!(headers.get("X-Forwarded-For").is_none());
    assert!(headers.get("X-Real-IP").is_none());
    assert!(headers.get("X-Internal-Secret").is_none());

    // Normal headers should remain
    assert!(headers.get("Content-Type").is_some());
}

#[test]
fn test_proxy_check_target_port_allowed() {
    use lumina::http_proxy::auth::check_target_port_allowed;

    // Create a config with proxy enabled
    let config_yaml = r#"
server:
  port: 3000
  host: 127.0.0.1

http_forward_proxy:
  enabled: true
  port: 8080
  host: 127.0.0.1

logging:
  level: info

statistics:
  enabled: false

routes:
  - model_name: "test-model"
    provider_type: "openai"
    api_key: "sk-test"
    enabled: true
"#;
    let mut config: Config = serde_yaml::from_str(config_yaml).unwrap();
    config.version = 1;

    // Test allowed port
    assert!(check_target_port_allowed(&config, 80).is_ok());
    assert!(check_target_port_allowed(&config, 443).is_ok());

    // Test blocked port
    assert!(check_target_port_allowed(&config, 22).is_err());
    assert!(check_target_port_allowed(&config, 3306).is_err());
}

#[test]
fn test_proxy_config_validation() {
    // Test valid config
    let yaml = r#"
server:
  port: 3000
  host: 127.0.0.1

http_forward_proxy:
  enabled: true
  port: 8080
  host: 127.0.0.1
  max_connections: 100
  idle_timeout_secs: 60

logging:
  level: info

statistics:
  enabled: false

routes:
  - model_name: "test-model"
    provider_type: "openai"
    api_key: "sk-test"
    enabled: true
"#;
    let mut config: Config = serde_yaml::from_str(yaml).unwrap();
    config.version = 1;
    assert!(config.validate().is_ok());

    // Test port 0 - invalid
    let yaml = r#"
server:
  port: 3000
  host: 127.0.0.1

http_forward_proxy:
  enabled: true
  port: 0
  host: 127.0.0.1

logging:
  level: info

statistics:
  enabled: false

routes:
  - model_name: "test-model"
    provider_type: "openai"
    api_key: "sk-test"
    enabled: true
"#;
    let mut config: Config = serde_yaml::from_str(yaml).unwrap();
    config.version = 1;
    assert!(config.validate().is_err());

    // Test same port as main server - invalid
    let yaml = r#"
server:
  port: 3000
  host: 127.0.0.1

http_forward_proxy:
  enabled: true
  port: 3000
  host: 127.0.0.1

logging:
  level: info

statistics:
  enabled: false

routes:
  - model_name: "test-model"
    provider_type: "openai"
    api_key: "sk-test"
    enabled: true
"#;
    let mut config: Config = serde_yaml::from_str(yaml).unwrap();
    config.version = 1;
    assert!(config.validate().is_err());

    // Test empty host - invalid
    let yaml = r#"
server:
  port: 3000
  host: 127.0.0.1

http_forward_proxy:
  enabled: true
  port: 8080
  host: ""

logging:
  level: info

statistics:
  enabled: false

routes:
  - model_name: "test-model"
    provider_type: "openai"
    api_key: "sk-test"
    enabled: true
"#;
    let mut config: Config = serde_yaml::from_str(yaml).unwrap();
    config.version = 1;
    assert!(config.validate().is_err());

    // Test max_connections 0 - invalid
    let yaml = r#"
server:
  port: 3000
  host: 127.0.0.1

http_forward_proxy:
  enabled: true
  port: 8080
  host: 127.0.0.1
  max_connections: 0

logging:
  level: info

statistics:
  enabled: false

routes:
  - model_name: "test-model"
    provider_type: "openai"
    api_key: "sk-test"
    enabled: true
"#;
    let mut config: Config = serde_yaml::from_str(yaml).unwrap();
    config.version = 1;
    assert!(config.validate().is_err());

    // Test idle_timeout_secs 0 - invalid
    let yaml = r#"
server:
  port: 3000
  host: 127.0.0.1

http_forward_proxy:
  enabled: true
  port: 8080
  host: 127.0.0.1
  idle_timeout_secs: 0

logging:
  level: info

statistics:
  enabled: false

routes:
  - model_name: "test-model"
    provider_type: "openai"
    api_key: "sk-test"
    enabled: true
"#;
    let mut config: Config = serde_yaml::from_str(yaml).unwrap();
    config.version = 1;
    assert!(config.validate().is_err());

    // Test proxy disabled - validation should pass even with bad proxy config
    let yaml = r#"
server:
  port: 3000
  host: 127.0.0.1

http_forward_proxy:
  enabled: false
  port: 0
  host: ""

logging:
  level: info

statistics:
  enabled: false

routes:
  - model_name: "test-model"
    provider_type: "openai"
    api_key: "sk-test"
    enabled: true
"#;
    let mut config: Config = serde_yaml::from_str(yaml).unwrap();
    config.version = 1;
    assert!(config.validate().is_ok());
}
