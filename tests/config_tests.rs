use lumina_proxy::config::{Config, ProviderType};

#[test]
fn test_parse_valid_config() {
    let yaml = r#"
server:
  port: 8080
  host: "0.0.0.0"
  auth_token: "test-token"

logging:
  level: "info"
  console: true
  file:
    enabled: true
    path: "./lumina.log"
    rotation: "daily"
    max_files: 5

statistics:
  enabled: true
  file_path: "./token_stats.jsonl"
  buffer_seconds: 1.0

routes:
  - model_name: "llama3:8b"
    provider_type: "ollama"
    base_url: "http://localhost:11434"
    api_key: "ollama"
    enabled: true
"#;
    let config: Config = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(config.server.port, 8080);
    assert_eq!(config.server.host, "0.0.0.0");
    assert_eq!(config.server.auth_token, Some("test-token".to_string()));
    assert_eq!(config.routes.len(), 1);

    let route = &config.routes[0];
    assert_eq!(route.model_name, "llama3:8b");
    assert!(matches!(route.provider_type, ProviderType::Ollama));
    assert_eq!(route.base_url, Some("http://localhost:11434".to_string()));
    assert_eq!(route.api_key, "ollama");
    assert!(route.enabled);
}

#[test]
fn test_all_provider_types_deserialization() {
    let yaml = r#"
server:
  port: 8080
  host: "0.0.0.0"

logging:
  level: "info"
  console: true

statistics:
  enabled: false

routes:
  - model_name: "gpt-4o"
    provider_type: "openai"
    api_key: "sk-test"
    enabled: true

  - model_name: "llama3:8b"
    provider_type: "ollama"
    base_url: "http://localhost:11434"
    api_key: "ollama"
    enabled: true

  - model_name: "claude-3-opus"
    provider_type: "anthropic"
    api_key: "sk-ant-test"
    enabled: true

  - model_name: "gemini-pro"
    provider_type: "gemini"
    api_key: "google-test"
    enabled: true

  - model_name: "custom-model"
    provider_type: "openai-compatible"
    base_url: "http://localhost:8000/v1"
    api_key: "test-key"
    enabled: true
"#;
    let config: Config = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(config.routes.len(), 5);

    let routes = &config.routes;
    assert!(matches!(routes[0].provider_type, ProviderType::OpenAi));
    assert!(matches!(routes[1].provider_type, ProviderType::Ollama));
    assert!(matches!(routes[2].provider_type, ProviderType::Anthropic));
    assert!(matches!(routes[3].provider_type, ProviderType::Gemini));
    assert!(matches!(routes[4].provider_type, ProviderType::OpenAiCompatible));
}

#[test]
fn test_find_backend_for_model() {
    let yaml = r#"
server:
  port: 8080
  host: "0.0.0.0"

logging:
  level: "info"
  console: true

statistics:
  enabled: false

routes:
  - model_name: "gpt-4o"
    provider_type: "openai"
    api_key: "sk-test"
    enabled: true

  - model_name: "llama3:8b"
    provider_type: "ollama"
    base_url: "http://localhost:11434"
    api_key: "ollama"
    enabled: true

  - model_name: "disabled-model"
    provider_type: "openai"
    api_key: "sk-test"
    enabled: false
"#;
    let config: Config = serde_yaml::from_str(yaml).unwrap();

    // Found existing enabled model
    let found = config.find_backend_for_model("gpt-4o");
    assert!(found.is_some());
    assert_eq!(found.unwrap().model_name, "gpt-4o");

    // Found another enabled model
    let found = config.find_backend_for_model("llama3:8b");
    assert!(found.is_some());
    assert!(matches!(found.unwrap().provider_type, ProviderType::Ollama));

    // Not found - disabled model should not be found
    let found = config.find_backend_for_model("disabled-model");
    assert!(found.is_none());

    // Not found - non-existent model
    let found = config.find_backend_for_model("non-existent");
    assert!(found.is_none());
}

#[test]
fn test_load_from_file() {
    // Create a temporary config file for testing
    let temp_path = "./test_config.yaml";
    let content = r#"
server:
  port: 9000
  host: "127.0.0.1"
  auth_token: "file-test-token"

logging:
  level: "debug"
  console: true
  file:
    enabled: false

statistics:
  enabled: true
  file_path: "./test_stats.jsonl"
  buffer_seconds: 2.0

routes:
  - model_name: "test-model"
    provider_type: "openai"
    api_key: "test"
    enabled: true
"#;
    std::fs::write(temp_path, content).unwrap();

    let config = Config::load_from_file(temp_path).unwrap();
    assert_eq!(config.server.port, 9000);
    assert_eq!(config.server.host, "127.0.0.1");
    assert_eq!(config.server.auth_token, Some("file-test-token".to_string()));
    assert_eq!(config.routes.len(), 1);

    // Clean up
    std::fs::remove_file(temp_path).unwrap();
}
