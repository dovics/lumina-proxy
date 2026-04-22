# Lumina-Proxy

[![Build](https://github.com/your-username/lumina-proxy/actions/workflows/build.yml/badge.svg)](https://github.com/your-username/lumina-proxy/actions/workflows/build.yml)

A high-performance LLM routing proxy supporting multiple backend providers with hot config reload and Windows system tray integration.

## Features

- 🌐 **Multi-provider Support**: OpenAI, Anthropic, Gemini, Ollama, and OpenAI-compatible APIs
- 🔄 **Hot Config Reload**: Update configuration via API or system tray without restart
- 📊 **Token Statistics**: Accurate token usage counting and persistence
- 🔒 **Authentication**: Optional API token authentication protection
- 🖥️ **Cross-Platform System Tray**: Native system tray for Windows, macOS, and Linux with quick actions
- ⚡ **High Performance**: Built on Axum and Tokio async runtime
- 📝 **Flexible Logging**: Console and file logging with date/size rotation
- 🔀 **Model Aliasing**: Map client model names to upstream model names

## Supported Providers

| Provider | Type | Endpoint |
|----------|------|----------|
| OpenAI | `openai` | `/v1/chat/completions` |
| Anthropic | `anthropic` | `/v1/messages` |
| Google Gemini | `gemini` | `streamGenerateContent` |
| Ollama | `ollama` | `/api/chat` |
| OpenAI-compatible | `openai-compatible` | Custom URL |

## Quick Start

### Download Pre-built Binaries

You can download pre-built binaries from the [Releases](https://github.com/your-username/lumina-proxy/releases) page:

- **Windows**: `lumina-windows-x86_64.zip` (with system tray support)
- **Linux**: `lumina-linux-x86_64.tar.gz`
- **macOS (Intel)**: `lumina-macos-x86_64.tar.gz`
- **macOS (Apple Silicon)**: `lumina-macos-aarch64.tar.gz`

### Build from Source

```bash
cargo build --release
```

### Run

```bash
# Use default config file ./config.yaml
./target/release/lumina

# Or specify a config file path
./target/release/lumina /path/to/config.yaml
```

## Configuration

Refer to the `config.yaml` example:

```yaml
# Server configuration
server:
  port: 8080
  host: "0.0.0.0"
  auth_token: "your-secret-key"  # Optional

# Logging configuration
logging:
  level: "info"
  console: false
  file:
    enabled: true
    path: "./lumina.log"
    rotation: "daily"
    max_size_mb: 100
    max_files: 5

# Token statistics
statistics:
  enabled: true
  file_path: "./token_stats.jsonl"
  buffer_seconds: 1.0

# Route configuration
routes:
  - model_name: "gpt-4o"
    provider_type: "openai"
    base_url: "https://api.openai.com"
    api_key: "sk-xxxx"
    enabled: true

  - model_name: "claude-3-opus"
    provider_type: "anthropic"
    base_url: "https://api.anthropic.com"
    api_key: "sk-xxxx"
    enabled: true
```

### Model Aliasing

Map client-requested model names to different upstream model names:

```yaml
routes:
  - model_name: "gpt-4o"              # Client-requested model name
    upstream_model: "gpt-4o-2024-08-06"  # Actual upstream model name
    provider_type: "openai"
    base_url: "https://api.openai.com"
    api_key: "sk-xxxx"
    enabled: true
```

## API Endpoints

### Proxy Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completions (streaming and non-streaming) |
| GET | `/v1/models` | List available models |

### Admin Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/admin/reload-config` | Reload configuration file |
| GET | `/v1/admin/config` | Get current configuration (redacted) |

**Reload config example**:
```bash
curl -X POST http://localhost:8080/v1/admin/reload-config
```

## Cross-Platform System Tray

Lumina displays a system tray icon on **Windows, macOS, and Linux** with:
- View server address
- Quick config reload
- Open config file in default editor
- Show current config summary
- Exit application
- **Internationalization**: Chinese/English menu based on system language (LANG environment variable)

### Platform-Specific Requirements

#### Linux
Install system dependencies before building:
```bash
# Ubuntu/Debian
sudo apt-get install libayatana-appindicator3-dev
```

Requires a desktop environment supporting AppIndicator (GNOME, KDE, Xfce, etc.). In headless environments, Lumina automatically falls back to running without the tray.

#### macOS
No additional dependencies. For proper Dock icon behavior (no Dock icon shown), build as a proper app bundle with Info.plist containing `LSUIElement = true`. In SSH/CI environments, Lumina automatically falls back to running without the tray.

#### Windows
No additional dependencies required. Uses native Win32 API. Defaults to Chinese UI for backward compatibility.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Client App    │────▶│  Lumina Proxy   │
│  (OpenAI API)   │     │   (Axum HTTP)   │
└─────────────────┘     └────────┬────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   OpenAI API    │     │  Anthropic API  │     │     Ollama      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

Core components:
- **`ProxyState`**: Lock-free atomic config updates using `ArcSwap`
- **Config Validation**: Automatically validates config on reload, retains old config on failure
- **Token Statistics**: Accurate input/output token counting with async flushing

## CI/CD

GitHub Actions automatically builds and tests the project on every push and pull request:

| Platform | Target | Notes |
|----------|--------|-------|
| Windows x64 | `x86_64-pc-windows-msvc` | ✓ Embedded icon, system tray support |
| Linux x64 | `x86_64-unknown-linux-gnu` | |
| macOS Intel | `x86_64-apple-darwin` | |
| macOS Apple Silicon | `aarch64-apple-darwin` | |

Build artifacts are available for download from the Actions page after each successful build.

## Tech Stack

- **Web Framework**: [Axum](https://github.com/tokio-rs/axum)
- **Runtime**: [Tokio](https://tokio.rs/)
- **Configuration**: `serde` + `serde_yaml`
- **Atomic Updates**: `arc-swap`
- **System Tray**: `tray-icon` + `tao` (Windows, macOS, Linux)
- **Token Counting**: `tiktoken-rs`
- **CI/CD**: GitHub Actions

## License

MIT License
