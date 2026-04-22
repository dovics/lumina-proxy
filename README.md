# Lumina-Proxy

一个高性能的 LLM 路由代理，支持多种后端提供商，具有配置热重载和 Windows 系统托盘集成。

## 功能特性

- 🌐 **多提供商支持**: OpenAI、Anthropic、Gemini、Ollama 和 OpenAI 兼容 API
- 🔄 **配置热重载**: 无需重启即可通过 API 或系统托盘更新配置
- 📊 **Token 统计**: 精确的 Token 使用量统计和持久化
- 🔒 **认证保护**: 可选的 API Token 认证保护
- 🖥️ **Windows 系统托盘**: 原生系统托盘集成，支持快速操作
- ⚡ **高性能**: 基于 Axum 和 Tokio 异步运行时
- 📝 **灵活日志**: 支持控制台和文件日志，支持按日期/大小轮转
- 🔀 **模型别名**: 支持客户端模型名到上游模型名的映射

## 支持的提供商

| 提供商 | 类型 | 端点 |
|--------|------|------|
| OpenAI | `openai` | `/v1/chat/completions` |
| Anthropic | `anthropic` | `/v1/messages` |
| Google Gemini | `gemini` | `streamGenerateContent` |
| Ollama | `ollama` | `/api/chat` |
| OpenAI 兼容 | `openai-compatible` | 自定义 URL |

## 快速开始

### 编译

```bash
cargo build --release
```

### 运行

```bash
# 使用默认配置文件 ./config.yaml
./target/release/lumina

# 或指定配置文件路径
./target/release/lumina /path/to/config.yaml
```

## 配置

参考 `config.yaml` 示例：

```yaml
# 服务器配置
server:
  port: 8080
  host: "0.0.0.0"
  auth_token: "your-secret-key"  # 可选

# 日志配置
logging:
  level: "info"
  console: false
  file:
    enabled: true
    path: "./lumina.log"
    rotation: "daily"
    max_size_mb: 100
    max_files: 5

# Token 统计
statistics:
  enabled: true
  file_path: "./token_stats.jsonl"
  buffer_seconds: 1.0

# 路由配置
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

### 模型别名

可以将客户端请求的模型名映射到不同的上游模型名：

```yaml
routes:
  - model_name: "gpt-4o"              # 客户端请求的模型名
    upstream_model: "gpt-4o-2024-08-06"  # 实际发送给上游的模型名
    provider_type: "openai"
    base_url: "https://api.openai.com"
    api_key: "sk-xxxx"
    enabled: true
```

## API 端点

### 代理端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/chat/completions` | 聊天补全（流式和非流式） |
| GET | `/v1/models` | 列出可用模型 |

### 管理端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/admin/reload-config` | 重新加载配置文件 |
| GET | `/v1/admin/config` | 获取当前配置（脱敏） |

**重载配置示例**:
```bash
curl -X POST http://localhost:8080/v1/admin/reload-config
```

## Windows 系统托盘

在 Windows 上运行时，Lumina 会在系统托盘中显示图标，支持：
- 查看服务器地址
- 快速重载配置
- 退出程序

## 架构

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

核心组件：
- **`ProxyState`**: 使用 `ArcSwap` 实现无锁原子配置更新
- **配置验证**: 重载时自动验证配置有效性，失败时保留旧配置
- **Token 统计**: 精确计数输入/输出 Token，支持异步刷盘

## 技术栈

- **Web 框架**: [Axum](https://github.com/tokio-rs/axum)
- **运行时**: [Tokio](https://tokio.rs/)
- **配置**: `serde` + `serde_yaml`
- **原子更新**: `arc-swap`
- **系统托盘**: `tray-icon` + `tao` (Windows)
- **Token 计数**: `tiktoken-rs`

## 许可证

MIT License
