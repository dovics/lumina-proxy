# 产品需求文档 (PRD): Rust 轻量级 LLM 路由代理

**项目代号**: `Lumina-Proxy`
**版本**: v0.1.0
**目标**: 使用 Rust 构建一个内存占用极低、延迟近乎为零的 LLM 统一分发网关。

---

## 1. 产品背景
在本地运行多个大模型（如通过 Ollama）或调用多个云端 API（OpenAI, Anthropic, Google Gemini）时，客户端需要管理不同的端点和秘钥。本代理旨在提供一个**统一的入口**，通过简单的路由规则和高性能的流量拦截，实现模型管理与用量统计。

### 1.1 核心价值
- **统一入口**: 客户端只需配置一个端点，通过模型名称自动路由
- **格式兼容**: 上游始终使用标准 OpenAI 格式，代理自动转换为后端提供商格式
- **低开销**: 内存占用 < 20MB，额外延迟 < 5ms
- **可观测**: 统一统计所有请求的 Token 用量，持久化到文件便于分析

---

## 2. 核心功能需求

### 2.1 统一 API 接口
* **兼容性**: 模拟标准的 OpenAI Chat Completions API 协议 (`/v1/chat/completions`)。
* **透传功能**: 支持流式 (Streaming) 和非流式响应。
* **入口唯一**: 所有模型都通过同一个 endpoint 访问。

### 2.2 动态模型路由 (Routing)
* **路由规则**: 支持在配置文件中根据请求的 `model` 字段转发到不同的下游地址。
* **后端映射**: 每个模型名称精确匹配一个后端提供商、baseURL 或完整 URL 和 API Key。

### 2.3 自动格式转换
* **输入转换**: 上游 OpenAI 格式 → 转换为后端提供商的原生格式
* **输出转换**: 后端提供商格式 → 转换回标准 OpenAI 格式
* **流式支持**: 对流式响应逐块转换，不增加延迟
* **多提供商**: 原生支持 OpenAI, Anthropic, Gemini, Ollama 格式
* **OpenAI 兼容**: 对于声明兼容 OpenAI 的提供商，无需格式转换，直接透传

### 2.4 高性能 Token 统计
* **请求计算**: 转发前异步计算 Prompt Tokens。
* **流式计算**: 在不阻塞流的情况下，通过 SSE 拦截器实时累计 Completion Tokens。
* **无感统计**: 统计过程不应增加显著的响应延迟。
* **持久存储**: Token 统计结果持久化到 JSONL 文件便于后续分析。

### 2.5 轻量化配置管理
* **静态配置**: 通过 `config.yaml` 加载路由表。
* **资源占用**: 目标静态内存占用 < 20MB，处理延迟损耗 < 5ms（不含网络 IO）。

---

## 3. 非功能需求
* **高性能并发**: 基于 `Tokio` 异步运行时，支持数千个并发连接。
* **安全性**: 支持简单的 `Bearer Token` 校验，防止本地代理被非法外部访问。
* **可观测性**: 控制台实时输出每笔请求的模型名称、状态码及消耗的 Token 总数。
* **日志输出**: 同时输出到控制台和配置文件。
* **可靠性**: 格式转换失败时返回清晰错误信息，不崩溃。

---

## 4. 技术架构 (细化)

### 4.1 组件分层架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Lumina-Proxy Architecture                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │  HTTP Server │ →  │ Auth Middle-  │ →  │  Request Router           │   │
│  │   (Axum)     │    │    ware       │    │                          │   │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘   │
│                                    ↓                                      │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              Format Conversion Layer                               │   │
│  │  • OpenAI → Provider Format (Outgoing) using provider SDK types   │   │
│  │  • Provider → OpenAI Format (Incoming)                             │   │
│  │  • Streaming-aware chunk-by-chunk conversion                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                      │
│  ┌──────────────────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  Backend Selection       │ →  │ HTTP Client  │ →  │ Token Counter│   │
│  │  (from config)           │    │  (Reqwest)   │    │ (tiktoken-rs)│   │
│  └──────────────────────────┘    └──────────────┘    └──────────────┘   │
│                                    ↓                                      │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              Response Interception & Incremental Token Counting     │   │
│  │  • Non-streaming: Count after completion                          │   │
│  │  • Streaming: Count on chunk arrival without blocking              │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                      │
│  ┌────────────────────┐      ┌────────────────────┐      ┌─────────────┐│
│  │  Structured Logging │  →   │ Token Persistence  │  →   │  Statistics ││
│  │  (Console + File)   │      │  (File-based JSONL) │      │  File       ││
│  └────────────────────┘       └────────────────────┘      └─────────────┘│
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.1.1 组件职责表

| 组件 | 职责 | 关键依赖 |
|-----------|----------------|------------------|
| **HTTP Server** | 监听 incoming OpenAI 格式请求，处理连接生命周期 | `Axum`, `Tokio` |
| **Auth Middleware** | 如配置了 auth_token，验证 Bearer token | `Axum` middleware |
| **Request Router** | 从请求中提取 model name，匹配后端配置 | `Serde`, 静态配置 |
| **Format Conversion Layer** | OpenAI 与后端提供商之间的双向格式转换 | 使用各提供商 SDK 类型定义 |
| **HTTP Client** | 发送转换后的请求到后端，流式响应回传 | `Reqwest` with streaming |
| **Token Counter** | 发送前计算 prompt tokens，接收时累计 completion tokens | `tiktoken-rs` |
| **Response Interceptor** | 包装流式响应，增量计数 Token | `Tower` service |
| **Structured Logging** | 同时输出到控制台和文件，可配置级别 | `tracing`, `tracing-appender` |
| **Token Persistence** | 将 Token 统计追加写入换行 JSON 文件 | `tokio::fs` |

### 4.1.2 技术选型确认

| 组件 | 选型 | 理由 |
| :--- | :--- | :--- |
| **Runtime** | `Tokio` | 行业标准的异步执行引擎。 |
| **Web 框架** | `Axum` | 基于 Tower 抽象，对中间件支持极佳。 |
| **HTTP 客户端**| `Reqwest` | 功能完善，原生支持 `Stream` 转发。 |
| **Tokenizer** | `tiktoken-rs` | 针对 OpenAI 模型的极速分词器，对其他模型也有很好的近似。 |
| **配置解析** | `Serde` + `Yaml` | Rust 最强序列化框架。 |
| **日志** | `tracing` + `tracing-appender` | 结构化日志，支持文件轮转。 |
| **格式转换** | 官方 SDK 类型复用 | 利用官方维护的类型定义，减少手写错误，易于更新。 |

### 4.2 数据流

### 4.2.1 非流式请求流程

```
1. Client  → OpenAI Chat Completions Request → Server
2. Server  → Validate Authentication → Router
3. Router  → Extract model field → Find matching backend
4. Router  → Parse OpenAI request struct → Convert to provider request struct (via SDK)
5. Convert → Async token count prompt → Store prompt tokens → Send to backend
6. Backend  → Full response → Parse provider response → Convert to OpenAI format
7. Convert → Token count completion → Store completion tokens
8. Convert → Write statistics to log + file → Return response to client
9. Client  ← OpenAI-format response ← Server
```

**额外延迟目标**: 格式转换 + Token 计数 < 2ms

### 4.2.2 流式请求流程

```
1. Client  → OpenAI Chat Completions Request (stream: true) → Server
2. Server  → Validate Authentication → Router
3. Router  → Extract model field → Find matching backend
4. Router  → OpenAI format → Convert to provider request format
5. Converter → Token count prompt → Store prompt tokens → Send to backend
6. Backend  → Streaming chunks (SSE) → [Chunk Interceptor]
7. Interceptor 对每个 chunk:
   a. 解析 provider 格式 chunk
   b. 转换为 OpenAI 格式 chunk
   c. 将转换后的 chunk 转发给客户端（不等待计数）
   d. 提取 delta content → 增量 Token 计数 → 原子累加
8. 流结束时:
   a. 总计 Token 数已经就绪
   b. 写入统计到日志 + 文件
9. Client  ← OpenAI-format streaming chunks ← Server
```

**关键设计点**: Token 计数与客户端转发并行，不会阻塞流，延迟增加可忽略。

## 4.3 格式转换规则 (使用 SDK 类型)

### 4.3.1 Provider 支持矩阵

| 提供商类型 | 请求转换 | 响应转换 | 端点处理 |
|---------|---------|---------|---------|
| **openai** | 无需转换（原生） | 无需转换（原生） | 用户配置完整 URL 或 baseURL + path |
| **ollama** | 完整转换 | 完整转换 | 自动拼接 `/api/chat` 到 baseURL |
| **anthropic** | 完整转换 | 完整转换 | 用户配置 baseURL 自动拼接 `/v1/messages` |
| **gemini** | 完整转换 | 完整转换 | 用户配置 baseURL 自动拼接 path 并插入 model name |
| **openai-compatible** | 无需转换（兼容） | 无需转换（兼容） | 直接透传，用户配置完整 URL |

**openai-compatible**: 适用于任何声明兼容 OpenAI API 的服务（如 DeepSeek, 第三方兼容服务等）

### 4.3.2 OpenAI → Ollama (请求转出)

| OpenAI 字段 | Ollama 等价字段 | 转换动作 |
|-------------|----------------|----------|
| `model` | `model` | 直接拷贝 |
| `messages[].role` | `messages[].role` | 直接拷贝 |
| `messages[].content` | `messages[].content` | 直接拷贝 |
| `temperature` | `temperature` | 直接拷贝 |
| `top_p` | `top_p` | 直接拷贝 |
| `max_tokens` | `num_predict` | 字段重命名 |
| `stream` | `stream` | 直接拷贝 |
| `stop` | `stop` | 直接拷贝 |
| `presence_penalty` | N/A | 丢弃 |
| `frequency_penalty` | N/A | 丢弃 |

**示例**:

OpenAI 输入:
```json
{
  "model": "llama3",
  "messages": [{"role": "user", "content": "Hello!"}],
  "temperature": 0.7,
  "max_tokens": 100,
  "stream": true
}
```

转换为 Ollama:
```json
{
  "model": "llama3",
  "messages": [{"role": "user", "content": "Hello!"}],
  "temperature": 0.7,
  "num_predict": 100,
  "stream": true
}
```

### 4.3.3 Ollama → OpenAI (响应转回)

| Ollama 字段 | OpenAI 等价字段 | 转换动作 |
|-------------|----------------|----------|
| `model` | `model` | 直接拷贝 |
| `created_at` | `created` | ISO 时间转 Unix 时间戳 |
| `message.content` (非流) | `choices[0].message.content` | 提取到最终位置 |
| `delta.content` (流) | `choices[0].delta.content` | 移动到 delta 位置 |
| `done` | N/A (终止) | `true` 时发送 OpenAI `[DONE]` |
| `eval_count` | `usage.completion_tokens` | 如果可用直接使用 |
| `prompt_eval_count` | `usage.prompt_tokens` | 如果可用直接使用 |

**流式块转换**:
- Ollama 每个 chunk 带 `delta` 字段
- 每个 chunk 转为 OpenAI SSE chunk 放到 `choices[0].delta`
- `done: true` 最终 chunk 包含完整 `usage` 统计

### 4.3.4 OpenAI → Anthropic (请求转出)

| OpenAI 字段 | Anthropic 等价字段 | 转换动作 |
|-------------|-------------------|----------|
| `model` | `model` | 直接拷贝 |
| `messages[].role` | `messages[].role` | 转换: `user`/`assistant` 保留，`system` → 提取到 `system` 顶级字段 |
| `messages[].content` | `messages[].content` | 直接拷贝 |
| `temperature` | `temperature` | 直接拷贝 |
| `top_p` | `top_p` | 直接拷贝 |
| `max_tokens` | `max_tokens` | 直接拷贝 |
| `stream` | `stream` | 直接拷贝 |
| `stop` | `stop_sequences` | 重命名字段 |
| `presence_penalty` | N/A | 丢弃 |
| `frequency_penalty` | N/A | 丢弃 |

**Anthropic endpoint**: `POST /v1/messages`

**System message 处理**: OpenAI 将 system 作为普通 message，Anthropic 顶级有 `system` 字段，提取合并。

### 4.3.5 Anthropic → OpenAI (响应转回)

| Anthropic 字段 | OpenAI 等价字段 | 转换动作 |
|----------------|-----------------|----------|
| `id` | `id` | 直接拷贝 |
| `model` | `model` | 直接拷贝 |
| `content[0].text` (non-stream) | `choices[0].message.content` | 提取 |
| `delta.text` (stream) | `choices[0].delta.content` | 移动到 delta 位置 |
| `usage.output_tokens` | `usage.completion_tokens` | 直接使用 |
| `usage.input_tokens` | `usage.prompt_tokens` | 直接使用 |
| `stop_reason` | `choices[0].finish_reason` | 映射转换 |

**流式块转换**:
- Anthropic 流式返回 `delta` 对象
- 每个 chunk 提取 `delta.text` 转换为 OpenAI delta format
- 最后一个 chunk 填入完整 usage 统计

### 4.3.6 OpenAI → Gemini (请求转出)

| OpenAI 字段 | Gemini 等价字段 | 转换动作 |
|-------------|----------------|----------|
| `model` | (in URL) | model 名插入 URL 路径 |
| `messages[].role` | `contents[].role` | `user` → `user`, `assistant` → `model` |
| `messages[].content` | `contents[].parts[].text` | 包装到 `parts[]` 数组 |
| `temperature` | `generationConfig.temperature` | 嵌套 |
| `top_p` | `generationConfig.top_p` | 嵌套 |
| `max_tokens` | `generationConfig.maxOutputTokens` | 重命名 + 嵌套 |
| `stream` | `alt=sse` | 处理在 URL |
| `stop` | `generationConfig.stopSequences` | 重命名 + 嵌套 |

**Gemini endpoint**: `POST /v1beta/models/{model}:streamGenerateContent?alt=sse`
- model name 从配置 model_name 获取，插入路径

### 4.3.7 Gemini → OpenAI (响应转回)

| Gemini 字段 | OpenAI 等价字段 | 转换动作 |
|-------------|-----------------|----------|
| `candidates[0].content.parts[0].text` (non-stream) | `choices[0].message.content` | 提取 |
| `candidates[0].content.parts[0].text` (stream) | `choices[0].delta.content` | 增量提取 |
| `usageMetadata.totalTokenCount` | `usage.total_tokens` | 使用总和 |
| `usageMetadata.promptTokenCount` | `usage.prompt_tokens` | 提取 |
| `usageMetadata.candidatesTokenCount` | `usage.completion_tokens` | 提取 |
| `finishReason` | `choices[0].finish_reason` | 转换映射 |

### 4.3.8 openai-compatible (透传)

对于标记为 `openai-compatible` 的后端（包括 DeepSeek 以及其他第三方兼容服务）:
- 请求无需转换，直接透传
- 响应无需转换，直接透传
- 只做 Token 计数和日志统计

## 4.4 配置文件格式 (完整细化)

```yaml
# 服务器配置
server:
  port: 8080
  host: "0.0.0.0"          # 默认: "0.0.0.0"
  auth_token: "local-secret-key"  # 可选，不设置则禁用认证

# 日志配置
logging:
  level: "info"            # 级别: "trace", "debug", "info", "warn", "error"
  console: true            # 启用控制台输出
  file:
    enabled: true          # 启用文件日志
    path: "./lumina.log"   # 日志文件路径
    rotation: "daily"      # 轮转策略: "daily", "size", "never"
    max_size_mb: 100       # 按大小轮转时的最大大小
    max_files: 5           # 保留最多多少个轮转文件

# Token 统计持久化配置
statistics:
  enabled: true
  file_path: "./token_stats.jsonl"  # JSONL 文件路径
  buffer_seconds: 1.0              # 缓冲写入多少秒后刷盘，减少 IO

# 路由配置
routes:
  - model_name: "gpt-4o"
    provider_type: "openai"          # openai | ollama | anthropic | gemini | openai-compatible
    base_url: "https://api.openai.com"
    api_key: "sk-xxxx"
    enabled: true

  - model_name: "claude-3-opus"
    provider_type: "anthropic"
    base_url: "https://api.anthropic.com"
    api_key: "sk-xxxx"
    enabled: true

  - model_name: "gemini-pro"
    provider_type: "gemini"
    base_url: "https://generativelanguage.googleapis.com"
    api_key: "xxxx"  # Gemini API key
    enabled: true

  - model_name: "deepseek-coder"
    provider_type: "openai-compatible"
    url: "https://api.deepseek.com/chat/completions"
    api_key: "sk-yyyy"
    enabled: true

  - model_name: "llama3:8b"
    provider_type: "ollama"
    base_url: "http://localhost:11434"  # Ollama 填写 base URL，自动拼接 /api/chat
    api_key: "ollama"                     # Ollama 本地通常不需要，可填任意值
    enabled: true
```

**配置说明**:
- 对于需要自动拼接 path 的提供商 (ollama, anthropic, gemini)，使用 `base_url`
- 对于 openai 和 openai-compatible，可以直接使用完整 `url`

### 4.4.1 Rust 配置结构 (Serde)

```rust
pub struct Config {
    pub server: ServerConfig,
    pub logging: Option<LoggingConfig>,
    pub statistics: Option<StatisticsConfig>,
    pub routes: Vec<RouteConfig>,
}

pub struct ServerConfig {
    pub port: u16,
    pub host: String,
    pub auth_token: Option<String>,
}

pub struct LoggingConfig {
    pub level: LevelFilter,
    pub console: bool,
    pub file: Option<FileLoggingConfig>,
}

pub struct FileLoggingConfig {
    pub enabled: bool,
    pub path: String,
    pub rotation: RotationStrategy,
    pub max_size_mb: Option<u64>,
    pub max_files: Option<u32>,
}

pub struct StatisticsConfig {
    pub enabled: bool,
    pub file_path: String,
    pub buffer_seconds: Option<f64>,
}

pub struct RouteConfig {
    pub model_name: String,  // 精确匹配
    pub provider_type: ProviderType,
    pub base_url: Option<String>,  // For providers that need auto-path
    pub url: Option<String>,       // For openai / openai-compatible
    pub api_key: String,
    pub enabled: bool,
}

pub enum ProviderType {
    OpenAi,
    Ollama,
    Anthropic,
    Gemini,
    OpenAiCompatible,
}

pub enum RotationStrategy {
    Daily,
    Size,
    Never,
}
```

## 4.5 Token 计数方案

### 4.5.1 请求 / Prompt Token 计数

**算法**:
1. 请求解析转换后，提取所有 messages
2. 对每个 message，拼接 `role` + `content`
3. 使用 `tiktoken-rs` 选择对应编码:
   - OpenAI 模型: `cl100k_base` (精确)
   - Anthropic Claude: `cl100k_base` (良好近似)
   - Gemini 模型: `cl100k_base` (良好近似)
   - 其他模型: 默认使用 `cl100k_base` (误差在 1-2% 以内，可以接受)
4. 异步 spawn 计数任务，不阻塞发送（本身极快，< 1ms）
5. 发送到后端前等待计数完成，结果存储用于后续持久化

**近似处理说明**: 对于非 OpenAI 模型，没有官方分词器，使用 cl100k_base 给出的结果是很好的近似，满足统计需求。如果需要精确值，可以由用户在配置中指定编码。

### 4.5.2 流式响应 Token 计数

**核心挑战**: 不能把整个响应缓存在内存中，必须增量计数。

**算法**:
1. 每个 SSE chunk 到达并转换为 OpenAI 格式后:
   - 提取 delta `content`
   - **立即**将转换后的 chunk 转发给客户端
   - 异步对 delta content 做增量计数
   - 使用 `AtomicUsize` 原子加到运行总计
2. 流结束时，总计已经可用，直接持久化

**优点**:
- 客户端零额外延迟接收数据
- Token 计数与客户端接收并发进行
- 无论输出多长，内存占用保持恒定

**实现细节**:
- 使用 `AtomicUsize` 存储运行中的 completion token 计数
- 因为只有一个线程顺序处理 chunks，不需要锁
- 流完成时立即可用最终值

## 4.6 Token 统计持久化

### 4.6.1 文件格式: 换行分隔 JSON (JSONL)

每个请求追加一行 JSON:

```json
{"timestamp":"2025-04-20T12:34:56Z","model":"llama3:8b","provider":"ollama","prompt_tokens":123,"completion_tokens":456,"total_tokens":579,"duration_ms":1234,"status":"success"}
{"timestamp":"2025-04-20T12:35:10Z","model":"gpt-4o","provider":"openai","prompt_tokens":512,"completion_tokens":200,"total_tokens":712,"duration_ms":3500,"status":"success"}
{"timestamp":"2025-04-20T12:35:15Z","model":"deepseek-coder","provider":"openai-compatible","prompt_tokens":1000,"completion_tokens":800,"total_tokens":1800,"duration_ms":2800,"status":"error","error_message":"request timeout"}
```

### 4.6.2 设计理由

- **JSONL**: 简单追加，无需重写整个文件，易于后续解析分析
- **一行一请求**: 原子追加，OS 保证不会损坏日志
- **包含所有相关字段**: 时间、模型、提供商、状态、错误信息，便于排障
- **可直接处理**: 可以用 `grep` 和 `jq` 直接查询统计

### 4.6.3 性能优化

- **缓冲写入**: 可配置缓冲刷新（默认 1 秒），减少系统调用
- **异步 IO**: 使用 `tokio::fs` 不阻塞请求处理
- **原子追加**: 每次写入原子性，不会造成日志损坏

### 4.6.4 数据查询示例

使用 `jq` 按日汇总 Token 使用量:
```bash
cat token_stats.jsonl | jq -s '
  group_by(.timestamp[0:10]) |
  map({
    date: .[0].timestamp[0:10],
    total_prompt: (map(.prompt_tokens) | add),
    total_completion: (map(.completion_tokens) | add),
    total_requests: length
  })
'
```

## 4.7 日志系统设计

### 4.7.1 架构

基于 Rust `tracing` 生态系统:

```
┌─────────────────────────────────────────────┐
│               tracing Events                 │
└─────────────────────────────────────────────┘
           ┌                           ┐
    ┌──────┴──────────┐       ┌────────┴──────┐
    │  Console Output  │       │   File Output  │
    └──────────────────┘       └────────────────┘
```

### 4.7.2 功能特性

- **双输出**: 可配置同时输出到控制台、文件，或只输出到一个
- **结构化日志**: 所有字段都是机器可解析的 JSON 格式
- **日志轮转**: 支持按日轮转或按大小轮转
- **自动清理**: 可配置保留最多个数轮转日志，自动删除旧文件

### 4.7.3 日志格式

**控制台输出** (对人类友好):
```
2025-04-20T12:34:56Z  INFO lumina::proxy: Request completed model=llama3:8b prompt_tokens=123 completion_tokens=456 total_tokens=579 duration_ms=1234 status=200
2025-04-20T12:35:15Z ERROR lumina::proxy: Backend request failed error="connection timeout"
```

**文件输出** (结构化 JSON):
```json
{"timestamp":"2025-04-20T12:34:56Z","level":"INFO","target":"lumina::proxy","message":"Request completed","model":"llama3:8b","prompt_tokens":123,"completion_tokens":456,"total_tokens":579,"duration_ms":1234,"status":200}
```

## 4.8 端到端示例: OpenAI → Ollama 流式

```
CLIENT (OpenAI format)
  ↓
{
  "model": "llama3",
  "messages": [{"role": "user", "content": "Hi!"}],
  "stream": true,
  "max_tokens": 100
}
  ↓
[Router] → 匹配 model "llama3" → provider=Ollama
  ↓
[转换 OpenAI → Ollama]
  ↓
{
  "model": "llama3",
  "messages": [{"role": "user", "content": "Hi!"}],
  "stream": true,
  "num_predict": 100
}
  ↓
[Token Counter] → 计数 prompt = 8 tokens → 存储
  ↓
发送到 http://localhost:11434/api/chat
  ↓
Ollama 流式返回 chunks:
{"model": "llama3", "delta": {"content": "Hel"}, "done": false}
  ↓
[Interceptor] → 转换 → 发送给客户端 → 计数 "Hel" = 1 token → 总计=1
  ↓
{"model": "llama3", "delta": {"content": "lo!"}, "done": true, "eval_count": 2}
  ↓
[Interceptor] → 转换 → 发送给客户端 → 计数 "lo!" = 1 token → 总计=2
  ↓
[Persistence] → 追加到 token_stats.jsonl
{"timestamp":"2025-04-20T...", "model":"llama3", "provider":"ollama",
 "prompt_tokens":8, "completion_tokens":2, "total_tokens":10, ...}
  ↓
[Logging] → 输出到控制台+文件: "Request completed ..."
  ↓
CLIENT (OpenAI format chunks):
data: {"id": "...", "model": "llama3", "choices":[{"delta": {"content": "Hel"}, ...}]}

data: {"id": "...", "model": "llama3", "choices":[{"delta": {"content": "lo!"}, ...}]}

data: [DONE]
```

## 4.9 非功能需求细化

| 指标 | 目标 | 实现途径 |
|--------|--------|---------------------|
| 静态内存占用 | < 20 MB | Rust + 精简依赖，无 unsafe 代码 |
| 单请求额外延迟 | < 5 ms (不含网络) | 格式转换使用 Serde 优化，Token 计数极速 |
| 并发连接 | > 10,000 | Tokio 异步运行时，连接池复用 |
| 持久化写入延迟 | < 10 ms | 缓冲异步写入，后台刷新 |
| 流式 Token 计数开销 | < 5% chunk 间隔时间 | 增量计数，不阻塞转发 |

---

## 5. 里程碑计划

### v0.1.0 - 最小可用版本
- [x] 配置解析（支持 YAML + 环境变量）
- [ ] HTTP 服务器 + 基础认证中间件
- [ ] 模型路由（精确匹配）
- [ ] 格式转换（OpenAI ↔ Ollama, OpenAI ↔ Anthropic, OpenAI ↔ Gemini）
- [ ] Prompt Token 计数 + 流式 Completion Token 增量计数
- [ ] 日志系统（控制台 + 文件输出）
- [ ] Token 统计持久化（JSONL）

### v0.2.0 - 计划中
- [ ] 配置热重载
- [ ] Prometheus 指标端点
- [ ] 请求重试 / 故障转移

---

## 6. 验收标准

1. **功能验收**
   - 客户端使用标准 OpenAI 格式请求不同模型，能正确路由到对应后端
   - Ollama/Anthropic/Gemini 后端能正确转换格式，返回标准 OpenAI 格式响应
   - openai-compatible 类型后端能直接透传
   - 流式和非流式请求都能正常工作
   - Token 计数误差 < 5%（对比官方计数）
   - 统计数据正确写入 JSONL 文件

2. **性能验收**
   - 静态内存占用 < 20MB
   - 单请求额外延迟 p99 < 5ms
   - 能支持 1000+ 并发连接

3. **可观测性验收**
   - 每个请求都有控制台日志
   - 日志同时输出到配置文件
   - 日志按配置策略正确轮转

---

## 附录 A: 依赖清单 (Cargo.toml)

```toml
[dependencies]
axum = "0.7"
tokio = { version = "1.0", features = ["full"] }
reqwest = { version = "0.12", features = ["stream", "json"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
serde_yaml = "0.9"
tiktoken-rs = "0.5"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
tracing-appender = "0.2"
thiserror = "1.0"
```
