# HTTP 正向代理功能设计文档

**日期**: 2026-05-18
**版本**: 1.2
**作者**: Lumina-Proxy Team

## 1. 概述

为 Lumina-Proxy 添加独立的 HTTP 正向代理功能，支持普通 HTTP 代理和 HTTPS CONNECT 隧道。该功能与现有的 LLM 代理服务完全隔离，运行在独立端口上。

## 2. 需求背景

用户需要一个独立的 HTTP 代理服务，用于：
- 转发 HTTP/HTTPS 请求
- 支持 CONNECT 隧道进行 HTTPS 透传
- 独立的认证机制

## 3. 设计原则

1. **隔离性**：HTTP 代理服务与主 LLM 代理服务完全隔离
2. **独立配置**：独立的端口、主机和认证配置
3. **架构清晰**：新建独立模块，不侵入现有核心代码
4. **可扩展性**：易于后续添加访问控制、日志审计等功能
5. **安全性**：内置安全限制防止滥用
6. **可用性优先**：默认配置应满足正常使用场景

## 4. 架构设计

### 4.1 整体架构

```
                    ┌─────────────────────────────────────────────────┐
                    │            Lumina-Proxy 进程                    │
                    │                                                 │
  LLM 请求 ───────► │  ┌─────────────┐                               │
  (端口 3000)       │  │  LLM 代理   │                               │
                    │  │  服务器     │                               │
                    │  └─────────────┘                               │
                    │        │                                       │
                    │  ┌─────▼──────────┐                            │
                    │  │   ArcSwap      │ 共享配置快照              │
                    │  │   Config       │                            │
                    │  └─────┬──────────┘                            │
                    │        │                                       │
                    │  ┌─────▼──────────┐                            │
 HTTP 代理 ───────► │  │ HTTP 正向       │ ◄────── HTTP/HTTPS        │
(端口 8080)        │  │ 代理服务器      │        客户端请求          │
                    │  └─────────────────┘                            │
                    │                                                 │
                    └─────────────────────────────────────────────────┘
```

**运行时说明**：每个服务使用独立的 Tokio Runtime，保证隔离性和故障隔离。

### 4.2 模块结构

```
src/http_proxy/
├── mod.rs          # 模块入口，导出公共 API
├── server.rs       # 代理服务器实现（Axum + hyper 混合）
├── handlers.rs     # HTTP/CONNECT 方法处理逻辑
└── auth.rs         # 代理认证中间件
```

## 5. 详细设计

### 5.1 配置结构

**重要**：使用 `HttpForwardProxyConfig` 命名，避免与现有 `ProxyConfig`（LLM 出站代理配置）冲突。

```rust
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct HttpForwardProxyConfig {
    /// 是否启用 HTTP 正向代理
    pub enabled: bool,
    /// 代理服务监听端口
    pub port: u16,
    /// 代理服务绑定主机地址
    pub host: String,
    /// 代理认证 Token（可选）
    pub auth_token: Option<String>,
    
    // === 安全与性能配置 ===
    /// 最大并发连接数（默认 1024）
    pub max_connections: Option<usize>,
    /// 连接空闲超时秒数（默认 60）
    pub idle_timeout_secs: Option<u64>,
    /// 最大请求体大小（字节），默认 100MB
    pub max_request_body_size: Option<usize>,
    /// 允许的目标端口白名单（可选，不设置则使用默认允许列表）
    pub allowed_target_ports: Option<Vec<u16>>,
    /// 禁止的目标端口黑名单（可选，默认包含危险端口）
    pub blocked_target_ports: Option<Vec<u16>>,
}
```

在 `Config` 结构中添加：

```rust
pub struct Config {
    pub server: ServerConfig,
    pub http_forward_proxy: Option<HttpForwardProxyConfig>,
    pub logging: LoggingConfig,
    pub statistics: StatisticsConfig,
    pub routes: Vec<RouteConfig>,
    // ...
}
```

#### 配置文件示例

```yaml
server:
  port: 3000
  host: 127.0.0.1

http_forward_proxy:
  enabled: true
  port: 8080
  host: 127.0.0.1
  auth_token: "proxy-secret-token"
  max_connections: 1024
  idle_timeout_secs: 60
  max_request_body_size: 104857600  # 100MB
  # allowed_target_ports: [80, 443, 8080]  # 不设置则使用默认允许列表
  blocked_target_ports: [22, 3306, 5432, 6379, 27017]  # SSH + 数据库端口

logging:
  level: info
```

#### 配置验证规则

1. 如果 `enabled: true`，端口不能为 0
2. 代理端口不能与主服务器端口相同
3. host 不能为空字符串
4. `max_connections` > 0（如果设置）
5. `idle_timeout_secs` > 0（如果设置）

#### 默认允许的目标端口

如果 `allowed_target_ports` 未设置，默认允许：
- 80 (HTTP)
- 443 (HTTPS)
- 8080-8090 (常见 Web 服务端口)

可通过设置 `allowed_target_ports` 覆盖此默认列表。

#### 默认禁止的目标端口

`blocked_target_ports` 默认包含（可覆盖）：
- 22 (SSH)
- 3306 (MySQL)
- 5432 (PostgreSQL)
- 6379 (Redis)
- 27017 (MongoDB)

**优先级**：端口同时在白名单和黑名单时，黑名单优先。

### 5.2 认证机制

使用标准的 `Proxy-Authorization` 请求头：

```http
Proxy-Authorization: Bearer <token>
```

认证流程：
1. 中间件检查请求是否包含 `Proxy-Authorization` 头
2. 如果配置了 `auth_token`，验证 Bearer token 是否匹配
3. 认证失败返回 `407 Proxy Authentication Required`，携带 `Proxy-Authenticate: Bearer` 头
4. 认证成功或未配置 token 则继续处理请求

### 5.3 关键技术难点：Axum CONNECT 方法支持

Axum 原生路由不支持 `CONNECT` 方法的匹配。解决方案：

**方案**：使用 `axum::routing::any()` 配合方法内部分发

```rust
// server.rs 实现思路
async fn proxy_dispatch_handler(
    State(state): State<Arc<ProxyState>>,
    req: Request<Body>,
) -> Result<Response<Body>, StatusCode> {
    match req.method() {
        &Method::CONNECT => handle_connect_tunnel(state, req).await,
        _ => handle_http_proxy(state, req).await,
    }
}

// 路由配置
let app = Router::new()
    .fallback(proxy_dispatch_handler)  // fallback 捕获所有路径
    .layer(axum::middleware::from_fn_with_state(
        shared_config.clone(),
        proxy_auth_middleware,
    ));
```

### 5.4 请求处理流程

#### 5.4.1 HTTP 普通代理请求

```
客户端 (curl -x http://proxy:8080 http://example.com)
    │
    ▼
GET http://example.com/path HTTP/1.1
Host: example.com
Proxy-Authorization: Bearer xxx
    │
    ▼
代理服务器 (端口 8080)
    ├─→ 认证中间件
    ├─→ 解析完整目标 URL (http://example.com/path)
    ├─→ 安全检查：目标端口在白名单/黑名单中
    ├─→ 请求头清理（移除内部敏感头）
    ├─→ 构建 reqwest 请求
    ├─→ 转发到目标服务器
    └─→ 流式返回响应
```

#### 5.4.2 HTTPS CONNECT 隧道

```
客户端 (curl -x http://proxy:8080 https://example.com)
    │
    ▼
CONNECT example.com:443 HTTP/1.1
Host: example.com:443
Proxy-Authorization: Bearer xxx
    │
    ▼
代理服务器 (端口 8080)
    ├─→ 认证中间件
    ├─→ 解析目标主机:端口
    ├─→ 安全检查：目标端口在白名单/黑名单中
    ├─→ 与目标服务器建立 TCP 连接（超时控制）
    ├─→ 返回 HTTP/1.1 200 Connection Established
    └─→ 双向字节流转发（带 idle 超时）
         ├─→ 客户端 → 代理 → 目标服务器
         └─→ 目标服务器 → 代理 → 客户端
```

**双向转发实现要点**：
- 使用 `tokio::io::copy_bidirectional`
- 设置读/写超时防止挂起连接
- 统计传输字节数用于日志
- 优雅关闭时等待传输完成（带超时）
- 正确处理 TCP 半关闭（half-close）

### 5.5 reqwest 客户端配置策略

`HttpForwardProxyServer` 中的客户端配置：

```rust
impl HttpForwardProxyServer {
    pub fn new(shared_config: Arc<ArcSwap<Config>>) -> Result<Self> {
        let client = reqwest::Client::builder()
            // 连接超时：10秒
            .connect_timeout(Duration::from_secs(10))
            // 请求超时：根据 idle_timeout_secs 配置，默认 60秒
            .timeout(Duration::from_secs(60))
            // 连接池大小：与 max_connections 一致
            .pool_max_idle_per_host(10)
            // 启用 TCP_NODELAY 减少延迟
            .tcp_nodelay(true)
            // 不继承系统代理（避免嵌套代理）
            .no_proxy()
            .build()?;
            
        Ok(Self { shared_config, client })
    }
}
```

**说明**：
- 连接池独立于主 LLM 代理的客户端，避免互相影响
- 不启用系统代理，防止无限嵌套代理
- 超时配置可通过配置文件动态调整

### 5.6 服务器启动流程

**重要变更**：两个服务共享同一个 Tokio Runtime，减少资源消耗。

```rust
// main.rs 启动流程伪代码

fn main() -> Result<()> {
    // 1. 加载配置
    let config = Config::load_from_file(&config_path)?;
    
    // 2. 创建 shutdown 广播通道（支持多个接收者）
    let (shutdown_tx, _shutdown_rx) = broadcast::channel::<()>(16);
    
    // 3. 创建共享配置（必须在启动服务之前）
    let shared_config = Arc::new(ArcSwap::from_pointee(config.clone()));
    
    // 4. 启动主 LLM 代理服务器
    let main_shutdown_rx = shutdown_tx.subscribe();
    let main_config = shared_config.clone();
    let server_handle = std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(run_server_with_shared_config(main_config, ..., main_shutdown_rx))
    });
    
    // 5. 启动 HTTP 正向代理服务（如果启用）
    if let Some(proxy_config) = &config.http_forward_proxy 
        && proxy_config.enabled 
    {
        let proxy_shared_config = shared_config.clone();
        let proxy_shutdown_rx = shutdown_tx.subscribe();
        
        // 在与主服务器相同的线程中启动，共享 Runtime
        // 或者通过消息通道通知主线程中的 Runtime spawn 任务
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async move {
                HttpForwardProxyServer::new(proxy_shared_config)
                    .with_shutdown(proxy_shutdown_rx)
                    .run()
                    .await
            })
        });
    }
    
    // 6. 等待系统托盘或 Ctrl+C 信号
    // ...
}
```

### 5.7 核心组件

#### HttpForwardProxyServer (server.rs)

```rust
pub struct HttpForwardProxyServer {
    /// 共享配置（支持热重载）
    shared_config: Arc<ArcSwap<Config>>,
    /// 关闭信号接收
    shutdown: broadcast::Receiver<()>,
    /// HTTP 客户端（复用连接池）
    client: reqwest::Client,
}

impl HttpForwardProxyServer {
    pub fn new(shared_config: Arc<ArcSwap<Config>>) -> Result<Self>;
    
    pub fn with_shutdown(self, shutdown: broadcast::Receiver<()>) -> Self;
    
    pub async fn run(self) -> Result<()> {
        // 1. 从共享配置读取当前代理配置快照
        // 2. 构建 Axum 路由 + fallback handler
        // 3. 应用认证中间件
        // 4. 绑定端口（使用 tokio::net::TcpListener）
        // 5. 启动 axum serve 并配置优雅关闭
        // 6. 关闭时等待活跃连接完成（带 5 秒强制超时）
    }
}
```

#### Handlers (handlers.rs)

```rust
/// 处理 CONNECT 方法，建立 HTTPS 隧道
pub async fn handle_connect_tunnel(
    State(state): State<Arc<ProxyState>>,
    req: Request<Body>,
) -> Result<Response<Body>, StatusCode>;

/// 处理普通 HTTP 代理请求
pub async fn handle_http_proxy(
    State(state): State<Arc<ProxyState>>,
    req: Request<Body>,
) -> Result<Response<Body>, StatusCode>;

/// 安全检查：目标端口是否允许访问
/// 返回 Ok(()) 表示允许，Err(message) 表示禁止
fn check_target_port_allowed(
    config: &HttpForwardProxyConfig,
    host: &str,
    port: u16,
) -> Result<(), String>;

/// 清理请求头：移除敏感/内部头
fn sanitize_request_headers(headers: &mut HeaderMap);
```

#### Auth Middleware (auth.rs)

```rust
/// 代理认证中间件
pub async fn proxy_auth_middleware(
    State(shared_config): State<Arc<ArcSwap<Config>>>,
    req: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    // 每次请求都从 ArcSwap 读取最新配置快照
    let config = shared_config.load();
    let proxy_config = match &config.http_forward_proxy {
        Some(c) if c.enabled => c,
        _ => return Ok(next.run(req).await),  // 未启用则直接通过
    };
    
    // 认证逻辑...
}
```

**注意**：代理未启用时服务器根本不会启动，中间件的短路逻辑仅作为防御性编程。

## 6. 配置热重载

### 6.1 热重载支持矩阵

| 配置项 | 热重载支持 | 说明 |
|--------|-----------|------|
| `enabled` | ⚠️ 支持但需重启 | 服务启停需要重启服务器 |
| `port` | ⚠️ 支持但需重启 | 端口绑定需要重启服务器 |
| `host` | ⚠️ 支持但需重启 | 地址绑定需要重启服务器 |
| `auth_token` | ✅ 动态生效 | 每次请求从共享配置读取 |
| `max_connections` | ✅ 动态生效 | 新连接使用新限制 |
| `idle_timeout_secs` | ✅ 动态生效 | 新连接使用新超时 |
| `allowed_target_ports` | ✅ 动态生效 | 新请求立即应用 |
| `blocked_target_ports` | ✅ 动态生效 | 新请求立即应用 |

### 6.2 配置变更检测与重启策略

**检测时机**：在 `reload_config_handler` 中检测代理相关配置变化。

**变更检测逻辑**：

```rust
// 伪代码：在 reload_config_handler 中
let old_config = state.config.load();
let new_config = Config::load_and_validate(&state.config_path)?;

// 检测是否需要重启代理服务器
let need_proxy_restart = match (&old_config.http_forward_proxy, &new_config.http_forward_proxy) {
    (None, None) => false,
    (Some(old), Some(new)) => {
        old.enabled != new.enabled 
        || old.port != new.port 
        || old.host != new.host
    }
    (None, Some(new)) => new.enabled,  // 新增启用
    (Some(old), None) => old.enabled,  // 被移除且之前已启用
};

if need_proxy_restart {
    // 发出警告：需要重启服务
    warnings.push("HTTP forward proxy configuration changed. Server restart required for port/host changes to take effect.".to_string());
    
    // 注意：本版本不实现"热重启"功能，仅发出警告提示用户手动重启
    // 未来版本可考虑：通过消息通道通知代理服务器任务停止，然后重新 spawn
}
```

**当前策略**：发出警告提示需要重启。不实现自动热重启，因为：
1. 端口/主机变更频率很低
2. 自动重启可能导致端口绑定失败等问题难以处理
3. 保持简单，遵循"显式优于隐式"原则

## 7. 安全设计

### 7.1 默认安全策略

- **默认绑定 127.0.0.1**：不暴露到公网除非显式配置
- **默认启用端口白名单**：仅允许常用 Web 端口
- **建议启用认证**：文档明确说明无认证的风险
- **请求体大小限制**：防止大流量攻击

### 7.2 请求头清理

转发前移除以下敏感头：
- `X-Forwarded-For`（防止伪造源 IP）
- `X-Real-IP`
- 任何 `X-Internal-*` 前缀的内部头

### 7.3 端口安全策略

采用"白名单 + 黑名单"双重机制：

1. **白名单 `allowed_target_ports`**：
   - 未设置时默认允许：80, 443, 8080-8090
   - 设置后仅允许列表中的端口

2. **黑名单 `blocked_target_ports`**：
   - 默认禁止：22 (SSH), 3306 (MySQL), 5432 (PostgreSQL), 6379 (Redis), 27017 (MongoDB)
   - 优先级高于白名单

3. **无特权端口特殊处理**：
   - 不再一刀切禁止所有 < 1024 端口
   - 80 和 443 默认在白名单中，满足正常使用

### 7.4 超时与连接限制

- TCP 连接超时：10 秒
- CONNECT 隧道 idle 超时：可配置，默认 60 秒
- 最大并发连接数：可配置，默认 1024
- 请求超时：与 idle 超时同步

## 8. 日志与监控

使用现有 `tracing` 框架，日志字段：

- `target_host`: 目标主机地址
- `method`: HTTP 方法
- `status`: 响应状态码
- `duration_ms`: 请求处理耗时
- `bytes_transferred`: 传输字节数（仅 CONNECT）
- `client_ip`: 客户端 IP 地址

日志示例：
```
INFO  HTTP proxy request: method=CONNECT target=example.com:443 status=200 duration_ms=45 bytes_transferred=12450 client_ip=127.0.0.1
```

**监控指标**（可选扩展）：
- 当前活跃 CONNECT 隧道数
- 总字节吞吐量（入站/出站）
- 按目标域名的统计

## 9. 错误处理

| 错误场景 | HTTP 状态码 | 说明 |
|----------|------------|------|
| 认证失败 | 407 | Proxy Authentication Required |
| 目标端口被禁止 | 403 | Forbidden - Target port not allowed |
| 目标主机无法连接 | 502 | Bad Gateway |
| 目标主机超时 | 504 | Gateway Timeout |
| 无效的 CONNECT 请求 | 400 | Bad Request |
| 超过最大连接数 | 503 | Service Unavailable |
| 请求体过大 | 413 | Payload Too Large |
| DNS 解析失败 | 502 | Bad Gateway - DNS resolution failed |
| 内部错误 | 500 | Internal Server Error |

**错误响应格式**：
```json
{
    "error": "Proxy Authentication Required",
    "code": 407,
    "message": "Valid Proxy-Authorization header is required"
}
```

## 10. 测试计划

### 单元测试
- 配置解析和验证测试
- 认证中间件测试
- 目标端口安全检查测试（白名单+黑名单）
- 请求头清理测试
- 默认端口列表验证测试

### 集成测试
- HTTP 代理请求转发测试（GET/POST）
- HTTPS CONNECT 隧道测试
- 双向数据传输正确性测试（含 half-close 场景）
- 认证功能测试
- 并发请求测试（验证 max_connections）
- 超时测试（idle timeout）
- 优雅关闭测试：关闭时正在传输的 CONNECT 隧道行为
- 热重载测试：修改 auth_token 后立即生效
- 端口限制测试：默认白名单/黑名单行为

### 性能测试
- 并发连接压力测试（验证文件描述符不会泄漏）
- 长连接稳定性测试
- 吞吐量测试

## 11. 实施步骤

1. **配置层**：添加 `HttpForwardProxyConfig` 结构和验证逻辑
2. **基础设施**：将 shutdown 通道从 mpsc 改为 broadcast，支持多接收者
3. **新建模块**：创建 `src/http_proxy/` 模块结构
4. **认证实现**：实现代理认证中间件
5. **安全工具**：实现端口检查（白名单+黑名单）、请求头清理等工具函数
6. **HTTP 代理 Handler**：实现普通 HTTP 请求转发
7. **CONNECT 隧道 Handler**：实现 HTTPS 隧道双向转发
8. **服务器实现**：实现 `HttpForwardProxyServer`
9. **服务器集成**：在 main.rs 中集成代理服务器启动逻辑
10. **热重载逻辑**：在 reload_config_handler 中添加配置变更检测和警告
11. **测试**：编写测试用例
12. **文档**：更新 config.yaml 示例

## 12. 风险与注意事项

1. **安全风险**：开放代理可能被滥用，建议默认启用认证且绑定 127.0.0.1
2. **资源消耗**：CONNECT 隧道是长连接，需要设置合理的超时和连接数限制
3. **端口冲突**：确保代理端口与主服务器端口不同
4. **文件描述符限制**：大量并发代理连接可能消耗较多文件描述符，需要系统级配置
5. **Windows 平台**：注意 Windows 下大量 TCP 连接的注册表配置限制
6. **系统托盘集成**：托盘菜单中应显示 HTTP 代理状态和端口
7. **半关闭行为**：CONNECT 隧道的 TCP half-close 行为需要充分测试

## 13. 变更历史

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| 1.0 | 2026-05-18 | 初始版本 |
| 1.1 | 2026-05-18 | 第一次代码审查更新：<br>- 配置重命名为 HttpForwardProxyConfig<br>- 添加 broadcast shutdown 通道<br>- 补充 Axum CONNECT 方法实现方案<br>- 添加 ArcSwap 热重载支持<br>- 增加安全配置项和限制<br>- 完善测试计划 |
| 1.2 | 2026-05-18 | 第二次代码审查更新：<br>- 修正端口限制策略：白名单+黑名单，默认允许 80/443<br>- 移除特权端口一刀切限制，改为危险端口黑名单<br>- 补充 reqwest 客户端配置细节<br>- 明确热重载服务器重启策略（警告+手动重启）<br>- 优化 Runtime 共享说明 |
