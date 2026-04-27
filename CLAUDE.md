# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Lumina-Proxy**: A high-performance LLM routing proxy supporting multiple backend providers (OpenAI, Anthropic, Gemini, Ollama, OpenAI-compatible APIs) with hot config reload and Windows system tray integration.

## Common Commands

```bash
# Build the project
cargo build
cargo build --release

# Run the proxy (uses ./config.yaml by default)
cargo run
cargo run -- /path/to/config.yaml

# Run tests
cargo test
cargo test -- test_name  # Run single test by name

# Run linter (MUST PASS with -D warnings before any push)
cargo clippy -- -D warnings

# Check formatting
cargo fmt --check
cargo fmt  # Auto-format
```

## Architecture

### Core Architecture Pattern

```
                   ┌─────────────────────────────────────┐
                   │          main.rs                     │
                   │  ─────────────────────────────────  │
Windows            │  │  Windows Tray (main thread)  │   │
Platform ─────────┼──▶                              │   │
                   │  └──────────┬───────────────────┘   │
                   │             │ shared_config          │
                   │             ▼                        │
                   │  ┌───────────────────────────────┐  │
                   │  │  Axum Server (bg thread)      │  │
                   │  │  ─────────────────────────    │  │
                   │  │  /v1/chat/completions         │  │
                   │  │  /v1/models                   │  │
                   │  │  /v1/admin/reload-config      │  │
                   │  │  /v1/admin/config             │  │
                   │  └──────────────┬────────────────┘  │
                   └─────────────────┼───────────────────┘
                                     │
                         ┌───────────▼───────────┐
                         │  proxy.rs             │
                         │  ────────────────    │
                         │  ProxyState + ArcSwap │
                         └───────────┬───────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              convert.rs         auth.rs         stats.rs
         (provider-specific  (token auth)   (token counting
          request/response                   + persistence)
          conversion)
```

### Key Design Decisions

1. **ArcSwap for Atomic Config Updates**
   - `ProxyState.config: Arc<ArcSwap<Config>>` enables lock-free, atomic config reloads
   - In-flight requests continue using old config snapshot
   - New requests use updated config immediately
   - Apply this pattern for any shared mutable state (also used for `stats_writer`)

2. **Windows Thread Model**
   - Windows: Tray event loop runs on **main thread** (Tao requirement)
   - Axum server runs on a background thread with its own Tokio runtime
   - `shutdown_tx` channel coordinates graceful shutdown

3. **Provider Conversion Layer** (`convert.rs`)
   - All requests arrive as OpenAI-compatible format
   - Each provider has its own `convert_openai_to_PROVIDER()` function
   - Streaming responses converted back to OpenAI SSE format

4. **Configuration Validation**
   - Before reload completes, `Config::load_and_validate()` runs
   - Invalid configs are rejected with errors logged
   - Old valid configuration remains active

### Module Responsibilities

| Module | Key Exports | Purpose |
|--------|-------------|---------|
| `config.rs` | `Config`, `RouteConfig`, `ProviderType` | YAML parsing, validation, route lookup |
| `proxy.rs` | `ProxyState`, `*_handler` | Core HTTP logic, URL building, streaming |
| `convert.rs` | `convert_openai_to_*` | Provider-specific request/response translation |
| `auth.rs` | `auth_middleware` | Axum layer for token authentication |
| `stats.rs` | `StatsWriter`, `TokenStats` | Async token usage persistence (JSONL) |
| `tray.rs` | `TrayManager` | Windows system tray menu and reload triggers |
| `types.rs` | `OpenAIChatRequest`, `ProxyError` | Type definitions for API contracts |
| `token_counter.rs` | `count_prompt_tokens()` | Token counting via tiktoken |

## Important Patterns to Preserve

1. **Always add `PartialEq` to config structs** - used in validation and testing
2. **Use `#[cfg(windows)]` guards** for platform-specific code
3. **`config_path` is passed through `ProxyState`** - needed for admin reload endpoint
4. **`stats_writer` also uses `ArcSwap`** - can be toggled on/off via config reload

## Config Reload Coverage

When modifying code, consider if your change:
- ✅ Works correctly if config changes mid-request (read config snapshot at handler start)
- ✅ Requires server restart (e.g., port/host binding) - document in tray reload message
- ✅ Updates `stats_writer` if statistics config changes (see `reload_config_handler`)

## Test File Locations

- `tests/config_tests.rs` - Configuration parsing and validation tests
- `tests/conversion_tests.rs` - Provider conversion tests

## Pre-Push Checklist

**ALWAYS** run these before pushing code:

```bash
# 1. Clippy with deny warnings (CI will fail on this)
cargo clippy -- -D warnings

# 2. All tests pass
cargo test

# 3. Formatting check
cargo fmt --check
```

## Key Files to Reference

- `config.yaml` - Example configuration with all provider types
- `src/proxy.rs` - Core proxy implementation with admin endpoints
- `src/tray.rs` - Windows system tray integration
