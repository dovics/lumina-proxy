# Windows System Tray Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Windows system tray support with graceful shutdown to Lumina-Proxy

**Architecture:** 
- Windows: Tao event loop runs on main thread, Axum server runs on a background tokio thread
- Communication via `tokio::sync::mpsc` channel for shutdown signals
- Non-Windows platforms compile cleanly without tray functionality
- Tray icon embedded in binary using `include_bytes!`

**Tech Stack:** tray-icon 0.15, muda 0.12, tao 0.28, tokio, axum

---

## Task 1: Add Dependencies and Create Icon Asset

**Files:**
- Modify: `Cargo.toml`
- Create: `assets/icon.ico`

- [ ] **Step 1: Add Windows-only dependencies to Cargo.toml**

```toml
# Add at the end of Cargo.toml
[target.'cfg(windows)'.dependencies]
tray-icon = "0.15"
muda = "0.12"
tao = "0.28"
```

- [ ] **Step 2: Create assets directory and download/generate a valid .ico file**

```bash
mkdir -p assets
# Download a simple default icon, or create one
# For development, you can use any valid .ico file
# For example: download a simple red square icon from a public source
# or use ImageMagick: convert -size 64x64 xc:red assets/icon.ico
# If no icon is provided during development, add a fallback in load_icon()
```

- [ ] **Step 3: Verify project compiles with new dependencies**

Run: `cargo check`
Expected: Success, no errors

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml
git commit -m "feat: add Windows system tray dependencies"
```

---

## Task 2: Create Tray Module Skeleton

**Files:**
- Create: `src/tray.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Create src/tray.rs with conditional compilation structure**

```rust
//! System Tray Module
//! Provides Windows system tray functionality for graceful shutdown

use anyhow::Result;

#[cfg(windows)]
pub mod imp {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::mpsc;

    /// Tray manager for Windows system tray
    pub struct TrayManager {
        shutdown_tx: mpsc::Sender<()>,
    }

    impl TrayManager {
        /// Create a new tray manager
        pub fn new(shutdown_tx: mpsc::Sender<()>) -> Result<Self> {
            Ok(Self { shutdown_tx })
        }

        /// Run the tray event loop (blocks the thread)
        pub fn run(self, server_addr: String) -> Result<()> {
            Ok(())
        }
    }
}

#[cfg(not(windows))]
pub mod imp {
    use super::*;
    use tokio::sync::mpsc;

    /// Stub TrayManager for non-Windows platforms
    pub struct TrayManager;

    impl TrayManager {
        pub fn new(_shutdown_tx: mpsc::Sender<()>) -> Result<Self> {
            Ok(Self)
        }

        pub fn run(self, _server_addr: String) -> Result<()> {
            Ok(())
        }
    }
}

pub use imp::TrayManager;
```

- [ ] **Step 2: Export tray module in src/lib.rs**

```rust
// Add at the end of src/lib.rs
pub mod tray;
```

- [ ] **Step 3: Verify compilation**

Run: `cargo check`
Expected: Success, no errors

- [ ] **Step 4: Commit**

```bash
git add src/tray.rs src/lib.rs
git commit -m "feat: add tray module skeleton with conditional compilation"
```

---

## Task 3: Implement Windows Tray Functionality

**Files:**
- Modify: `src/tray.rs`

- [ ] **Step 1: Implement the Windows tray in src/tray.rs**

Replace the contents of the `#[cfg(windows)] pub mod imp` block with:

```rust
#[cfg(windows)]
pub mod imp {
    use super::*;
    use anyhow::Context;
    use muda::{Menu, MenuItem, PredefinedMenuItem};
    use tao::event_loop::{EventLoop, ControlFlow};
    use tray_icon::{TrayIconBuilder, Icon};
    use tokio::sync::mpsc;

    /// Tray manager for Windows system tray
    pub struct TrayManager {
        shutdown_tx: mpsc::Sender<()>,
    }

    impl TrayManager {
        /// Create a new tray manager
        pub fn new(shutdown_tx: mpsc::Sender<()>) -> Result<Self> {
            Ok(Self { shutdown_tx })
        }

        /// Run the tray event loop (blocks the thread)
        /// This must run on the main thread on Windows for proper message loop
        pub fn run(self, server_addr: String) -> Result<()> {
            // Create the menu
            let menu = Self::build_menu(&server_addr);

            // Load icon - use built-in default if no embedded icon
            let icon = Self::load_icon()?;

            // Create tray icon
            let _tray_icon = TrayIconBuilder::new()
                .with_menu(Box::new(menu))
                .with_icon(icon)
                .with_tooltip(format!("Lumina Proxy - {}", server_addr))
                .build()
                .context("Failed to create tray icon")?;

            // Create event loop
            let event_loop = EventLoop::new();
            let shutdown_tx = self.shutdown_tx;

            // Run event loop (blocks)
            event_loop.run(move |_event, _, control_flow| {
                *control_flow = ControlFlow::Wait;

                // Check for menu events
                if let Ok(event) = muda::MenuEvent::receiver().try_recv() {
                    if event.id() == "exit" {
                        // Send shutdown signal (best effort, non-blocking)
                        let _ = shutdown_tx.try_send(());
                        // Exit the event loop
                        *control_flow = ControlFlow::Exit;
                    }
                }
            });

            Ok(())
        }

        fn build_menu(server_addr: &str) -> Menu {
            let status_item = MenuItem::new(
                format!("🟢 服务器运行中\n{}", server_addr),
                false,
                None,
                None,
            );

            let exit_item = MenuItem::new(
                "退出",
                true,
                None,
                Some("exit".into()),
            );

            let menu = Menu::new();
            menu.append_items(&[
                &status_item,
                &PredefinedMenuItem::separator(),
                &exit_item,
            ]).unwrap();

            menu
        }

        fn load_icon() -> Result<Icon> {
            // First try to use the embedded icon
            static ICON_DATA: &[u8] = include_bytes!("../../assets/icon.ico");
            match Icon::from_bytes(ICON_DATA, None) {
                Ok(icon) => Ok(icon),
                Err(_) => {
                    // Fallback: create a simple RGBA icon (red square)
                    tracing::warn!("Failed to load embedded icon, using fallback");
                    Icon::from_rgba(vec![255, 0, 0, 255; 256 * 256 * 4], 256, 256)
                        .context("Failed to create fallback tray icon")
                }
            }
        }
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check`
Expected: Success, no errors

- [ ] **Step 3: Commit**

```bash
git add src/tray.rs
git commit -m "feat: implement Windows system tray functionality"
```

---

## Task 4: Refactor main.rs for Thread Model

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Update main.rs with new thread model**

Replace the entire contents of src/main.rs with:

```rust
//! Lumina-Proxy - LLM Routing Proxy
//! Main entry point

use std::sync::Arc;
use anyhow::Result;
use axum::Router;
use axum::routing::{get, post};
use tokio::sync::mpsc;

use lumina_proxy::config::Config;
use lumina_proxy::logging::init_logging;
use lumina_proxy::stats::StatsWriter;
use lumina_proxy::proxy::{ProxyState, models_handler, proxy_handler};
use lumina_proxy::auth::auth_middleware;
use lumina_proxy::tray::TrayManager;

#[cfg(windows)]
fn main() -> Result<()> {
    // Windows: Tao event loop must run on main thread
    // So we run Axum on a background thread

    // Get config path from command line argument or use default
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./config.yaml".to_string());

    // Load configuration
    let config = Config::load_from_file(&config_path)?;

    // Initialize logging
    init_logging(&config.logging)?;

    // Create shutdown channel
    let (shutdown_tx, shutdown_rx) = mpsc::channel::<()>(1);

    // Spawn Axum server on a background thread
    let server_addr = format!("{}:{}", config.server.host, config.server.port);
    let server_config = config.clone();
    let server_handle = std::thread::spawn(move || {
        // Create a tokio runtime for the server thread
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(run_server(server_config, shutdown_rx))
    });

    // Run tray on main thread (Windows only)
    tracing::info!("Starting tray on main thread");
    if let Err(e) = TrayManager::new(shutdown_tx)?.run(server_addr) {
        tracing::warn!("Tray failed to start: {}, running without tray", e);
    }

    // Wait for server to finish
    server_handle.join().unwrap()?;

    Ok(())
}

#[cfg(not(windows))]
#[tokio::main]
async fn main() -> Result<()> {
    // Non-Windows: Simple, just run Axum on main thread
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./config.yaml".to_string());

    let config = Config::load_from_file(&config_path)?;
    init_logging(&config.logging)?;

    let (_shutdown_tx, shutdown_rx) = mpsc::channel::<()>(1);
    run_server(config, shutdown_rx).await
}

/// Run the Axum server with graceful shutdown support
async fn run_server(config: Config, mut shutdown_rx: mpsc::Receiver<()>) -> Result<()> {
    // Initialize HTTP client
    let client = reqwest::Client::new();

    // Initialize stats writer if statistics enabled
    let stats_writer = if config.statistics.enabled {
        Some(StatsWriter::new(&config.statistics).await?)
    } else {
        None
    };

    // Create proxy state
    let proxy_state = Arc::new(ProxyState {
        config: config.clone(),
        client,
        stats_writer,
    });

    // Build Axum router
    let mut router = Router::new()
        .route("/v1/chat/completions", post(proxy_handler))
        .route("/v1/models", get(models_handler))
        .with_state(proxy_state);

    // Add authentication middleware if auth token is configured
    if config.server.auth_token.is_some() {
        router = router.layer(axum::middleware::from_fn_with_state(config.clone(), auth_middleware));
    }

    // Bind to configured host/port and serve with graceful shutdown
    let addr = format!("{}:{}", config.server.host, config.server.port);
    tracing::info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    // Use graceful shutdown
    axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            // Wait for shutdown signal
            let _ = shutdown_rx.recv().await;
            tracing::info!("Received shutdown signal, initiating graceful shutdown");
        })
        .await?;

    tracing::info!("Server shutdown complete");
    Ok(())
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check`
Expected: Success, no errors

- [ ] **Step 3: Commit**

```bash
git add src/main.rs
git commit -m "feat: refactor main for Windows tray thread model

- Windows: Tao event loop on main thread, Axum on background thread
- Non-Windows: Axum on main thread (no change)
- Add graceful shutdown support via mpsc channel"
```

---

## Task 5: Test and Verify

**Files:**
- No new files (verification only)

- [ ] **Step 1: Run full compilation**

Run: `cargo build --release`
Expected: Success, no errors

- [ ] **Step 2: Run existing tests**

Run: `cargo test`
Expected: All tests pass

- [ ] **Step 3: Manual verification checklist**

1. [ ] Start the proxy: `cargo run`
2. [ ] Verify tray icon appears in system tray
3. [ ] Right-click tray icon, verify menu shows status and "Exit"
4. [ ] Click "Exit", verify server logs graceful shutdown message
5. [ ] Verify tray icon is removed from system tray

- [ ] **Step 4: Commit verification note**

```bash
# No code changes to commit - this is a verification step
echo "System tray implementation complete and verified"
```

---

## Acceptance Criteria (Final Check)

- [ ] Windows compilation succeeds
- [ ] Tray icon appears in system tray on Windows
- [ ] Right-click menu displays server status and "Exit" option
- [ ] Clicking "Exit" triggers graceful server shutdown
- [ ] Tray icon is properly removed after exit
- [ ] Linux/macOS compilation succeeds (no tray functionality)
- [ ] All existing tests pass
