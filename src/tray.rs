//! System Tray Module
//! Provides cross-platform system tray functionality for graceful shutdown and config management

use anyhow::Result;
use anyhow::Context;
use arc_swap::ArcSwap;
use std::sync::Arc;
use crate::config::Config;
use tao::event_loop::{EventLoop, ControlFlow};
use tokio::sync::mpsc;
use tray_icon::{
    menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem},
    TrayIconBuilder, Icon,
};
use std::path::Path;

pub struct TrayManager {
    shutdown_tx: mpsc::Sender<()>,
    config: Arc<ArcSwap<Config>>,
    config_path: String,
}

/// Detect system language from environment variables
/// Returns "zh" for Chinese, "en" for English or unknown
pub fn detect_system_language() -> &'static str {
    // Check environment variables in priority order
    let env_vars = ["LANG", "LC_ALL", "LC_MESSAGES", "USERLANGUAGE"];

    for var in env_vars {
        if let Ok(val) = std::env::var(var) {
            if val.starts_with("zh") {
                return "zh";
            }
            return "en";
        }
    }

    // Windows fallback: DEFAULT TO CHINESE for backward compatibility
    #[cfg(windows)]
    return "zh";

    // Unix fallback: English
    #[cfg(not(windows))]
    "en"
}

impl TrayManager {
    /// Create new TrayManager with shutdown sender and shared config
    pub fn new(shutdown_tx: mpsc::Sender<()>, config: Arc<ArcSwap<Config>>, config_path: String) -> Result<Self> {
        Ok(Self { shutdown_tx, config, config_path })
    }

    /// Run tray event loop (blocks main thread)
    pub fn run(self, server_addr: String) -> Result<()> {
        if Self::is_headless() {
            tracing::info!("Headless environment detected, running without tray");
            Self::wait_for_ctrl_c();
            return Ok(());
        }

        // Wrap tray creation in catch_unwind because:
        // - GTK on Linux can panic via C FFI if display unavailable
        // - AppKit on macOS can panic in certain headless scenarios
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.run_inner(server_addr)
        })) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => {
                tracing::warn!("Tray failed to start: {}, running without tray", e);
                Self::wait_for_ctrl_c();
                Ok(())
            }
            Err(panic) => {
                tracing::warn!("Tray panicked: {:?}, running without tray", panic);
                Self::wait_for_ctrl_c();
                Ok(())
            }
        }
    }

    /// Inner tray implementation (separated for panic safety)
    fn run_inner(self, server_addr: String) -> Result<()> {
        let menu = Self::build_menu(&server_addr);
        let icon = Self::load_icon()?;

        let _tray_icon = TrayIconBuilder::new()
            .with_menu(Box::new(menu))
            .with_icon(icon)
            .with_tooltip(format!("Lumina Proxy - {}", server_addr))
            .build()
            .context("Failed to create tray icon")?;

        let event_loop = EventLoop::new();
        let shutdown_tx = self.shutdown_tx;
        let config = self.config;
        let config_path = self.config_path;

        // Menu item IDs
        let exit_id = "exit";
        let reload_id = "reload_config";
        let open_config_id = "open_config";
        let show_config_id = "show_config";

        event_loop.run(move |_event, _, control_flow| {
            *control_flow = ControlFlow::Wait;

            if let Ok(event) = MenuEvent::receiver().try_recv() {
                if event.id == exit_id {
                    let _ = shutdown_tx.try_send(());
                    *control_flow = ControlFlow::Exit;
                } else if event.id == reload_id {
                    // Reload configuration
                    match Config::load_and_validate(&config_path) {
                        Ok(new_config) => {
                            config.store(Arc::new(new_config));
                            tracing::info!("Configuration reloaded successfully via tray menu");
                            tracing::info!(
                                "Configuration reloaded successfully via tray menu. Note: Some changes may require server restart."
                            );
                        }
                        Err(e) => {
                            tracing::error!("Failed to reload configuration via tray: {}", e);
                        }
                    }
                } else if event.id == open_config_id {
                    // Open config file with default editor
                    Self::open_config_file(&config_path);
                } else if event.id == show_config_id {
                    // Show current configuration summary
                    let cfg = config.load();
                    let enabled_routes = cfg.routes.iter().filter(|r| r.enabled).count();
                    let summary = format!(
                        "Lumina Proxy Configuration\n\n\
                         Server: {}:{}\n\
                         Enabled models: {}\n\
                         Log level: {}\n\
                         Statistics: {}",
                        cfg.server.host,
                        cfg.server.port,
                        enabled_routes,
                        cfg.logging.level,
                        if cfg.statistics.enabled { "Enabled" } else { "Disabled" }
                    );
                    tracing::info!("Tray config summary requested\n{}", summary);
                }
            }
        });

        #[allow(unreachable_code)]
        Ok(())
    }

    /// Build tray menu with internationalized text
    fn build_menu(server_addr: &str) -> Menu {
        let lang = detect_system_language();
        let t = |key: &str, default: &str| -> String {
            match (lang, key) {
                ("zh", "status_running") => "🟢 服务器运行中".to_string(),
                ("en", "status_running") => "🟢  Server Running".to_string(),
                ("zh", "reload_config") => "重新加载配置".to_string(),
                ("en", "reload_config") => "Reload Configuration".to_string(),
                ("zh", "open_config") => "打开配置文件".to_string(),
                ("en", "open_config") => "Open Config File".to_string(),
                ("zh", "show_config") => "显示当前配置".to_string(),
                ("en", "show_config") => "Show Current Config".to_string(),
                ("zh", "exit") => "退出".to_string(),
                ("en", "exit") => "Quit".to_string(),
                (_, _) => default.to_string(),
            }
        };

        let status_text = format!("{}\n{}", t("status_running", "🟢  Running"), server_addr);

        let status_item = MenuItem::new(
            status_text,
            false,
            None,
        );

        let reload_item = MenuItem::with_id(
            "reload_config",
            t("reload_config", "Reload Configuration"),
            true,
            None,
        );

        let open_config_item = MenuItem::with_id(
            "open_config",
            t("open_config", "Open Config File"),
            true,
            None,
        );

        let show_config_item = MenuItem::with_id(
            "show_config",
            t("show_config", "Show Current Config"),
            true,
            None,
        );

        let exit_item = MenuItem::with_id(
            "exit",
            t("exit", "Exit"),
            true,
            None,
        );

        let menu = Menu::new();
        menu.append_items(&[
            &status_item,
            &PredefinedMenuItem::separator(),
            &reload_item,
            &open_config_item,
            &show_config_item,
            &PredefinedMenuItem::separator(),
            &exit_item,
        ]).unwrap();

        menu
    }

    /// Load tray icon from embedded resources (cross-platform)
    fn load_icon() -> Result<Icon> {
        static ICON_RASTER: &[u8] = include_bytes!("../assets/icon.rgba");
        const SIZE: u32 = 64;

        match Icon::from_rgba(ICON_RASTER.to_vec(), SIZE, SIZE) {
            Ok(icon) => Ok(icon),
            Err(_) => {
                tracing::warn!("Failed to load embedded icon, using fallback");
                let pixel: [u8; 4] = [0, 120, 215, 255];
                let rgba_data = std::iter::repeat(&pixel)
                    .take((SIZE * SIZE) as usize)
                    .flat_map(|p| p.iter().copied())
                    .collect::<Vec<u8>>();
                Icon::from_rgba(rgba_data, SIZE, SIZE)
                    .context("Failed to create fallback tray icon")
            }
        }
    }

    /// Platform-specific: Open config file with default editor
    #[cfg(windows)]
    fn open_config_file(path: &str) {
        let path = Path::new(path);
        if path.exists() {
            let result = std::process::Command::new("cmd")
                .args(&["/C", "start", "", path.to_str().unwrap_or_default()])
                .spawn();
            if let Err(e) = result {
                tracing::error!("Failed to open config file: {}", e);
            }
        } else {
            tracing::warn!("Config file not found: {}", path.display());
        }
    }

    #[cfg(target_os = "macos")]
    fn open_config_file(path: &str) {
        let path = Path::new(path);
        if path.exists() {
            let result = std::process::Command::new("open")
                .arg(path)
                .spawn();
            if let Err(e) = result {
                tracing::error!("Failed to open config file: {}", e);
            }
        } else {
            tracing::warn!("Config file not found: {}", path.display());
        }
    }

    #[cfg(target_os = "linux")]
    fn open_config_file(path: &str) {
        let path = Path::new(path);
        if path.exists() {
            let result = std::process::Command::new("xdg-open")
                .arg(path)
                .spawn();
            if let Err(e) = result {
                tracing::error!("Failed to open config file: {}", e);
            }
        } else {
            tracing::warn!("Config file not found: {}", path.display());
        }
    }

    /// Platform-specific: Detect if running in headless environment
    #[cfg(windows)]
    fn is_headless() -> bool {
        std::env::var("CI").is_ok()
    }

    #[cfg(target_os = "macos")]
    fn is_headless() -> bool {
        std::env::var("SSH_CONNECTION").is_ok() || std::env::var("CI").is_ok()
    }

    #[cfg(target_os = "linux")]
    fn is_headless() -> bool {
        std::env::var("SSH_CONNECTION").is_ok()
            || std::env::var("CI").is_ok()
            || (std::env::var("DISPLAY").is_err() && std::env::var("WAYLAND_DISPLAY").is_err())
    }

    /// Wait for Ctrl+C signal (keeps main thread alive when tray is unavailable)
    fn wait_for_ctrl_c() {
        tokio::runtime::Runtime::new()
            .expect("Failed to create runtime for Ctrl+C wait")
            .block_on(async {
                tokio::signal::ctrl_c()
                    .await
                    .expect("Failed to listen for ctrl+c");
            });
    }
}
