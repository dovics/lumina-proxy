//! System Tray Module
//! Provides Windows system tray functionality for graceful shutdown and config management

use anyhow::Result;

#[cfg(windows)]
pub mod imp {
    use super::*;
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

    impl TrayManager {
        pub fn new(shutdown_tx: mpsc::Sender<()>, config: Arc<ArcSwap<Config>>, config_path: String) -> Result<Self> {
            Ok(Self { shutdown_tx, config, config_path })
        }

        pub fn run(self, server_addr: String) -> Result<()> {
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
                        let path = Path::new(&config_path);
                        if path.exists() {
                            // Use ShellExecute to open the file
                            let result = std::process::Command::new("cmd")
                                .args(&["/C", "start", "", config_path.as_str()])
                                .spawn();
                            if let Err(e) = result {
                                tracing::error!("Failed to open config file: {}", e);
                            }
                        } else {
                            tracing::warn!("Config file not found: {}", config_path);
                        }
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

        fn build_menu(server_addr: &str) -> Menu {
            let status_item = MenuItem::new(
                format!("🟢 服务器运行中\n{}", server_addr),
                false,
                None,
            );

            let reload_item = MenuItem::with_id(
                "reload_config",
                "Reload Configuration",
                true,
                None,
            );

            let open_config_item = MenuItem::with_id(
                "open_config",
                "Open Config File",
                true,
                None,
            );

            let show_config_item = MenuItem::with_id(
                "show_config",
                "Show Current Config",
                true,
                None,
            );

            let exit_item = MenuItem::with_id(
                "exit",
                "退出",
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
