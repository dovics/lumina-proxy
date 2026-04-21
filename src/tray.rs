//! System Tray Module
//! Provides Windows system tray functionality for graceful shutdown

use anyhow::Result;

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
