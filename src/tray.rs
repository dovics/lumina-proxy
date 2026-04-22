//! System Tray Module
//! Provides Windows system tray functionality for graceful shutdown

use anyhow::Result;

#[cfg(windows)]
pub mod imp {
    use super::*;
    use anyhow::Context;
    use tao::event_loop::{EventLoop, ControlFlow};
    use tokio::sync::mpsc;
    use tray_icon::{
        menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem},
        TrayIconBuilder, Icon,
    };

    pub struct TrayManager {
        shutdown_tx: mpsc::Sender<()>,
    }

    impl TrayManager {
        pub fn new(shutdown_tx: mpsc::Sender<()>) -> Result<Self> {
            Ok(Self { shutdown_tx })
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
            let exit_item_id = "exit";

            event_loop.run(move |_event, _, control_flow| {
                *control_flow = ControlFlow::Wait;

                if let Ok(event) = MenuEvent::receiver().try_recv() {
                    if event.id == exit_item_id {
                        let _ = shutdown_tx.try_send(());
                        *control_flow = ControlFlow::Exit;
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
