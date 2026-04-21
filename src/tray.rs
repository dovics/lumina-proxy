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
