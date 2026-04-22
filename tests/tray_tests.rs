#[cfg(test)]
mod tests {
    use std::env;

    fn clear_env_vars() {
        for key in &["LANG", "LC_ALL", "LC_MESSAGES", "USERLANGUAGE"] {
            unsafe { env::remove_var(key); }
        }
    }

    // NOTE: These tests are marked ignore because:
    // 1. Environment variables are process-global
    // 2. Tests run in parallel, causing race conditions
    // 3. Run them individually: cargo test --test tray_tests -- --ignored --exact <test_name>

    #[test]
    #[ignore = "env vars are process-global, run manually"]
    fn test_language_detection_zh() {
        clear_env_vars();
        unsafe { env::set_var("LANG", "zh_CN.UTF-8"); }
        assert_eq!(lumina::tray::detect_system_language(), "zh");
    }

    #[test]
    #[ignore = "env vars are process-global, run manually"]
    fn test_language_detection_en() {
        clear_env_vars();
        unsafe { env::set_var("LANG", "en_US.UTF-8"); }
        assert_eq!(lumina::tray::detect_system_language(), "en");
    }

    #[test]
    #[ignore = "env vars are process-global, run manually"]
    fn test_language_detection_lc_all_override() {
        clear_env_vars();
        unsafe {
            env::set_var("LANG", "zh_CN.UTF-8");
            env::set_var("LC_ALL", "en_US.UTF-8");
        }
        // LC_ALL is checked first in priority order
        assert_eq!(lumina::tray::detect_system_language(), "en");
    }
}
