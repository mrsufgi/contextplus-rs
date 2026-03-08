use std::collections::HashSet;
use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    pub ollama_host: String,
    pub ollama_embed_model: String,
    pub ollama_chat_model: String,
    pub ollama_api_key: Option<String>,
    pub embed_batch_size: usize,
    pub embed_tracker_enabled: bool,
    pub embed_tracker_debounce_ms: u64,
    pub embed_tracker_max_files: usize,
    pub ignore_dirs: HashSet<String>,
}

const DEFAULT_OLLAMA_HOST: &str = "http://localhost:11434";
const DEFAULT_EMBED_MODEL: &str = "snowflake-arctic-embed2";
const DEFAULT_CHAT_MODEL: &str = "llama3.2";
const DEFAULT_EMBED_BATCH_SIZE: usize = 50;
const DEFAULT_EMBED_TRACKER_DEBOUNCE_MS: u64 = 700;
const DEFAULT_EMBED_TRACKER_MAX_FILES: usize = 8;
const MIN_EMBED_BATCH_SIZE: usize = 5;
const MAX_EMBED_BATCH_SIZE: usize = 512;

const BASE_IGNORE_DIRS: &[&str] = &[
    "node_modules",
    ".git",
    "dist",
    "build",
    ".next",
    "target",
    ".svn",
    ".hg",
    "__pycache__",
    ".DS_Store",
    ".nuxt",
    ".mcp_data",
    ".mcp-shadow-history",
    "coverage",
    ".cache",
    ".turbo",
    ".parcel-cache",
];

fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_bool(key: &str, default: bool) -> bool {
    match env::var(key).ok().as_deref() {
        Some("true") | Some("1") | Some("yes") => true,
        Some("false") | Some("0") | Some("no") => false,
        _ => default,
    }
}

fn build_ignore_dirs() -> HashSet<String> {
    let mut dirs: HashSet<String> = BASE_IGNORE_DIRS.iter().map(|s| (*s).to_string()).collect();
    if let Ok(extra) = env::var("CONTEXTPLUS_IGNORE_DIRS") {
        for dir in extra.split(',') {
            let trimmed = dir.trim();
            if !trimmed.is_empty() {
                dirs.insert(trimmed.to_string());
            }
        }
    }
    dirs
}

impl Config {
    pub fn from_env() -> Self {
        let batch_size: usize = env_parse("CONTEXTPLUS_EMBED_BATCH_SIZE", DEFAULT_EMBED_BATCH_SIZE);
        let batch_size = batch_size.clamp(MIN_EMBED_BATCH_SIZE, MAX_EMBED_BATCH_SIZE);

        Config {
            ollama_host: env_or("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
            ollama_embed_model: env_or("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL),
            ollama_chat_model: env_or("OLLAMA_CHAT_MODEL", DEFAULT_CHAT_MODEL),
            ollama_api_key: env::var("OLLAMA_API_KEY").ok(),
            embed_batch_size: batch_size,
            embed_tracker_enabled: env_bool("CONTEXTPLUS_EMBED_TRACKER", true),
            embed_tracker_debounce_ms: env_parse(
                "CONTEXTPLUS_EMBED_TRACKER_DEBOUNCE_MS",
                DEFAULT_EMBED_TRACKER_DEBOUNCE_MS,
            ),
            embed_tracker_max_files: env_parse(
                "CONTEXTPLUS_EMBED_TRACKER_MAX_FILES",
                DEFAULT_EMBED_TRACKER_MAX_FILES,
            ),
            ignore_dirs: build_ignore_dirs(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Env var tests must be serialized to avoid races
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_env<F: FnOnce()>(vars: &[(&str, &str)], f: F) {
        let _guard = ENV_LOCK.lock().unwrap();
        let mut old_vals: Vec<(&str, Option<String>)> = Vec::new();
        for (key, val) in vars {
            old_vals.push((key, env::var(key).ok()));
            // SAFETY: test-only, serialized by ENV_LOCK
            unsafe { env::set_var(key, val) };
        }
        f();
        for (key, old_val) in old_vals {
            unsafe {
                match old_val {
                    Some(v) => env::set_var(key, v),
                    None => env::remove_var(key),
                }
            }
        }
    }

    fn with_cleared_env<F: FnOnce()>(keys: &[&str], f: F) {
        let _guard = ENV_LOCK.lock().unwrap();
        let mut old_vals: Vec<(&str, Option<String>)> = Vec::new();
        for key in keys {
            old_vals.push((key, env::var(key).ok()));
            // SAFETY: test-only, serialized by ENV_LOCK
            unsafe { env::remove_var(key) };
        }
        f();
        for (key, old_val) in old_vals {
            if let Some(v) = old_val {
                // SAFETY: test-only, serialized by ENV_LOCK
                unsafe { env::set_var(key, v) };
            }
        }
    }

    #[test]
    fn defaults_are_correct() {
        with_cleared_env(
            &[
                "OLLAMA_HOST",
                "OLLAMA_EMBED_MODEL",
                "OLLAMA_CHAT_MODEL",
                "OLLAMA_API_KEY",
                "CONTEXTPLUS_EMBED_BATCH_SIZE",
                "CONTEXTPLUS_EMBED_TRACKER",
                "CONTEXTPLUS_EMBED_TRACKER_DEBOUNCE_MS",
                "CONTEXTPLUS_EMBED_TRACKER_MAX_FILES",
                "CONTEXTPLUS_IGNORE_DIRS",
            ],
            || {
                let cfg = Config::from_env();
                assert_eq!(cfg.ollama_host, "http://localhost:11434");
                assert_eq!(cfg.ollama_embed_model, "snowflake-arctic-embed2");
                assert_eq!(cfg.ollama_chat_model, "llama3.2");
                assert!(cfg.ollama_api_key.is_none());
                assert_eq!(cfg.embed_batch_size, 50);
                assert!(cfg.embed_tracker_enabled);
                assert_eq!(cfg.embed_tracker_debounce_ms, 700);
                assert_eq!(cfg.embed_tracker_max_files, 8);
                assert!(cfg.ignore_dirs.contains("node_modules"));
                assert!(cfg.ignore_dirs.contains(".git"));
                assert!(cfg.ignore_dirs.contains("target"));
            },
        );
    }

    #[test]
    fn env_overrides() {
        with_env(
            &[
                ("OLLAMA_HOST", "http://my-host:11434"),
                ("OLLAMA_EMBED_MODEL", "my-model"),
                ("OLLAMA_CHAT_MODEL", "my-chat"),
                ("OLLAMA_API_KEY", "secret-key"),
                ("CONTEXTPLUS_EMBED_BATCH_SIZE", "100"),
                ("CONTEXTPLUS_EMBED_TRACKER", "false"),
                ("CONTEXTPLUS_EMBED_TRACKER_DEBOUNCE_MS", "1500"),
                ("CONTEXTPLUS_EMBED_TRACKER_MAX_FILES", "20"),
            ],
            || {
                let cfg = Config::from_env();
                assert_eq!(cfg.ollama_host, "http://my-host:11434");
                assert_eq!(cfg.ollama_embed_model, "my-model");
                assert_eq!(cfg.ollama_chat_model, "my-chat");
                assert_eq!(cfg.ollama_api_key.as_deref(), Some("secret-key"));
                assert_eq!(cfg.embed_batch_size, 100);
                assert!(!cfg.embed_tracker_enabled);
                assert_eq!(cfg.embed_tracker_debounce_ms, 1500);
                assert_eq!(cfg.embed_tracker_max_files, 20);
            },
        );
    }

    #[test]
    fn batch_size_clamps() {
        with_env(&[("CONTEXTPLUS_EMBED_BATCH_SIZE", "2")], || {
            let cfg = Config::from_env();
            assert_eq!(cfg.embed_batch_size, MIN_EMBED_BATCH_SIZE);
        });
        with_env(&[("CONTEXTPLUS_EMBED_BATCH_SIZE", "9999")], || {
            let cfg = Config::from_env();
            assert_eq!(cfg.embed_batch_size, MAX_EMBED_BATCH_SIZE);
        });
    }

    #[test]
    fn invalid_batch_size_uses_default() {
        with_env(&[("CONTEXTPLUS_EMBED_BATCH_SIZE", "not_a_number")], || {
            let cfg = Config::from_env();
            assert_eq!(cfg.embed_batch_size, DEFAULT_EMBED_BATCH_SIZE);
        });
    }

    #[test]
    fn ignore_dirs_includes_extra() {
        with_env(&[("CONTEXTPLUS_IGNORE_DIRS", "vendor, .idea , ")], || {
            let cfg = Config::from_env();
            assert!(cfg.ignore_dirs.contains("vendor"));
            assert!(cfg.ignore_dirs.contains(".idea"));
            // base dirs still present
            assert!(cfg.ignore_dirs.contains("node_modules"));
        });
    }

    #[test]
    fn tracker_bool_variants() {
        with_env(&[("CONTEXTPLUS_EMBED_TRACKER", "1")], || {
            assert!(Config::from_env().embed_tracker_enabled);
        });
        with_env(&[("CONTEXTPLUS_EMBED_TRACKER", "yes")], || {
            assert!(Config::from_env().embed_tracker_enabled);
        });
        with_env(&[("CONTEXTPLUS_EMBED_TRACKER", "0")], || {
            assert!(!Config::from_env().embed_tracker_enabled);
        });
        with_env(&[("CONTEXTPLUS_EMBED_TRACKER", "no")], || {
            assert!(!Config::from_env().embed_tracker_enabled);
        });
    }
}
