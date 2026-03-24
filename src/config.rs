use std::collections::HashSet;
use std::env;
use std::fmt;

/// Controls how the embedding tracker starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackerMode {
    Off,
    Lazy,
    Eager,
}

impl fmt::Display for TrackerMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrackerMode::Off => write!(f, "off"),
            TrackerMode::Lazy => write!(f, "lazy"),
            TrackerMode::Eager => write!(f, "eager"),
        }
    }
}

/// Parse a string into a TrackerMode.
pub fn parse_tracker_mode(value: Option<&str>) -> TrackerMode {
    match value.map(|v| v.trim().to_lowercase()).as_deref() {
        Some("off") | Some("false") | Some("0") | Some("no") | Some("disabled") | Some("none") => {
            TrackerMode::Off
        }
        Some("eager") | Some("startup") | Some("boot") => TrackerMode::Eager,
        _ => TrackerMode::Lazy,
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub ollama_host: String,
    pub ollama_embed_model: String,
    pub ollama_chat_model: String,
    pub ollama_api_key: Option<String>,
    pub embed_batch_size: usize,
    pub embed_tracker_mode: TrackerMode,
    pub embed_tracker_debounce_ms: u64,
    pub embed_tracker_max_files: usize,
    pub ignore_dirs: HashSet<String>,
    pub cache_ttl_secs: u64,
    pub max_embed_file_size: usize,
    pub embed_num_gpu: Option<i32>,
    pub embed_main_gpu: Option<i32>,
    pub embed_num_thread: Option<i32>,
    pub embed_num_batch: Option<i32>,
    pub embed_num_ctx: Option<i32>,
    pub embed_low_vram: Option<bool>,
    pub idle_timeout_ms: u64,
    pub parent_poll_ms: u64,
    pub embed_chunk_chars: usize,
}

const DEFAULT_OLLAMA_HOST: &str = "http://localhost:11434";
const DEFAULT_EMBED_MODEL: &str = "snowflake-arctic-embed2";
const DEFAULT_CHAT_MODEL: &str = "llama3.2";
const DEFAULT_EMBED_BATCH_SIZE: usize = 50;
const DEFAULT_EMBED_TRACKER_DEBOUNCE_MS: u64 = 700;
const DEFAULT_EMBED_TRACKER_MAX_FILES: usize = 8;
const DEFAULT_CACHE_TTL_SECS: u64 = 300;
const DEFAULT_EMBED_CHUNK_CHARS: usize = 2000;
const MIN_EMBED_CHUNK_CHARS: usize = 256;
const MAX_EMBED_CHUNK_CHARS: usize = 8000;
const MIN_EMBED_BATCH_SIZE: usize = 5;
const MAX_EMBED_BATCH_SIZE: usize = 512;
const DEFAULT_MAX_EMBED_FILE_SIZE: usize = 50 * 1024;
const MIN_MAX_EMBED_FILE_SIZE: usize = 1024;

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

fn env_opt<T: std::str::FromStr>(key: &str) -> Option<T> {
    env::var(key).ok().and_then(|v| v.parse().ok())
}
fn env_opt_bool(key: &str) -> Option<bool> {
    match env::var(key).ok().as_deref() {
        Some("true") | Some("1") | Some("yes") => Some(true),
        Some("false") | Some("0") | Some("no") => Some(false),
        _ => None,
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
            embed_tracker_mode: parse_tracker_mode(
                env::var("CONTEXTPLUS_EMBED_TRACKER").ok().as_deref(),
            ),
            embed_tracker_debounce_ms: env_parse(
                "CONTEXTPLUS_EMBED_TRACKER_DEBOUNCE_MS",
                DEFAULT_EMBED_TRACKER_DEBOUNCE_MS,
            ),
            embed_tracker_max_files: env_parse(
                "CONTEXTPLUS_EMBED_TRACKER_MAX_FILES",
                DEFAULT_EMBED_TRACKER_MAX_FILES,
            ),
            ignore_dirs: build_ignore_dirs(),
            cache_ttl_secs: env_parse("CONTEXTPLUS_CACHE_TTL_SECS", DEFAULT_CACHE_TTL_SECS),
            max_embed_file_size: env_parse(
                "CONTEXTPLUS_MAX_EMBED_FILE_SIZE",
                DEFAULT_MAX_EMBED_FILE_SIZE,
            )
            .max(MIN_MAX_EMBED_FILE_SIZE),
            embed_num_gpu: env_opt("CONTEXTPLUS_EMBED_NUM_GPU"),
            embed_main_gpu: env_opt("CONTEXTPLUS_EMBED_MAIN_GPU"),
            embed_num_thread: env_opt("CONTEXTPLUS_EMBED_NUM_THREAD"),
            embed_num_batch: env_opt("CONTEXTPLUS_EMBED_NUM_BATCH"),
            embed_num_ctx: env_opt("CONTEXTPLUS_EMBED_NUM_CTX"),
            embed_low_vram: env_opt_bool("CONTEXTPLUS_EMBED_LOW_VRAM"),
            idle_timeout_ms: crate::core::process_lifecycle::get_idle_shutdown_ms(
                env::var("CONTEXTPLUS_IDLE_TIMEOUT_MS").ok().as_deref(),
            ),
            parent_poll_ms: crate::core::process_lifecycle::get_parent_poll_ms(
                env::var("CONTEXTPLUS_PARENT_POLL_MS").ok().as_deref(),
            ),
            embed_chunk_chars: env_parse(
                "CONTEXTPLUS_EMBED_CHUNK_CHARS",
                DEFAULT_EMBED_CHUNK_CHARS,
            )
            .clamp(MIN_EMBED_CHUNK_CHARS, MAX_EMBED_CHUNK_CHARS),
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
                "CONTEXTPLUS_CACHE_TTL_SECS",
                "CONTEXTPLUS_IDLE_TIMEOUT_MS",
                "CONTEXTPLUS_PARENT_POLL_MS",
                "CONTEXTPLUS_EMBED_CHUNK_CHARS",
            ],
            || {
                let cfg = Config::from_env();
                assert_eq!(cfg.ollama_host, "http://localhost:11434");
                assert_eq!(cfg.ollama_embed_model, "snowflake-arctic-embed2");
                assert_eq!(cfg.ollama_chat_model, "llama3.2");
                assert!(cfg.ollama_api_key.is_none());
                assert_eq!(cfg.embed_batch_size, 50);
                assert_eq!(cfg.embed_tracker_mode, TrackerMode::Lazy);
                assert_eq!(cfg.embed_tracker_debounce_ms, 700);
                assert_eq!(cfg.embed_tracker_max_files, 8);
                assert!(cfg.ignore_dirs.contains("node_modules"));
                assert!(cfg.ignore_dirs.contains(".git"));
                assert!(cfg.ignore_dirs.contains("target"));
                assert_eq!(cfg.cache_ttl_secs, 300);
                assert_eq!(cfg.idle_timeout_ms, 900_000); // 15 min default
                assert_eq!(cfg.parent_poll_ms, 5_000); // 5s default
                assert_eq!(cfg.embed_chunk_chars, 2000);
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
                assert_eq!(cfg.embed_tracker_mode, TrackerMode::Off);
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
    fn cache_ttl_secs_default() {
        with_cleared_env(&["CONTEXTPLUS_CACHE_TTL_SECS"], || {
            let cfg = Config::from_env();
            assert_eq!(cfg.cache_ttl_secs, 300);
        });
    }

    #[test]
    fn cache_ttl_secs_custom() {
        with_env(&[("CONTEXTPLUS_CACHE_TTL_SECS", "600")], || {
            let cfg = Config::from_env();
            assert_eq!(cfg.cache_ttl_secs, 600);
        });
    }

    #[test]
    fn ollama_chat_model_default() {
        with_cleared_env(&["OLLAMA_CHAT_MODEL"], || {
            let cfg = Config::from_env();
            assert_eq!(cfg.ollama_chat_model, "llama3.2");
        });
    }

    #[test]
    fn ollama_chat_model_custom() {
        with_env(&[("OLLAMA_CHAT_MODEL", "qwen3:8b")], || {
            let cfg = Config::from_env();
            assert_eq!(cfg.ollama_chat_model, "qwen3:8b");
        });
    }

    #[test]
    fn tracker_mode_variants() {
        for val in &["off", "false", "0", "no", "disabled", "none"] {
            with_env(&[("CONTEXTPLUS_EMBED_TRACKER", val)], || {
                assert_eq!(
                    Config::from_env().embed_tracker_mode,
                    TrackerMode::Off,
                    "expected Off for '{val}'"
                );
            });
        }
        for val in &["eager", "startup", "boot"] {
            with_env(&[("CONTEXTPLUS_EMBED_TRACKER", val)], || {
                assert_eq!(
                    Config::from_env().embed_tracker_mode,
                    TrackerMode::Eager,
                    "expected Eager for '{val}'"
                );
            });
        }
        for val in &["true", "1", "yes", "lazy"] {
            with_env(&[("CONTEXTPLUS_EMBED_TRACKER", val)], || {
                assert_eq!(
                    Config::from_env().embed_tracker_mode,
                    TrackerMode::Lazy,
                    "expected Lazy for '{val}'"
                );
            });
        }
    }

    #[test]
    fn parse_tracker_mode_unit() {
        assert_eq!(parse_tracker_mode(None), TrackerMode::Lazy);
        assert_eq!(parse_tracker_mode(Some("")), TrackerMode::Lazy);
        assert_eq!(parse_tracker_mode(Some("off")), TrackerMode::Off);
        assert_eq!(parse_tracker_mode(Some("OFF")), TrackerMode::Off);
        assert_eq!(parse_tracker_mode(Some(" False ")), TrackerMode::Off);
        assert_eq!(parse_tracker_mode(Some("eager")), TrackerMode::Eager);
        assert_eq!(parse_tracker_mode(Some("BOOT")), TrackerMode::Eager);
        assert_eq!(parse_tracker_mode(Some("startup")), TrackerMode::Eager);
        assert_eq!(parse_tracker_mode(Some("anything_else")), TrackerMode::Lazy);
    }

    #[test]
    fn idle_timeout_disabled() {
        with_env(&[("CONTEXTPLUS_IDLE_TIMEOUT_MS", "0")], || {
            let cfg = Config::from_env();
            assert_eq!(cfg.idle_timeout_ms, 0);
        });
        with_env(&[("CONTEXTPLUS_IDLE_TIMEOUT_MS", "off")], || {
            let cfg = Config::from_env();
            assert_eq!(cfg.idle_timeout_ms, 0);
        });
    }

    #[test]
    fn idle_timeout_custom() {
        with_env(&[("CONTEXTPLUS_IDLE_TIMEOUT_MS", "120000")], || {
            let cfg = Config::from_env();
            assert_eq!(cfg.idle_timeout_ms, 120_000);
        });
    }

    #[test]
    fn idle_timeout_clamps_to_min() {
        with_env(&[("CONTEXTPLUS_IDLE_TIMEOUT_MS", "5000")], || {
            let cfg = Config::from_env();
            assert_eq!(cfg.idle_timeout_ms, 60_000); // MIN_IDLE_TIMEOUT_MS
        });
    }

    #[test]
    fn parent_poll_custom() {
        with_env(&[("CONTEXTPLUS_PARENT_POLL_MS", "3000")], || {
            let cfg = Config::from_env();
            assert_eq!(cfg.parent_poll_ms, 3_000);
        });
    }

    #[test]
    fn parent_poll_clamps_to_min() {
        with_env(&[("CONTEXTPLUS_PARENT_POLL_MS", "500")], || {
            let cfg = Config::from_env();
            assert_eq!(cfg.parent_poll_ms, 1_000); // MIN_PARENT_POLL_MS
        });
    }

    #[test]
    fn embed_chunk_chars_clamps() {
        with_env(&[("CONTEXTPLUS_EMBED_CHUNK_CHARS", "100")], || {
            let cfg = Config::from_env();
            assert_eq!(cfg.embed_chunk_chars, MIN_EMBED_CHUNK_CHARS);
        });
        with_env(&[("CONTEXTPLUS_EMBED_CHUNK_CHARS", "99999")], || {
            let cfg = Config::from_env();
            assert_eq!(cfg.embed_chunk_chars, MAX_EMBED_CHUNK_CHARS);
        });
    }

    #[test]
    fn embed_chunk_chars_custom() {
        with_env(&[("CONTEXTPLUS_EMBED_CHUNK_CHARS", "4000")], || {
            let cfg = Config::from_env();
            assert_eq!(cfg.embed_chunk_chars, 4000);
        });
    }

    #[test]
    fn max_file_size_default() {
        with_cleared_env(&["CONTEXTPLUS_MAX_EMBED_FILE_SIZE"], || {
            assert_eq!(Config::from_env().max_embed_file_size, 50 * 1024);
        });
    }
    #[test]
    fn max_file_size_custom() {
        with_env(&[("CONTEXTPLUS_MAX_EMBED_FILE_SIZE", "102400")], || {
            assert_eq!(Config::from_env().max_embed_file_size, 102400);
        });
    }
    #[test]
    fn max_file_size_clamps() {
        with_env(&[("CONTEXTPLUS_MAX_EMBED_FILE_SIZE", "100")], || {
            assert_eq!(
                Config::from_env().max_embed_file_size,
                MIN_MAX_EMBED_FILE_SIZE
            );
        });
    }
    #[test]
    fn gpu_opts_none() {
        with_cleared_env(
            &[
                "CONTEXTPLUS_EMBED_NUM_GPU",
                "CONTEXTPLUS_EMBED_MAIN_GPU",
                "CONTEXTPLUS_EMBED_NUM_THREAD",
                "CONTEXTPLUS_EMBED_NUM_BATCH",
                "CONTEXTPLUS_EMBED_NUM_CTX",
                "CONTEXTPLUS_EMBED_LOW_VRAM",
            ],
            || {
                let c = Config::from_env();
                assert!(
                    c.embed_num_gpu.is_none()
                        && c.embed_main_gpu.is_none()
                        && c.embed_num_thread.is_none()
                        && c.embed_num_batch.is_none()
                        && c.embed_num_ctx.is_none()
                        && c.embed_low_vram.is_none()
                );
            },
        );
    }
    #[test]
    fn gpu_opts_set() {
        with_env(
            &[
                ("CONTEXTPLUS_EMBED_NUM_GPU", "1"),
                ("CONTEXTPLUS_EMBED_MAIN_GPU", "0"),
                ("CONTEXTPLUS_EMBED_NUM_THREAD", "4"),
                ("CONTEXTPLUS_EMBED_NUM_BATCH", "512"),
                ("CONTEXTPLUS_EMBED_NUM_CTX", "2048"),
                ("CONTEXTPLUS_EMBED_LOW_VRAM", "true"),
            ],
            || {
                let c = Config::from_env();
                assert_eq!(c.embed_num_gpu, Some(1));
                assert_eq!(c.embed_main_gpu, Some(0));
                assert_eq!(c.embed_num_thread, Some(4));
                assert_eq!(c.embed_num_batch, Some(512));
                assert_eq!(c.embed_num_ctx, Some(2048));
                assert_eq!(c.embed_low_vram, Some(true));
            },
        );
    }
    #[test]
    fn gpu_opts_invalid() {
        with_env(
            &[
                ("CONTEXTPLUS_EMBED_NUM_GPU", "x"),
                ("CONTEXTPLUS_EMBED_LOW_VRAM", "maybe"),
            ],
            || {
                let c = Config::from_env();
                assert!(c.embed_num_gpu.is_none() && c.embed_low_vram.is_none());
            },
        );
    }
}
