use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand, ValueEnum};
use contextplus_rs::config::Config;
use contextplus_rs::core::process_lifecycle;
use contextplus_rs::server::ContextPlusServer;
use rmcp::ServiceExt;

#[derive(Parser)]
#[command(
    name = "contextplus-rs",
    version,
    about = "Context+ MCP server for semantic code analysis"
)]
struct Cli {
    /// Root directory to analyze (defaults to current directory)
    #[arg(long, global = true)]
    root_dir: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Clone, Copy, ValueEnum)]
enum AgentTarget {
    Claude,
    Cursor,
    Vscode,
    Windsurf,
    Opencode,
}

impl AgentTarget {
    fn config_path(self) -> &'static str {
        match self {
            Self::Claude => ".mcp.json",
            Self::Cursor => ".cursor/mcp.json",
            Self::Vscode => ".vscode/mcp.json",
            Self::Windsurf => ".windsurf/mcp.json",
            Self::Opencode => "opencode.json",
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Start the MCP server over stdio (legacy, still supported).
    Serve,
    /// Run as a per-workspace daemon listening on a Unix socket.
    Daemon,
    /// Run as a thin stdio<->socket proxy, spawning a daemon if absent.
    Client,
    /// Generate MCP config file for a coding agent
    Init {
        #[arg(value_enum, default_value = "claude")]
        target: AgentTarget,
    },
    /// Print file skeleton for a file
    Skeleton { file: String },
    /// Print context tree for a directory
    Tree {
        #[arg(long)]
        max_tokens: Option<usize>,
    },
    /// Manage git hooks that nudge contextplus-rs on commit/checkout/merge
    Hooks {
        #[command(subcommand)]
        action: HooksAction,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TransportMode {
    Daemon,
    Stdio,
    Auto,
}

const TRANSPORT_ENV: &str = "CONTEXTPLUS_TRANSPORT";

fn resolve_transport_mode() -> TransportMode {
    match std::env::var(TRANSPORT_ENV)
        .ok()
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("stdio") => TransportMode::Stdio,
        Some("daemon") => TransportMode::Daemon,
        Some("auto") | None | Some("") => TransportMode::Auto,
        Some(other) => {
            tracing::warn!("unknown {TRANSPORT_ENV}={other:?}; falling back to auto");
            TransportMode::Auto
        }
    }
}

fn mcp_data_writable(root_dir: &std::path::Path) -> bool {
    let dir = root_dir.join(contextplus_rs::transport::paths::MCP_DATA_DIR);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::debug!(
            "mcp_data not writable: create_dir_all({}) failed: {e}",
            dir.display()
        );
        return false;
    }
    let probe = dir.join(".write-probe");
    match std::fs::write(&probe, b"") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(e) => {
            tracing::debug!("mcp_data not writable: probe failed: {e}");
            false
        }
    }
}

#[derive(Subcommand)]
enum HooksAction {
    Install,
    Uninstall,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("contextplus_rs=info".parse()?),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let root_dir = cli
        .root_dir
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));

    let config = Config::from_env();

    match cli.command {
        None => {
            let mode = resolve_transport_mode();
            run_implicit_default(mode, root_dir, config).await?;
        }
        Some(Commands::Serve) => {
            run_mcp_server(root_dir, config).await?;
        }
        Some(Commands::Daemon) => {
            let owner =
                contextplus_rs::transport::daemon::run_if_owner(root_dir.clone(), config).await?;
            if !owner {
                tracing::info!("another daemon already owns this workspace -- exiting");
            }
        }
        Some(Commands::Client) => {
            contextplus_rs::transport::client::run(&root_dir).await?;
        }
        Some(Commands::Init { target }) => {
            let binary_path = std::env::current_exe()
                .and_then(|p| p.canonicalize())
                .unwrap_or_else(|_| PathBuf::from("contextplus-rs"));

            let content = match target {
                AgentTarget::Opencode => serde_json::to_string_pretty(&serde_json::json!({
                    "$schema": "https://opencode.ai/config.json",
                    "mcp": {
                        "contextplus": {
                            "type": "local",
                            "command": [binary_path.to_string_lossy()],
                            "enabled": true,
                            "environment": {
                                "OLLAMA_EMBED_MODEL": "snowflake-arctic-embed2",
                                "OLLAMA_CHAT_MODEL": "llama3.2",
                                "OLLAMA_HOST": "http://localhost:11434",
                                "CONTEXTPLUS_EMBED_BATCH_SIZE": "256",
                                "CONTEXTPLUS_EMBED_TRACKER": "true"
                            }
                        }
                    }
                }))?,
                _ => serde_json::to_string_pretty(&serde_json::json!({
                    "mcpServers": {
                        "contextplus": {
                            "command": binary_path.to_string_lossy(),
                            "args": [],
                            "env": {
                                "OLLAMA_EMBED_MODEL": "snowflake-arctic-embed2",
                                "OLLAMA_CHAT_MODEL": "llama3.2",
                                "OLLAMA_HOST": "http://localhost:11434",
                                "CONTEXTPLUS_EMBED_BATCH_SIZE": "256",
                                "CONTEXTPLUS_EMBED_TRACKER": "true"
                            }
                        }
                    }
                }))?,
            };

            let output_path = root_dir.join(target.config_path());
            if let Some(parent) = output_path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            tokio::fs::write(&output_path, format!("{content}\n")).await?;
            let target_name = match target {
                AgentTarget::Claude => "claude",
                AgentTarget::Cursor => "cursor",
                AgentTarget::Vscode => "vscode",
                AgentTarget::Windsurf => "windsurf",
                AgentTarget::Opencode => "opencode",
            };
            eprintln!("Context+ initialized for {target_name}.");
            eprintln!("Wrote MCP config: {}", output_path.display());
        }
        Some(Commands::Skeleton { file }) => {
            use contextplus_rs::tools::file_skeleton::{
                SkeletonAnalysis, SkeletonOptions, SkeletonSymbol,
            };

            let full_path = root_dir.join(&file);
            let content = tokio::fs::read_to_string(&full_path).await?;
            let ext = file.rsplit('.').next().unwrap_or("");
            let code_symbols =
                contextplus_rs::core::tree_sitter::parse_with_tree_sitter(&content, ext)
                    .unwrap_or_default();

            fn code_to_skel(s: &contextplus_rs::core::parser::CodeSymbol) -> SkeletonSymbol {
                SkeletonSymbol {
                    name: s.name.clone(),
                    kind: s.kind.clone(),
                    line: s.line,
                    end_line: s.end_line,
                    signature: s.signature.clone().unwrap_or_default(),
                    children: s.children.iter().map(code_to_skel).collect(),
                }
            }

            let header = contextplus_rs::core::parser::extract_header(&content);
            let analysis = SkeletonAnalysis {
                header: if header.is_empty() {
                    None
                } else {
                    Some(header)
                },
                symbols: code_symbols.iter().map(code_to_skel).collect(),
                line_count: content.lines().count(),
            };
            let options = SkeletonOptions {
                file_path: file,
                root_dir: root_dir.clone(),
            };
            let result = contextplus_rs::tools::file_skeleton::get_file_skeleton(
                options,
                Some(&analysis),
                Some(&content),
            )
            .await?;
            println!("{}", result);
        }
        Some(Commands::Tree { max_tokens }) => {
            use contextplus_rs::tools::context_tree;

            let walker_entries = contextplus_rs::core::walker::walk_with_config(&root_dir, &config);

            let entries: Vec<context_tree::FileEntry> = walker_entries
                .iter()
                .map(|e| context_tree::FileEntry {
                    relative_path: e.relative_path.clone(),
                    is_directory: e.is_directory,
                    depth: e.depth,
                })
                .collect();

            fn code_to_tree(
                s: &contextplus_rs::core::parser::CodeSymbol,
            ) -> context_tree::TreeSymbol {
                context_tree::TreeSymbol {
                    name: s.name.clone(),
                    kind: s.kind.clone(),
                    line: s.line,
                    end_line: s.end_line,
                    signature: s.signature.clone().unwrap_or_default(),
                    children: s.children.iter().map(code_to_tree).collect(),
                }
            }

            let mut analyses = std::collections::BTreeMap::new();
            for entry in &walker_entries {
                let full_path = root_dir.join(&entry.relative_path);
                if let Ok(content) = std::fs::read_to_string(&full_path) {
                    let ext = entry.relative_path.rsplit('.').next().unwrap_or("");
                    if let Ok(symbols) =
                        contextplus_rs::core::tree_sitter::parse_with_tree_sitter(&content, ext)
                    {
                        let header = contextplus_rs::core::parser::extract_header(&content);
                        analyses.insert(
                            entry.relative_path.clone(),
                            context_tree::FileAnalysis {
                                header: if header.is_empty() {
                                    None
                                } else {
                                    Some(header)
                                },
                                symbols: symbols.iter().map(code_to_tree).collect(),
                            },
                        );
                    }
                }
            }

            let options = context_tree::ContextTreeOptions {
                root_dir: root_dir.clone(),
                target_path: None,
                depth_limit: None,
                include_symbols: Some(true),
                max_tokens,
            };
            let result = context_tree::get_context_tree(options, &entries, &analyses).await?;
            println!("{}", result);
        }
        Some(Commands::Hooks { action }) => match action {
            HooksAction::Install => {
                let installed = contextplus_rs::git::hooks::install_hooks(&root_dir)?;
                eprintln!("Installed {} hook(s):", installed.len());
                for path in installed {
                    eprintln!("  {}", path.display());
                }
            }
            HooksAction::Uninstall => {
                let removed = contextplus_rs::git::hooks::uninstall_hooks(&root_dir)?;
                eprintln!("Cleaned {} hook(s).", removed.len());
            }
        },
    }

    Ok(())
}

async fn run_implicit_default(
    mode: TransportMode,
    root_dir: PathBuf,
    config: Config,
) -> anyhow::Result<()> {
    let want_daemon = match mode {
        TransportMode::Stdio => false,
        TransportMode::Daemon => true,
        TransportMode::Auto => mcp_data_writable(&root_dir),
    };

    if want_daemon {
        match contextplus_rs::transport::client::run(&root_dir).await {
            Ok(()) => return Ok(()),
            Err(e) if mode == TransportMode::Auto => {
                tracing::warn!("daemon transport failed ({e}); falling back to stdio");
            }
            Err(e) => return Err(e),
        }
    }

    run_mcp_server(root_dir, config).await
}

async fn run_mcp_server(root_dir: PathBuf, config: Config) -> anyhow::Result<()> {
    tracing::info!(
        "Starting contextplus MCP server on {} (model: {})",
        root_dir.display(),
        config.ollama_embed_model
    );

    let server = ContextPlusServer::new(root_dir.clone(), config.clone());

    let root_str = root_dir.to_string_lossy().to_string();
    if let Err(e) = server
        .state
        .memory_graph
        .get_graph(&root_str, |_graph| {})
        .await
    {
        tracing::warn!("Failed to pre-load memory graph from disk: {e}");
    }

    let _debounce_handle = server.state.memory_graph.spawn_debounce_task();

    use contextplus_rs::config::TrackerMode;
    tracing::info!(mode = %config.embed_tracker_mode, "Embedding tracker mode");
    if config.embed_tracker_mode == TrackerMode::Eager {
        server.ensure_tracker_started();
    }

    if config.warmup_on_start {
        tracing::info!("Spawning SearchIndex warmup task (CONTEXTPLUS_WARMUP_ON_START=true)");
        server.spawn_warmup_task();
    }

    let idle_timeout_ms = config.idle_timeout_ms;
    let idle_monitor = process_lifecycle::create_idle_monitor(idle_timeout_ms, move || {
        tracing::info!(
            "Idle timeout ({}ms) reached -- initiating shutdown",
            idle_timeout_ms
        );
        std::process::exit(0);
    });
    if config.idle_timeout_ms > 0 {
        tracing::info!(
            "Idle shutdown monitor started ({}ms)",
            config.idle_timeout_ms
        );
    }

    let idle_monitor = Arc::new(idle_monitor);
    {
        let mut guard = server.state.idle_monitor.write().await;
        *guard = Some(idle_monitor.clone());
    }

    #[cfg(unix)]
    let _parent_monitor = {
        let parent_pid = std::os::unix::process::parent_id();
        let handle =
            process_lifecycle::start_parent_monitor(parent_pid, config.parent_poll_ms, move || {
                tracing::info!(
                    "Parent process (pid={}) exited -- initiating shutdown",
                    parent_pid
                );
                std::process::exit(0);
            });
        tracing::info!(
            "Parent PID monitor started (pid={}, poll={}ms)",
            parent_pid,
            config.parent_poll_ms
        );
        handle
    };

    let memory_graph = Arc::clone(&server.state.memory_graph);
    let state_for_shutdown = server.state.clone();

    let transport = rmcp::transport::io::stdio();
    let ct = server.serve(transport).await?;

    #[cfg(unix)]
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    #[cfg(unix)]
    let mut sighup = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup())?;

    tokio::select! {
        result = ct.waiting() => { result?; }
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("SIGINT received -- cancelling embeddings and shutting down");
            state_for_shutdown.ollama.cancel_all_embeddings();
        }
        _ = async {
            #[cfg(unix)]
            { sigterm.recv().await }
            #[cfg(not(unix))]
            { std::future::pending::<Option<()>>().await }
        } => {
            tracing::info!("SIGTERM received -- shutting down");
            state_for_shutdown.ollama.cancel_all_embeddings();
        }
        _ = async {
            #[cfg(unix)]
            { sighup.recv().await }
            #[cfg(not(unix))]
            { std::future::pending::<Option<()>>().await }
        } => {
            tracing::info!("SIGHUP received -- shutting down");
            state_for_shutdown.ollama.cancel_all_embeddings();
        }
    }

    state_for_shutdown.ollama.flush_query_cache();

    if let Err(e) = memory_graph.flush().await {
        tracing::warn!("Failed to persist memory graph on shutdown: {e}");
    }

    idle_monitor.stop();
    #[cfg(unix)]
    _parent_monitor.stop();

    Ok(())
}
