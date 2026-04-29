use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use contextplus_rs::config::Config;
use contextplus_rs::transport::dispatch::{resolve_transport_mode, run_implicit_default, run_mcp_server};

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
