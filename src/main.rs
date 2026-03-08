use std::path::PathBuf;

use clap::{Parser, Subcommand};
use contextplus_rs::config::Config;
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

#[derive(Subcommand)]
enum Commands {
    /// Start the MCP server over stdio (default behavior)
    Serve,
    /// Initialize .mcp_data directory
    Init,
    /// Print file skeleton for a file
    Skeleton {
        /// Path to the file
        file: String,
    },
    /// Print context tree for a directory
    Tree {
        /// Max tokens for output
        #[arg(long)]
        max_tokens: Option<usize>,
    },
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
        None | Some(Commands::Serve) => {
            run_mcp_server(root_dir, config).await?;
        }
        Some(Commands::Init) => {
            let mcp_data = root_dir.join(".mcp_data");
            tokio::fs::create_dir_all(&mcp_data).await?;
            let backups = mcp_data.join("backups");
            tokio::fs::create_dir_all(&backups).await?;
            eprintln!("Initialized .mcp_data/ in {}", root_dir.display());
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

            let header =
                contextplus_rs::core::parser::extract_header(&content.lines().collect::<Vec<_>>());
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

            // Convert walker entries to context_tree entries
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
                        let header = contextplus_rs::core::parser::extract_header(
                            &content.lines().collect::<Vec<_>>(),
                        );
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
    }

    Ok(())
}

async fn run_mcp_server(root_dir: PathBuf, config: Config) -> anyhow::Result<()> {
    tracing::info!(
        "Starting contextplus MCP server on {} (model: {})",
        root_dir.display(),
        config.ollama_embed_model
    );

    let server = ContextPlusServer::new(root_dir, config);

    let transport = rmcp::transport::io::stdio();
    let ct = server.serve(transport).await?;
    ct.waiting().await?;

    Ok(())
}
