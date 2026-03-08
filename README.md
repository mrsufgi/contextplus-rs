# contextplus-rs

High-performance [MCP](https://modelcontextprotocol.io/) server for semantic code analysis, written in Rust. Drop-in replacement for the original [Context+](https://github.com/mrsufgi/contextplus) TypeScript implementation with **5-20x faster** warm queries.

## Why Rust?

The original TypeScript Context+ server has measured bottlenecks:

| Bottleneck | TypeScript | Rust | Improvement |
|------------|-----------|------|-------------|
| Cache load (120MB) | 115ms (VectorStore) | ~2ms (mmap) | **57x** |
| Cosine scan 30K vectors | ~50ms (JS loop) | ~15ms (SIMD) | **3x** |
| Tree-sitter parse | 5-20ms (WASM) | 1-5ms (native) | **4x** |
| Warm semantic search | ~1.5s | <100ms | **15x** |
| No-op file refresh | 870ms | <10ms | **87x** |

Rust eliminates all overhead via:
- **Zero-copy cache** with `rkyv` + `memmap2` (no deserialization)
- **SIMD cosine similarity** via `simsimd` (AVX-512/AVX2 auto-dispatch)
- **Native tree-sitter** (compiled in, no WASM VM)
- **Binary serialization** (rkyv replaces JSON for memory graph)

## Install

### Build from source

```bash
git clone https://github.com/mrsufgi/contextplus-rs.git
cd contextplus-rs
cargo build --release
```

The binary is at `target/release/contextplus-rs`.

### Prerequisites

- [Ollama](https://ollama.com/) running locally with an embedding model:

```bash
ollama pull snowflake-arctic-embed2   # embeddings
ollama pull llama3.2                  # chat (for cluster labeling)
```

## Configuration

Same environment variables as the TypeScript version:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `OLLAMA_EMBED_MODEL` | `snowflake-arctic-embed2` | Embedding model |
| `OLLAMA_CHAT_MODEL` | `llama3.2` | Chat model (for cluster labels) |
| `OLLAMA_API_KEY` | _(none)_ | Optional API key |
| `CONTEXTPLUS_EMBED_BATCH_SIZE` | `32` | Embedding batch size |
| `CONTEXTPLUS_EMBED_TRACKER` | `true` | Enable file watcher |
| `CONTEXTPLUS_EMBED_TRACKER_DEBOUNCE_MS` | `700` | File watcher debounce |
| `CONTEXTPLUS_EMBED_TRACKER_MAX_FILES` | `8` | Max files per watcher tick |
| `CONTEXTPLUS_IGNORE_DIRS` | _(none)_ | Extra directories to ignore (comma-separated) |

## Usage

### As an MCP server (stdio)

```bash
contextplus-rs --root-dir /path/to/project
```

### Claude Code integration

Add to your MCP config (`~/.claude/mcp.json` or project `.mcp.json`):

```json
{
  "mcpServers": {
    "contextplus": {
      "command": "/path/to/contextplus-rs",
      "args": ["--root-dir", "/path/to/project"],
      "env": {
        "OLLAMA_HOST": "http://127.0.0.1:11434",
        "OLLAMA_EMBED_MODEL": "snowflake-arctic-embed2"
      }
    }
  }
}
```

### CLI subcommands

```bash
# Initialize .mcp_data directory
contextplus-rs --root-dir . init

# Print file skeleton
contextplus-rs --root-dir . skeleton src/main.rs

# Print context tree
contextplus-rs --root-dir . tree --max-tokens 5000
```

## Tools (17)

### Code Analysis
| Tool | Description |
|------|-------------|
| `get_context_tree` | Token-aware file tree with symbols and line ranges |
| `get_file_skeleton` | Function signatures and structure without full file read |
| `get_blast_radius` | Map every file that imports or references a symbol |
| `run_static_analysis` | Run available linters (tsc, eslint, cargo check, ruff) |

### Semantic Search
| Tool | Description |
|------|-------------|
| `semantic_code_search` | Hybrid semantic + keyword file search via Ollama embeddings |
| `semantic_identifier_search` | Find functions/classes by meaning with call-site ranking |
| `semantic_navigate` | Cluster files by semantic similarity (spectral clustering) |

### File Management
| Tool | Description |
|------|-------------|
| `propose_commit` | Write files with validation and shadow restore points |
| `list_restore_points` | List all shadow restore points |
| `undo_change` | Restore files from a restore point |

### Memory Graph
| Tool | Description |
|------|-------------|
| `upsert_memory_node` | Create or update a memory graph node |
| `create_relation` | Create or update edges between nodes |
| `search_memory_graph` | Semantic search with BFS traversal |
| `retrieve_with_traversal` | Retrieve node neighborhood via BFS |
| `add_interlinked_context` | Batch-add nodes with auto-linking |
| `prune_stale_links` | Remove decayed edges and orphan nodes |

### Navigation
| Tool | Description |
|------|-------------|
| `get_feature_hub` | Navigate Obsidian-style wikilinks between feature docs |

## Architecture

```
src/
  main.rs                    # CLI + MCP server entry point
  server.rs                  # ServerHandler + tool dispatch
  config.rs                  # Environment variable configuration
  error.rs                   # ContextPlusError enum (thiserror)
  core/
    embeddings.rs            # OllamaClient + VectorStore + simsimd cosine
    tree_sitter.rs           # Native multi-lang parser (10 languages)
    parser.rs                # Code symbol extraction + hash_content
    walker.rs                # gitignore-aware file walker (ignore crate)
    embedding_tracker.rs     # File watcher with debounced refresh
    clustering.rs            # Spectral clustering (nalgebra)
    memory_graph.rs          # petgraph + rkyv persistence
    hub.rs                   # Wikilink parser
  tools/                     # One file per tool
  git/shadow.rs              # Restore points (file-based backup)
  cache/rkyv_store.rs        # Zero-copy rkyv+mmap VectorStore
```

### Supported Languages (tree-sitter)

TypeScript, TSX, JavaScript, Python, Rust, Go, Java, C, C++, Bash

### Key Crates

| Crate | Purpose |
|-------|---------|
| `rmcp` | MCP SDK with stdio transport |
| `simsimd` | SIMD-accelerated cosine distance |
| `rkyv` + `memmap2` | Zero-copy cache persistence |
| `tree-sitter` | Native code parsing (10 languages) |
| `petgraph` | Memory graph with stable indices |
| `nalgebra` | Spectral clustering (eigendecomposition) |
| `notify` | File system watching |
| `ignore` | gitignore-aware file walking |

## Development

```bash
cargo test                  # Run all tests
cargo clippy                # Lint
cargo fmt --check           # Format check
cargo bench                 # Run benchmarks
```

## Credits

This is a Rust rewrite of [Context+](https://github.com/mrsufgi/contextplus) (TypeScript), originally created by the Context+ community. The Rust port was built from the [`fix/merge-cache-on-save`](https://github.com/mrsufgi/contextplus/tree/fix/merge-cache-on-save) branch which includes performance optimizations (hash-based cache invalidation, VectorStore extraction, parallel call-site ranking, tree-sitter dedup).

## License

MIT
