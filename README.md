# contextplus-rs

High-performance [MCP](https://modelcontextprotocol.io/) server for semantic code analysis, written in Rust. Drop-in replacement for the original [Context+](https://github.com/mrsufgi/contextplus) TypeScript implementation with **5-20x faster** warm queries.

## Why Rust?

### Cold Start Benchmark (fresh process per call)

Measured against the workspace (~2800 source files), each call spawns a fresh process:

| Tool | Rust | TypeScript | Grep | Rust vs TS |
|------|------|-----------|------|------------|
| `get_file_skeleton` | **1,712 ms** | 4,030 ms | 1,315 ms* | **2.4x faster** |
| `get_context_tree` | **1,857 ms** | 4,002 ms | 1,382 ms* | **2.2x faster** |
| `get_blast_radius` | **1,881 ms** | 4,148 ms | 1,323 ms* | **2.2x faster** |
| `semantic_code_search` | **1,684 ms** | 4,081 ms | 2,448 ms* | **2.4x faster** |
| `semantic_identifier_search` | **1,854 ms** | 4,128 ms | 1,296 ms* | **2.2x faster** |
| **Average** | **1,798 ms** | **4,078 ms** | **1,553 ms** | **2.3x faster** |

\* Grep is fast but returns raw matches — see token comparison below.

### Token Efficiency (MCP vs Grep)

MCP tools return ranked, structured results. Grep returns raw lines.

| Query | MCP Output | Grep Output | Reduction |
|-------|-----------|-------------|-----------|
| File skeleton (profile.ts) | ~400 tokens | ~1,678 tokens | **4x fewer** |
| Context tree (scheduling) | ~1,250 tokens | ~707K tokens | **566x fewer** |
| Blast radius (createProfileService) | ~300 tokens | ~1,419 tokens | **5x fewer** |
| Semantic search ("form validation") | ~500 tokens | ~42.7M tokens | **85,000x fewer** |
| Identifier search ("membership") | ~625 tokens | ~23.9M tokens | **38,000x fewer** |

### 20-Search Session Cost

| Engine | Wall Time | Tokens Consumed |
|--------|-----------|-----------------|
| **Rust MCP** | **~28s** | **~10K tokens** |
| TS MCP | ~73s | ~10K tokens |
| Grep | ~96s | **~268M tokens** |

### Internal Bottleneck Comparison

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
- **Disk-persistent embedding cache** with content-hash staleness detection

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
ollama pull qwen3.5:9b                # chat (for cluster labeling)
```

## Configuration

Same environment variables as the TypeScript version:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `OLLAMA_EMBED_MODEL` | `snowflake-arctic-embed2` | Embedding model |
| `OLLAMA_CHAT_MODEL` | `llama3.2` | Chat model for cluster labels (recommended: `qwen3.5:9b`) |
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
        "OLLAMA_EMBED_MODEL": "snowflake-arctic-embed2",
        "OLLAMA_CHAT_MODEL": "qwen3.5:9b",
        "OLLAMA_HOST": "http://127.0.0.1:11434",
        "CONTEXTPLUS_EMBED_BATCH_SIZE": "256",
        "CONTEXTPLUS_EMBED_TRACKER": "true"
      }
    }
  }
}
```

> **Note:** `think: false` is sent automatically to the chat model to avoid slow thinking-mode responses. Models like `qwen3.5:9b` produce cluster labels in <1s with thinking disabled vs 45s+ with thinking enabled.

### CLI subcommands

```bash
# Generate MCP config for your editor (claude, cursor, vscode, windsurf, opencode)
contextplus-rs init claude
contextplus-rs init cursor

# Print file skeleton
contextplus-rs skeleton src/main.rs

# Print context tree
contextplus-rs tree --max-tokens 5000
```

## Tools (17)

### Code Analysis
| Tool | Description |
|------|-------------|
| `get_context_tree` | Token-aware file tree with symbols and line ranges. Supports `depth_limit` to cap directory depth and `max_tokens` (default 50K) for automatic pruning (Level 2 → 1 → 0) |
| `get_file_skeleton` | Function signatures and structure without full file read |
| `get_blast_radius` | Map every file that imports or references a symbol |
| `run_static_analysis` | Run available linters (tsc with `--build` for project references, eslint, cargo check, ruff) |

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

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed internals (data flow, caching strategy, memory
layout, performance architecture, and how to add new tools).

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
  tools/                     # One file per tool (context_tree supports depth_limit filtering)
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

## Benchmarks (`cargo bench`)

Four criterion benchmark suites cover the critical hot paths — no Ollama dependency, fully reproducible.

### Cache Load (rkyv + mmap)

How fast the embedding cache loads from disk. This was the #1 bottleneck in TS (1,109ms raw, 115ms with VectorStore optimization).

| Operation | 1K vectors | 5K vectors | 30K vectors |
|-----------|-----------|-----------|------------|
| rkyv read | 0.36 ms | 2.8 ms | 181 ms |
| rkyv mmap | 0.63 ms | 3.1 ms | 103 ms |
| to_store (HashMap build) | 0.16 ms | 0.97 ms | 91 ms |

At 5K vectors (typical project size), total load is **~4ms**. At 30K vectors, mmap beats read by 43%.

### Cosine Similarity (simsimd SIMD vs scalar)

| Operation | 1K×1024 | 5K×1024 | 30K×1024 | SIMD speedup |
|-----------|---------|---------|----------|-------------|
| simsimd scan | 68 µs | 404 µs | **4.1 ms** | — |
| naive scan | 663 µs | 3.4 ms | 20 ms | **~5x** |
| Single pair (1024-dim) | 0.08 µs | — | — | **8x** vs naive |

30K-vector scan in 4.1ms — well under the 20ms target.

### Tree-sitter Parse (native, 10 languages)

| Language | Parse time |
|----------|-----------|
| TypeScript | 113 µs |
| TSX | 143 µs |
| JavaScript | 102 µs |
| Python | 141 µs |
| Rust | 171 µs |
| Go | 100 µs |
| Java | 96 µs |
| C | 98 µs |
| C++ | 91 µs |
| Bash | ~83 µs |
| **All 10 combined** | **~1.4 ms** |

All 10 languages parsed in **1.4ms total** — vs 50-200ms for WASM in TS.

### Warm Search Pipeline

End-to-end: disk load → VectorStore build → find_nearest(top_5) → format results.

| Scenario | 1K files | 5K files | 30K files |
|----------|---------|---------|----------|
| Full pipeline (mmap + search) | 0.9 ms | 5.3 ms | 224 ms |
| Warm search only (in-memory) | **78 µs** | **418 µs** | **4.3 ms** |
| Hash-check staleness (no-op) | 23 µs | 131 µs | 975 µs |

Warm search on 30K files: **4.3ms**. Hash-check (no-op refresh): **<1ms**.

## Development

```bash
cargo test                  # 584 tests
cargo bench                 # Criterion benchmarks (cache, cosine, tree-sitter, search)
cargo clippy --all-targets  # Lint
cargo fmt --check           # Format check
```

## Credits

This is a Rust rewrite of [Context+](https://github.com/mrsufgi/contextplus) (TypeScript), originally created by the Context+ community. The Rust port was built from the [`fix/merge-cache-on-save`](https://github.com/mrsufgi/contextplus/tree/fix/merge-cache-on-save) branch which includes performance optimizations (hash-based cache invalidation, VectorStore extraction, parallel call-site ranking, tree-sitter dedup).

## License

MIT
