# contextplus-rs

High-performance [MCP](https://modelcontextprotocol.io/) server for semantic code analysis, written in Rust. Drop-in replacement for the original [Context+](https://github.com/mrsufgi/contextplus) TypeScript implementation with **5-20x faster** warm queries.

## Why Rust?

### Server Latency (~9.7K-file monorepo)

Measured on a ~9.7K-file TypeScript/Go monorepo with a warm embedding cache, release build, Intel Core Ultra 7 265 (20 CPUs), Linux. Cold start = first call from a fresh process (three runs, median reported). Warm = median of 20 consecutive calls to the same running server.

**Cold start** (process boot + cache load included):

| Phase | Median |
|-------|--------|
| Server initialize | 126 ms |
| `explore` first call | 551 ms |
| `impact cycles` first call | 1,144 ms |
| `impact dead` first call | 1,814 ms |

Second-call speedup from the running server: `explore` 551 ms → 7 ms (79×); `impact dead` 1,814 ms → 43 ms (42×).

**Warm per-call latency** (subsequent calls, same process):

| Tool | Median |
|------|--------|
| `explore` files (semantic) | 5.6 ms |
| `explore` files (keywords) | 0.9 ms |
| `explore` files (scoped) | 4.2 ms |
| `explore` identifiers (semantic) | 39 ms |
| `explore` identifiers (keywords) | 80 ms |
| `explore` clusters | 92 ms |
| `outline` file | 0.9 ms |
| `outline` dir (small) | 2.2 ms |
| `outline` dir (large, depth 1) | 13.5 ms |
| `impact` symbol | 73 ms |
| `impact` cycles | 10 ms |
| `impact` dead | 51 ms |
| `check` embeddings | 69 ms |
| `worktrees` list | 0.3 ms |

Memory stays flat across repeated calls: 20 back-to-back `outline` directory calls hold at 700–750 MiB of process memory (RSS + swap), where the previous glibc-malloc build climbed from 835 to 1,002 MiB.

### Historical Comparison (cold-process methodology)

The rows below were collected with a different methodology: each call spawns a fresh process. They are retained as a rough Rust-vs-TypeScript baseline.

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
| File skeleton (any module) | ~400 tokens | ~1,678 tokens | **4x fewer** |
| Context tree (any directory) | ~1,250 tokens | ~707K tokens | **566x fewer** |
| Blast radius (any symbol) | ~300 tokens | ~1,419 tokens | **5x fewer** |
| Semantic search ("form validation") | ~500 tokens | ~42.7M tokens | **85,000x fewer** |
| Identifier search ("any concept") | ~625 tokens | ~23.9M tokens | **38,000x fewer** |

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
- **Native tree-sitter** (compiled in, no WASM VM) — 16 languages + regex fallback
- **Disk-persistent embedding cache** with content-hash staleness detection
- **Adaptive embedding retry** with exponential backoff and cancellation tokens
- **Automatic chunking** for oversized embedding inputs (chunk → embed → merge)
- **Process lifecycle management** — idle timeout, parent PID orphan detection, SIGTERM/SIGHUP handling
- **jemalloc** allocator with background purge via `tikv-jemallocator`

## Install

### Build from source

```bash
git clone https://github.com/mrsufgi/contextplus-rs.git
cd contextplus-rs
cargo build --release
```

The binary is at `target/release/contextplus-rs`.

### Default Ollama setup

The default configuration uses [Ollama](https://ollama.com/) locally:

```bash
ollama pull snowflake-arctic-embed2   # embeddings
ollama pull qwen3.5:9b                # chat (for cluster labeling)
```

## Configuration

### Providers

Ollama remains the default for both embeddings and cluster-label chat. The binary reads provider credentials only from its process environment; it does not fetch from or integrate with a secrets manager.

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTEXTPLUS_EMBED_PROVIDER` | `ollama` | Embedding provider: `ollama` or `openai` |
| `CONTEXTPLUS_CHAT_PROVIDER` | `ollama` | Cluster-label chat provider: `ollama`, `openai`, `claude`, or `anthropic` |
| `CONTEXTPLUS_OPENAI_API_KEY` | _(none)_ | Bearer key for OpenAI-compatible embeddings and, unless overridden, OpenAI-compatible chat |
| `CONTEXTPLUS_OPENAI_BASE_URL` | `https://api.openai.com/v1` | Base URL for OpenAI-compatible embeddings and, unless overridden, chat |
| `CONTEXTPLUS_OPENAI_EMBED_MODEL` | `text-embedding-3-small` | OpenAI-compatible embedding model |
| `CONTEXTPLUS_OPENAI_CHAT_MODEL` | `gpt-4o-mini` | OpenAI-compatible cluster-label model |
| `CONTEXTPLUS_CHAT_BASE_URL` | _(uses OpenAI base URL)_ | Optional OpenAI-compatible chat-only base URL |
| `CONTEXTPLUS_CHAT_API_KEY` | _(uses provider fallback)_ | Optional OpenAI-compatible chat-only bearer key; takes precedence over other keys |
| `GROQ_API_KEY` | _(none)_ | Chat key fallback when the effective chat base URL host is `api.groq.com` |
| `CONTEXTPLUS_CLAUDE_PATH` | `claude` | Claude Code executable for the `claude` chat provider |
| `CONTEXTPLUS_CLAUDE_MODEL` | `claude-haiku-4-5` | Claude Code model for cluster labels |
| `CLAUDE_CODE_OAUTH_TOKEN` | _(Claude Code login)_ | Passed through unchanged to Claude Code; contextplus does not read or log it |
| `CONTEXTPLUS_ANTHROPIC_CHAT_MODEL` | `claude-haiku-4-5` | Messages API model for the `anthropic` chat provider |
| `ANTHROPIC_API_KEY` | _(none)_ | Messages API `x-api-key`; takes precedence over `ANTHROPIC_AUTH_TOKEN`. Also inherited unchanged by the Claude CLI |
| `ANTHROPIC_AUTH_TOKEN` | _(none)_ | Messages API bearer token used with the OAuth beta header when no API key is set |
| `HOME` | _(inherited)_ | Claude Code's existing login/config location; not read or rewritten by the Claude subprocess adapter |
| `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, `NO_PROXY` (and lowercase equivalents) | _(inherited)_ | Proxy settings passed through unchanged to Claude Code; HTTP providers use reqwest's proxy support |

OpenAI-compatible embeddings (including vLLM, LiteLLM, and compatible local/proxy servers):

```bash
export CONTEXTPLUS_EMBED_PROVIDER=openai
export CONTEXTPLUS_OPENAI_BASE_URL=https://api.openai.com/v1
export CONTEXTPLUS_OPENAI_API_KEY=your-key
export CONTEXTPLUS_OPENAI_EMBED_MODEL=text-embedding-3-small
contextplus-rs --root-dir /path/to/project
```

Groq chat with embeddings configured independently:

```bash
export CONTEXTPLUS_CHAT_PROVIDER=openai
export CONTEXTPLUS_CHAT_BASE_URL=https://api.groq.com/openai/v1
export GROQ_API_KEY=your-groq-key
export CONTEXTPLUS_OPENAI_CHAT_MODEL=your-groq-model
contextplus-rs --root-dir /path/to/project
```

`claude` runs one non-interactive Claude Code process per label prompt using the existing Claude Code login. `anthropic` calls the Messages API directly. All chat providers retain the 90-second timeout and fall back to non-LLM labels on errors.

The Claude adapter uses print/JSON mode, safe mode, an empty strict MCP configuration, no tools or skills, empty settings sources, no session persistence or permission prompts, and a temporary working directory. The installed CLI must support these flags; unsupported flags produce a fallback label. Claude Code's admin-managed policies still apply. `--bare` is deliberately not used because it disables OAuth login. The child inherits the entire environment without credential inspection and is killed on timeout; its stderr is discarded.

Embedding batching, adaptive context-length retries, chunk-and-merge, prefixes, cancellation, and query caching are shared by both providers. Existing Ollama cache names are unchanged. OpenAI cache names include the provider and a model/base-URL fingerprint so different endpoints or model aliases do not share vectors. Provider errors omit remote response bodies and request URLs to prevent credential disclosure.

### Ollama / embedding

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `OLLAMA_EMBED_MODEL` | `snowflake-arctic-embed2` | Embedding model |
| `OLLAMA_CHAT_MODEL` | `llama3.2` | Chat model for cluster labels |
| `OLLAMA_API_KEY` | _(none)_ | Optional API key |
| `CONTEXTPLUS_EMBED_BATCH_SIZE` | `50` | Document embedding batch size (clamped 5–512) |
| `CONTEXTPLUS_QUERY_BATCH_SIZE` | `1` | Query embedding batch size for live searches |
| `CONTEXTPLUS_EMBED_QUERY_PREFIX` | Model-specific | Query prefix: `task: code retrieval \| query: ` for `embeddinggemma`, `query: ` for `snowflake-arctic-embed2`, empty otherwise. Set to an empty string to disable |
| `CONTEXTPLUS_EMBED_DOC_PREFIX` | Model-specific | Document prefix: `title: none \| text: ` for `embeddinggemma`, empty otherwise. Set to an empty string to disable |
| `CONTEXTPLUS_EMBED_DOC_SHAPE` | Model-specific | File document shape: `outline` for `embeddinggemma`, `head` otherwise |
| `CONTEXTPLUS_EMBED_CHUNK_CHARS` | `2000` | Max chars per embedding input (clamped 256–8000). Oversized inputs are chunked and merged |
| `CONTEXTPLUS_MAX_EMBED_FILE_SIZE` | `51200` (50 KB) | Skip files larger than this (bytes) for embedding. Min 1 KB |
| `CONTEXTPLUS_IGNORE_DIRS` | _(none)_ | Extra directories to ignore (comma-separated), appended to the built-in list |
| `CONTEXTPLUS_CACHE_TTL_SECS` | `300` | Embedding cache TTL in seconds |
| `CONTEXTPLUS_IDENTIFIER_PRUNE_GRACE_SECS` | `86400` | How long an identifier embedding goes unused by every attached ref before a save drops it from the cache |
| `CONTEXTPLUS_EMBED_NUM_GPU` | _(none)_ | Ollama `num_gpu` option (GPU layer count) |
| `CONTEXTPLUS_EMBED_MAIN_GPU` | _(none)_ | Ollama `main_gpu` option (primary GPU index) |
| `CONTEXTPLUS_EMBED_NUM_THREAD` | _(none)_ | Ollama `num_thread` option |
| `CONTEXTPLUS_EMBED_NUM_BATCH` | _(none)_ | Ollama `num_batch` option |
| `CONTEXTPLUS_EMBED_NUM_CTX` | _(none)_ | Ollama `num_ctx` option |
| `CONTEXTPLUS_EMBED_LOW_VRAM` | _(none)_ | Ollama `low_vram` option (`true`/`false`) |

### Search / indexing

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTEXTPLUS_HNSW_EF_CONSTRUCTION` | `100` | HNSW `efConstruction` — higher values improve index quality at the cost of build time |
| `CONTEXTPLUS_HNSW_EF_SEARCH` | `32` | HNSW `ef_search` — higher values improve recall at the cost of query latency. Set explicitly when higher recall is needed |
| `CONTEXTPLUS_HNSW_MIN_VECTORS` | `50000` | Embedded files from which semantic search builds the HNSW graph and re-ranks its shortlist. Below it every file is scored exactly and no graph is built |
| `CONTEXTPLUS_ANN_CANDIDATE_MULTIPLIER` | `10` | ANN candidate pool multiplier: fetches `top_k × N` HNSW candidates before re-ranking. Larger values improve recall; only applies from `CONTEXTPLUS_HNSW_MIN_VECTORS` embedded files |

### Warmup

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTEXTPLUS_WARMUP_ON_START` | `true` | Warm the `SearchIndex` cache at server startup. Set to `false` / `0` / `no` / `off` to disable |
| `CONTEXTPLUS_SNAPSHOTS` | `true` | Persist the primary checkout's built indexes under `.mcp_data/snapshots/` and load them on the next start, indexing only the files changed since. Set to `false` / `0` / `no` / `off` to disable |
| `CONTEXTPLUS_WARMUP_CONCURRENCY` | `1` | Number of parallel Ollama embed requests during `warmup_embeddings` / `warmup_identifiers`. Set to match `OLLAMA_NUM_PARALLEL` on the host |

### Tracker (file-watcher)

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTEXTPLUS_EMBED_TRACKER` | `lazy` | Tracker mode: `lazy` (start on first search), `eager` / `startup` (start at boot), `off` / `false` (disabled) |
| `CONTEXTPLUS_EMBED_TRACKER_DEBOUNCE_MS` | `700` | File-watcher debounce window in milliseconds |
| `CONTEXTPLUS_EMBED_TRACKER_MAX_FILES` | `8` | Max files re-embedded per watcher tick |

### Process lifecycle

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTEXTPLUS_IDLE_TIMEOUT_MS` | `900000` | Auto-shutdown after this many ms idle (0 or `off` to disable, min 60 s) |
| `CONTEXTPLUS_PARENT_POLL_MS` | `5000` | Poll interval for parent PID orphan detection in milliseconds (min 1 s) |

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
        "CONTEXTPLUS_EMBED_TRACKER": "eager"
      }
    }
  }
}
```

> **Note:** `think: false` is sent automatically to the chat model to avoid slow thinking-mode responses. Models like `qwen3.5:9b` produce cluster labels in <1s with thinking disabled vs 45s+ with thinking enabled.

#### Git worktrees

One `--root-dir` (the primary checkout) serves every worktree of the repo. A tool call runs
against the worktree the host process is in (Claude Code's `EnterWorktree`, Codex started
inside a worktree) or the one an absolute path argument points into; relative paths resolve
from that worktree's root. A worktree is attached on first use, forking the primary's
embedding cache, so no `attach_worktree` call is needed. The host's cwd is read from
`/proc/<parent pid>/cwd`, so the automatic part is Linux-only; absolute paths route everywhere.

Each worktree's tracker also watches its gitdir and the shared `refs/heads`: when HEAD moves
(pull, merge, rebase, checkout, reset) the files that differ between the old and new commit are
re-embedded for that worktree. The `contextplus-rs hooks install` sentinels remain an optional
fast path; nothing depends on them, which matters when another tool (lefthook, husky) owns
`.git/hooks`.

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

## MCP Resources

| URI | Description |
|-----|-------------|
| `contextplus://instructions` | Returns tool usage instructions fetched from the Context+ API. Cached in memory after first fetch |

## Tools

Five tools, named for what an agent is doing. Each description says when to use it, so no
separate instructions are needed.

| Tool | Use it to | Wraps |
|------|-----------|-------|
| `explore` | find code by what it does: files (`kind: files`, default), functions and classes with call sites (`identifiers`), or the codebase grouped by topic (`clusters`); `match: keywords` for exact identifiers without embeddings | `semantic_code_search`, `semantic_identifier_search`, `semantic_navigate`, `lexical_search` |
| `outline` | see a file's signatures and line ranges, or a directory's file and symbol tree, before reading anything | `get_file_skeleton`, `get_context_tree` |
| `impact` | learn what breaks if a symbol changes (`what: symbol`, default) or what a whole diff touches (`diff`), the project's import cycles (`cycles`), or symbols nothing references (`dead`) | `get_blast_radius`, `review_pr_diff`, `detect_dependency_loops`, `find_dead_code` |
| `check` | run the project's linters and compilers on a path (`what: lint`, default) or audit the embedding cache (`embeddings`) | `run_static_analysis`, `check_embedding_quality` |
| `worktrees` | list, attach or detach git worktrees (`action`); attaching also happens automatically when a call names a path inside one | `list_worktrees`, `attach_worktree`, `detach_worktree` |

The pre-facade names in the last column still dispatch for one release but are not listed.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed internals (data flow, caching strategy,
performance architecture, and how to add new tools).

```
src/
  main.rs                    # CLI + MCP server entry point
  server_adapters.rs         # rmcp ServerHandler impl + tool dispatch
  server_definitions.rs      # Tool definitions (names, descriptions, JSON schemas)
  server_helpers.rs          # Shared handler utilities
  config.rs                  # Environment variable configuration
  error.rs                   # ContextPlusError enum (thiserror)
  core/
    embeddings.rs            # OllamaClient + adaptive retry + chunking + cancellation
    tree_sitter.rs           # Native multi-lang parser (16 languages)
    parser.rs                # Code symbol extraction + regex fallback for unsupported langs
    walker.rs                # gitignore-aware file walker (ignore crate)
    embedding_tracker.rs     # File watcher with lazy/eager/off modes
    clustering.rs            # Spectral clustering (nalgebra)
    hub.rs                   # Wikilink parser
    process_lifecycle.rs     # Idle timeout + parent PID orphan detection
    safe_path.rs             # Path traversal prevention
    utils.rs                 # Shared utilities
  tools/                     # One file per tool (context_tree supports depth_limit filtering)
  git/shadow.rs              # Restore points (file-based backup)
  cache/rkyv_store.rs        # Zero-copy rkyv+mmap VectorStore
```

### Supported Languages (tree-sitter)

TypeScript, TSX, JavaScript, Python, Rust, Go, Java, C, C++, Bash, Ruby, PHP, C#, Kotlin, HTML, CSS

Unsupported file types fall back to regex-based symbol extraction.

### Key Crates

| Crate | Purpose |
|-------|---------|
| `rmcp` | MCP SDK with stdio transport |
| `simsimd` | SIMD-accelerated cosine distance |
| `rkyv` + `memmap2` | Zero-copy cache persistence |
| `tree-sitter` | Native code parsing (16 languages) |
| `nalgebra` | Spectral clustering (eigendecomposition) |
| `tikv-jemallocator` | jemalloc allocator with background purge |
| `notify` | File system watching |
| `ignore` | gitignore-aware file walking |

## Benchmarks (`cargo bench`)

Nine Criterion benchmark suites cover the critical hot paths — no Ollama dependency, fully reproducible.

### Cache Load (rkyv + mmap)

How fast the embedding cache loads from disk. This was the #1 bottleneck in TS (1,109ms raw, 115ms with VectorStore optimization).

| Operation | 1K×1024 | 5K×1024 | 30K×1024 |
|-----------|---------|---------|----------|
| rkyv read | 401 µs | 2.41 ms | 18.2 ms |
| rkyv mmap | 219 µs | 903 µs | 7.75 ms |
| to_store (HashMap build) | 219 µs | 823 µs | 7.00 ms |

At 5K vectors (typical project size), mmap + store build totals **~1.7ms**. mmap beats read by 57% at 30K.

### Cosine Similarity (simsimd SIMD vs scalar)

Single pair:

| Dims | simsimd | naive | Speedup |
|------|---------|-------|---------|
| 128 | 11.5 ns | 72.2 ns | 6.3x |
| 512 | 40.4 ns | 281 ns | 7.0x |
| 1024 | 90.3 ns | 562 ns | 6.2x |

Full corpus scan (1024-dim vectors):

| Operation | 1K vectors | 5K vectors | 30K vectors |
|-----------|-----------|-----------|------------|
| simsimd scan | 90.9 µs | 627 µs | **4.60 ms** |
| naive scan | 570 µs | 3.12 ms | 20.3 ms |
| SIMD speedup | 6.3x | 5.0x | **4.4x** |

30K-vector scan in **4.6ms** — well under the 20ms target.

### Tree-sitter Parse (native, 16 languages)

Bench covers 10 of the 16 supported languages; the remaining 6 (Ruby, PHP, C#, Kotlin, HTML, CSS) are not benchmarked.

| Language | Median |
|----------|--------|
| TypeScript | 95.7 µs |
| TSX | 111 µs |
| JavaScript | 80.6 µs |
| Python | 96.0 µs |
| Rust | 137 µs |
| Go | 73.4 µs |
| Java | 78.2 µs |
| C | 82.8 µs |
| C++ | 76.8 µs |
| Bash | 64.7 µs |
| **All 10 combined** | **1.20 ms** |

10 languages parsed sequentially in **1.2ms** — vs 50-200ms for WASM in TS.

### Warm Search Pipeline

End-to-end: disk load → `SearchIndex` build → `search()` hybrid scorer (brute-force for ≤50K files, which is the production path for every normal-sized repo).

| Scenario | 1K files | 5K files | 30K files |
|----------|---------|---------|----------|
| Warm search only (in-memory) | 285 µs | 1.87 ms | **6.42 ms** |
| Full pipeline (mmap + index + search) | 2.07 ms | 21.8 ms | 148 ms |
| Hash-check staleness (all fresh) | 15.1 µs | 83.0 µs | 676 µs |

Warm search on 30K files: **6.4ms**. Hash-check (no-op refresh): **<1ms**.

### Blast Radius

Import reachability scan across the symbol-to-file dependency graph.

| Scenario | 500 files / 100 lines | 2K files / 100 lines | 2K files / 200 lines |
|----------|-----------------------|----------------------|----------------------|
| No context | 737 µs | 2.81 ms | 5.69 ms |
| With context | 751 µs | 2.81 ms | 5.68 ms |
| Special chars | 744 µs | — | — |

Context inclusion adds negligible overhead. 2K-file scan in **~2.8ms**.

### Context Tree

Directory tree with optional symbol resolution and depth limiting.

| Scenario | 500 files | 2K files | 5K files |
|----------|-----------|---------|---------|
| No symbols | 124 µs | 913 µs | 2.88 ms |
| With symbols | 490 µs | 2.13 ms | 6.01 ms |
| Depth limit 2 | 75.0 µs | 347 µs | 919 µs |

Depth limiting cuts 5K-file time from 2.9ms to **919µs** (3.1x).

### Spectral Clustering

K-means over the eigenvector embedding of the similarity graph (nalgebra eigendecomposition).

| Scenario | 50 items | 200 items | 500 items |
|----------|---------|---------|---------|
| 64-dim vectors | 162 µs | 299 µs | 1.49 ms |
| 768-dim vectors | 211 µs | 495 µs | — |

### Identifier Score

Ranking identifiers by semantic relevance against a query vector.

| Operation | 1K ids | 5K ids | 10K ids | 114K ids |
|-----------|--------|--------|---------|---------|
| top-5 | 83.6 µs | 201 µs | 418 µs | 5.40 ms |
| top-20 | 86.4 µs | 203 µs | 365 µs | 5.94 ms |
| keywords | 85.0 µs | 199 µs | 345 µs | 5.70 ms |
| filtered kinds | 74.8 µs | 143 µs | 193 µs | 3.08 ms |

114K-identifier scan (large monorepo) in **~5-6ms**.

### Walker

gitignore-aware directory traversal (ignore crate).

| Scenario | Median |
|----------|--------|
| Small tree (100 files, depth 1) | 99.3 µs |
| Medium tree (500 files, depth 1) | 363 µs |
| Medium tree (500 files, depth 2) | 229 µs |

## Development

```bash
cargo test                  # 1835+ tests
cargo bench                 # 9 Criterion benchmark suites
cargo clippy --all-targets  # Lint
cargo fmt --check           # Format check
```

## Credits

This is a Rust rewrite of [Context+](https://github.com/mrsufgi/contextplus) (TypeScript), originally created by the Context+ community. The Rust port was built from the [`fix/merge-cache-on-save`](https://github.com/mrsufgi/contextplus/tree/fix/merge-cache-on-save) branch which includes performance optimizations (hash-based cache invalidation, VectorStore extraction, parallel call-site ranking, tree-sitter dedup).

## License

MIT
