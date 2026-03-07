# contextplus-rs: Rust MCP Server

## Enhancement Summary

**Deepened on:** 2026-03-07
**Sections enhanced:** 12
**Research agents used:** best-practices-researcher, security-sentinel, code-simplicity-reviewer

### Key Improvements
1. Added comprehensive Testing Strategy (80%+ coverage, 20%+ integration tests, wiremock mock Ollama)
2. Added Security Hardening section (path traversal, command injection, mmap safety mitigations)
3. Added Research Insights with validated crate APIs (simsimd returns distance not similarity, tree-sitter Send/Sync caveats, notify-debouncer-full 0.7 API)
4. Added Simplification Decisions (10 grammars not 36, no legacy cache reader, rkyv everywhere)

---

## Context

The Context+ MCP server (TypeScript, fork at `mrsufgi/contextplus` branch `fix/merge-cache-on-save`)
provides 17 semantic code analysis tools via Ollama embeddings. We already committed O1-O5
optimizations to the TS fork that reduced warm identifier search from 3.7s to 1.7s:

**Already fixed in TS fork (commits `de63204`, `05b4db2`):**
- O1: Skip redundant cache saves when nothing changed (no-op refresh: 2.4s -> 870ms)
- O2: VectorStore callsite extraction (avoids one loadEmbeddingCache call per query)
- O3: Parallel rankCallSites via Promise.all
- O4-O5: Tree-sitter init/grammar Promise dedup (prevents concurrent WASM loads)
- Hash-check in refreshIdentifierEmbeddings (prevents 80-98% unnecessary Ollama calls)

**Remaining bottlenecks that Rust eliminates:**
- **Array.from() still exists** in `VectorStore.getVector()`, `.toCache()`, `loadBinaryCache()` —
  ~997ms when these paths are hit (e.g. full cache rebuild, identifier search warm load)
- **120MB binary cache** still read via `readFile` into V8 heap (even with VectorStore, the
  Float32Array is copied into memory — not mmap'd)
- **Tree-sitter WASM** overhead: 50-100ms per parse vs native (dedup helps but doesn't eliminate)
- **No SIMD** in cosine similarity: scalar JS loop over 1024 floats
- **JSON serialization** for memory graph with embedded 1024-dim vectors

Rust eliminates all remaining bottlenecks with mmap + zero-copy (rkyv), native tree-sitter,
SIMD-accelerated cosine (LLVM auto-vectorizes to AVX2/AVX-512), and binary serialization.
Expected speedup over the **already-optimized** TS fork: **5-20x** on warm queries.

## Rust Crate Stack

| Purpose | Crate | Version | Why |
|---------|-------|---------|-----|
| MCP SDK | `rmcp` | v0.16+ | `#[tool_router]` macros, stdio transport |
| Async | `tokio` + `reqwest` | latest | Ollama HTTP, file I/O, concurrency |
| Tree-sitter | `tree-sitter` + per-lang crates | 0.25.x | Native C grammars, compile-time linked |
| Vector SIMD | `simsimd` v6 | latest | AVX-512/AVX2 auto-dispatch cosine distance, 8.2M ops/s |
| Zero-copy cache | `rkyv` + `memmap2` | rkyv 0.8, memmap2 0.9 | 0ms load vs 1,109ms (TS) |
| File watching | `notify` + `notify-debouncer-full` | notify 7, debouncer 0.7 | Used by cargo-watch, rust-analyzer |
| Spectral clustering | `nalgebra` (SymmetricEigen) | latest | Pure Rust eigendecomposition |
| Graph | `petgraph` StableGraph | latest | Memory graph with stable indices |
| Gitignore | `ignore` crate | latest | From ripgrep, handles .gitignore natively |
| Errors | `thiserror` | latest | Typed error enum |
| Serialization | `serde` + `serde_json` | latest | JSON for MCP protocol, config |
| Binary serialize | `bytemuck` | latest | Safe cast for f32 slices in legacy-free cache |
| CLI | `clap` | latest | Subcommands: init, skeleton, tree |
| Testing | `wiremock` + `assert_cmd` + `tempfile` | latest | Mock Ollama, CLI tests, temp dirs |
| Benchmarks | `criterion` | latest | `cargo bench` performance validation |

### Research Insights: Crate Stack

**simsimd:** Returns cosine **distance** (not similarity) — always compute `1.0 - distance` to
get similarity. API: `f32::cosine(&vec_a, &vec_b) -> Option<f64>`. Returns `f64` even for `f32`
inputs. Auto-dispatches to AVX-512/AVX2/SSE at runtime — no compile-time CPU target needed.
Use `SpatialSimilarity` trait. ~8.2M ops/s for 1536-dim f32 vectors.

**tree-sitter Send/Sync:** In v0.25+, `Parser` `Send+Sync` is under scrutiny due to the new
`progress_callback` closure in `ParseOptions` (non-Send). The `thread_local!` pool pattern in
our plan is the correct approach — Parser stays on the thread that created it, no cross-thread
sharing. Pin to `tree-sitter = "0.25"` and avoid `parse_with_options` unless needed.

**notify-debouncer-full 0.7:** API is `new_debouncer(Duration, None, tx)` where `tx` can be
`mpsc::Sender` or `crossbeam_channel::Sender`. MSRV is 1.85. Use channel pattern (not callback)
for cleaner async integration with tokio via `tokio::sync::mpsc`.

**rkyv 0.8:** Major API change from 0.7 — uses `#[rkyv(attr)]` attribute macros instead of
separate derive traits. Zero-copy deserialization requires `unsafe` access to archived data.
Pin exact version. Include a format version byte in the header for forward compatibility.

## Architecture

### Crate Structure (single binary)

```
contextplus-rs/
  Cargo.toml
  src/
    main.rs                    # CLI args, MCP server startup, shutdown
    server.rs                  # ContextPlusServer + #[tool_router] + #[tool_handler]
    config.rs                  # Env var loading (same vars as TS)
    error.rs                   # ContextPlusError enum
    core/
      embeddings.rs            # OllamaClient, VectorStore, cosine (simsimd)
      tree_sitter.rs           # Native multi-lang parser, thread_local pool
      parser.rs                # extractHeader, analyzeFile, CodeSymbol
      walker.rs                # ignore-crate walker + CONTEXTPLUS_IGNORE_DIRS
      embedding_tracker.rs     # notify file watcher + debounced refresh
      clustering.rs            # Spectral clustering (nalgebra + kmeans)
      memory_graph.rs          # petgraph StableGraph + rkyv+mmap persistence
      hub.rs                   # Wikilink parser
    tools/
      context_tree.rs          # Token-aware AST tree
      file_skeleton.rs         # Signatures + line ranges
      semantic_search.rs       # File-level semantic search
      semantic_identifiers.rs  # Identifier search + call-site ranking
      blast_radius.rs          # Import/usage tracing
      static_analysis.rs       # Delegate to native linters
      propose_commit.rs        # Validate + write + restore point
      semantic_navigate.rs     # Spectral clustering + Ollama labels
      feature_hub.rs           # Obsidian wikilink navigator
      memory_tools.rs          # 6 memory graph tools
    git/
      shadow.rs                # Restore points (file-based backup)
    cache/
      rkyv_store.rs            # rkyv+mmap VectorStore format (zero-copy)
  tests/
    integration/
      mcp_server.rs            # Full MCP protocol integration tests
      ollama_mock.rs           # wiremock-based Ollama API mock
      search_tools.rs          # Search tool integration tests
      analysis_tools.rs        # Analysis tool integration tests
      navigation_tools.rs      # Navigation tool integration tests
      memory_tools.rs          # Memory graph integration tests
      file_management.rs       # Propose commit + restore point tests
    fixtures/
      sample-project/          # Minimal multi-lang project for testing
  benches/
    cache_load.rs              # mmap load benchmark
    cosine_scan.rs             # Cosine similarity benchmark (30K vectors)
    tree_sitter_parse.rs       # Parse benchmark across languages
    search_query.rs            # End-to-end warm query benchmark
```

### Shared State

```rust
#[derive(Clone)]
pub struct ContextPlusServer {
    state: Arc<SharedState>,
}

pub struct SharedState {
    pub config: Config,
    pub root_dir: PathBuf,
    pub search_store: RwLock<Option<Arc<VectorStore>>>,
    pub identifier_store: RwLock<Option<Arc<VectorStore>>>,
    pub memory_graph: RwLock<Option<MemoryGraph>>,
    pub ollama: OllamaClient,
}
```

Tools are `async fn` methods on `ContextPlusServer` via `#[tool_router]`. Each grabs an
`Arc<VectorStore>` clone from the RwLock (read lock held <1us, then released). Mmap'd stores are
inherently thread-safe for reads.

### Zero-Copy VectorStore (the key win)

```rust
pub struct VectorStore {
    _mmap: Mmap,                          // Keeps file mapped
    dims: u32,
    count: u32,
    vectors: *const f32,                  // Pointer into mmap, 64-byte aligned for SIMD
    key_index: HashMap<&'static str, u32>,// Built once on load (~2ms for 30K)
    archived: &'static ArchivedMeta,      // Zero-copy keys + hashes from rkyv
}
```

- **Load**: mmap syscall + build HashMap = ~2ms (vs 1,109ms in TS)
- **Cosine scan**: `simsimd::SpatialSimilarity::cosine()` per entry = ~15ms for 30K (SIMD auto-dispatch)
- **Write**: Build new file in temp → atomic rename. Old mmap stays valid for active readers.

### Tree-Sitter: Native vs WASM

Grammars compiled into the binary at build time via per-language crates. Parser is `!Send`, so use
`thread_local!` pool. No WASM VM, no dynamic loading, no `.wasm` file I/O.

```rust
thread_local! {
    static PARSERS: RefCell<HashMap<&'static str, Parser>> = RefCell::new(HashMap::new());
}
```

**Start with 10 grammars** (TS, TSX, JS, Python, Rust, Go, Java, C, C++, Bash). Add more behind
feature flags. The TS version supports 36 but most projects use <10 languages.

## Implementation Phases

### Phase 1: Core Infrastructure
**Files**: main.rs, server.rs, config.rs, error.rs, core/tree_sitter.rs, core/parser.rs, core/walker.rs

- MCP server with stdio transport (rmcp)
- Tree-sitter native parser (10 languages: TS, TSX, JS, Python, Rust, Go, Java, C, C++, Bash)
- File walker with `ignore` crate + CONTEXTPLUS_IGNORE_DIRS
- Subcommands: init, skeleton, tree
- `hashContent()` ported exactly (Java-style hash: `h = ((h << 5) - h + charCode) | 0`, then `.toString(36)`)

**Unit Tests (TDD)**:
- RED: `hashContent("hello")` matches TS output `"gg1h7l"`. GREEN: Port hash function.
- RED: Parse a TS file, assert symbols match expected names/kinds/lines. GREEN: Implement parser.
- RED: Parse each of the 10 supported languages. GREEN: Wire grammars.
- RED: Walk test directory, assert .gitignore paths excluded. GREEN: Implement walker.
- RED: Walk with CONTEXTPLUS_IGNORE_DIRS=node_modules,dist, assert excluded. GREEN: Wire env var.
- RED: MCP server responds to `initialize` with capabilities. GREEN: Wire rmcp.
- RED: Config reads OLLAMA_HOST, falls back to default. GREEN: Implement config.
- RED: `isDefinitionLine("export function foo()", "foo")` returns true. GREEN: Port regex.
- RED: `escapeRegex("foo.bar()")` returns `"foo\\.bar\\(\\)"`. GREEN: Port util.

### Phase 2: Embedding Engine
**Files**: core/embeddings.rs, cache/rkyv_store.rs

- OllamaClient with reqwest + connection pooling
- Adaptive batch/retry (binary split on context length errors, 0.75x shrink for single)
- VectorStore with simsimd cosine distance (`1.0 - distance` for similarity)
- rkyv+mmap persistence with format version header
- `hashContent` used for cache invalidation (same Java-style hash as TS)

**Unit Tests (TDD)**:
- RED: `hashContent("hello")` matches TS output. GREEN: Port hash function.
- RED: rkyv round-trip: save 100 entries → mmap load → verify all keys and vectors match. GREEN: Implement.
- RED: `cosine_similarity([1,0,0], [0,1,0])` returns 0.0 (simsimd distance=1.0). GREEN: Wire simsimd.
- RED: `cosine_similarity([1,0,0], [1,0,0])` returns 1.0 (simsimd distance=0.0). GREEN: Verify.
- RED: VectorStore.find_nearest returns correct top-K ordering. GREEN: Wire cosine scan.
- RED: Mock Ollama returns context-length error → verify binary split retry halves batch. GREEN: Implement.
- RED: Mock Ollama returns context-length error on single → verify 0.75x text shrink. GREEN: Implement.
- RED: Format version byte in rkyv header → version mismatch returns error. GREEN: Implement.

**Integration Tests**:
- RED: wiremock Ollama server → embed 10 texts → save rkyv cache → reload → vectors match. GREEN: Full pipeline.
- RED: wiremock Ollama returns 500 → OllamaClient retries with backoff. GREEN: Error handling.

### Phase 3: Search Tools
**Files**: tools/semantic_search.rs, tools/semantic_identifiers.rs

- SearchIndex: walk files → extract headers/symbols → hash-check → embed uncached → hybrid score
- IdentifierIndex: parse symbols → build identifier docs → embed → call-site ranking
- Shared file lines cache for blast_radius
- Index TTL (5 min), deduplicated concurrent builds
- `splitCamelCase`, `termCoverage` keyword scoring ported from TS

**Unit Tests (TDD)**:
- RED: `splitCamelCase("getUserById")` → `["get", "user", "by", "id"]`. GREEN: Implement.
- RED: `termCoverage("get user", ["getUserById"])` returns expected score. GREEN: Port scoring.
- RED: Build search index from fixture dir, verify doc count matches file count. GREEN: Implement.
- RED: Hybrid ranking with mock embeddings returns correct order. GREEN: Implement scoring.
- RED: Call-site ranking finds usages of `createForm` in fixture project. GREEN: Implement.
- RED: Hash-check skips unchanged files on re-index (embed count = 0). GREEN: Implement.
- RED: Concurrent index builds are deduplicated (only one runs). GREEN: Implement.
- RED: Index TTL expires after 5 min → next query rebuilds. GREEN: Implement.

**Integration Tests**:
- RED: wiremock Ollama → `semantic_code_search("authentication")` → returns ranked files. GREEN: End-to-end.
- RED: wiremock Ollama → `semantic_identifier_search("validate form")` → returns identifiers with call sites. GREEN: End-to-end.

### Phase 4: Analysis Tools
**Files**: tools/blast_radius.rs, tools/static_analysis.rs, tools/context_tree.rs, tools/file_skeleton.rs

- blast_radius: regex search for symbol across all files, `isDefinitionLine` filter
  (TS ref: `/tmp/contextplus-fork/src/tools/blast-radius.ts`)
- static_analysis: exec linters via `tokio::process::Command` (tsc, cargo check, etc.)
- context_tree: tree builder with token-aware pruning (Level 2→1→0)
  (TS ref: `/tmp/contextplus-fork/src/tools/context-tree.ts`)
- file_skeleton: analyzeFile + signature formatting

**Unit Tests (TDD)**:
- RED: `blast_radius("createForm", fixture_dir)` finds 3 usages across 2 files. GREEN: Implement.
- RED: blast_radius excludes the definition line when `fileContext` is provided. GREEN: Implement.
- RED: blast_radius with symbol not found returns "not used anywhere" message. GREEN: Implement.
- RED: `context_tree(fixture_dir, maxTokens=50000)` renders Level 2 (full). GREEN: Implement.
- RED: `context_tree(fixture_dir, maxTokens=100)` prunes to Level 0 (files only). GREEN: Implement.
- RED: `context_tree(fixture_dir, maxTokens=500)` prunes to Level 1 (headers only). GREEN: Implement.
- RED: `file_skeleton("fixture.ts")` output includes function signatures with line ranges. GREEN: Implement.
- RED: `estimateTokens("hello world")` returns `ceil(11/4)` = 3. GREEN: Implement.

**Integration Tests**:
- RED: MCP tool call `get_blast_radius({symbolName: "App", rootDir: fixture})` → returns formatted output. GREEN: Wire.
- RED: MCP tool call `get_context_tree({rootDir: fixture, depthLimit: 2})` → returns tree. GREEN: Wire.

### Phase 5: Navigation Tools
**Files**: tools/semantic_navigate.rs, tools/feature_hub.rs, core/clustering.rs, core/hub.rs

- Spectral clustering: affinity matrix → normalized Laplacian → SymmetricEigen → eigengap k → kmeans
- Ollama chat for cluster labeling
- Wikilink parser (regex: `\[\[path\]\]`)
- Feature hub orchestration

**Unit Tests (TDD)**:
- RED: Spectral cluster on known 2-cluster data (4 points, 2 tight pairs). GREEN: Implement.
- RED: Eigengap heuristic selects k=2 for 2-cluster data. GREEN: Implement.
- RED: Wikilink parser extracts `[[path/to/file]]` links from markdown. GREEN: Implement.
- RED: Wikilink parser handles `[[path|alias]]` syntax. GREEN: Implement.
- RED: `discoverHubs(fixture_dir)` finds .md files containing wikilinks. GREEN: Implement.

**Integration Tests**:
- RED: wiremock Ollama → `semantic_navigate("forms")` → returns labeled clusters. GREEN: End-to-end.

### Phase 6: Memory Graph
**Files**: core/memory_graph.rs, tools/memory_tools.rs

- petgraph StableGraph + HashMap<(label, type), NodeIndex> side-index
- rkyv persistence (replaces JSON — 10x smaller with embedded vectors)
- Temporal decay: `w * exp(-0.05 * days)`
- Auto-similarity linking (cosine > 0.72)
- BFS traversal with depth penalty
- 6 tool wrappers: upsert_memory_node, create_relation, search_memory_graph,
  prune_stale_links, add_interlinked_context, retrieve_with_traversal

**Unit Tests (TDD)**:
- RED: `upsert_node("auth", "concept", "Authentication system")` creates node. GREEN: Implement.
- RED: Re-upsert same (label, type) updates content, not duplicate. GREEN: Implement.
- RED: `create_relation("auth", "concept", "login", "feature", "CONTAINS")` creates edge. GREEN: Implement.
- RED: Duplicate relation updates weight instead of creating new edge. GREEN: Implement.
- RED: `prune(threshold=0.1)` removes edges below decay threshold. GREEN: Implement.
- RED: Prune also removes orphan nodes (no edges). GREEN: Implement.
- RED: `auto_link()` creates edges for pairs with cosine > 0.72. GREEN: Implement.
- RED: `traverse("auth", "concept", max_depth=2)` returns neighbors within 2 hops. GREEN: Implement.
- RED: Traverse applies depth penalty (deeper = lower score). GREEN: Implement.
- RED: `search(query_vector, top_k=5)` returns nearest nodes by cosine. GREEN: Implement.
- RED: rkyv round-trip: save graph → load → verify nodes and edges preserved. GREEN: Implement.

**Integration Tests**:
- RED: MCP `upsert_memory_node` → `search_memory_graph` → finds the node. GREEN: End-to-end.
- RED: MCP `create_relation` → `retrieve_with_traversal` → follows edges. GREEN: End-to-end.

### Phase 7: File Management
**Files**: tools/propose_commit.rs, git/shadow.rs

- Validation: 2-line header, no inline comments, max nesting 6, max lines 1000
- Shadow restore points: file-based backup in .mcp_data/backups/
  (TS ref: `/tmp/contextplus-fork/src/git/shadow.ts`)
- Manifest: .mcp_data/restore-points.json (max 100 entries)
- Cache invalidation on write/restore

**Unit Tests (TDD)**:
- RED: `validateHeader(content_with_header)` passes. GREEN: Implement.
- RED: `validateHeader(content_without_header)` fails with specific error. GREEN: Implement.
- RED: `createRestorePoint(files)` saves backup files + manifest entry. GREEN: Implement.
- RED: `restorePoint(id)` reads backup and writes to original location. GREEN: Implement.
- RED: Manifest capped at 100 entries (add 101, verify oldest removed). GREEN: Implement.
- RED: Restore point ID format matches `rp-{timestamp}-{random}`. GREEN: Implement.
- RED: `listRestorePoints()` returns all entries from manifest. GREEN: Implement.

**Integration Tests**:
- RED: MCP `propose_commit` → file written → `list_restore_points` → shows entry → `undo_change` → file restored. GREEN: Full cycle.

### Phase 8: Embedding Tracker
**Files**: core/embedding_tracker.rs

- notify + notify-debouncer-full (700ms debounce, 8 files/tick)
- Segment-based shouldTrack (not prefix — the bug we fixed in TS fork)
- Calls refreshFileSearchEmbeddings + refreshIdentifierEmbeddings in parallel
- CancellationToken for graceful shutdown

**Unit Tests (TDD)**:
- RED: `shouldTrack("src/main.rs")` returns true. GREEN: Implement.
- RED: `shouldTrack("node_modules/foo/bar.js")` returns false. GREEN: Implement.
- RED: `shouldTrack("dist/bundle.js")` returns false when "dist" in ignore dirs. GREEN: Implement.
- RED: Segment-based: `shouldTrack("src/node_modules_backup/file.ts")` returns true (not prefix match). GREEN: Implement.
- RED: Tracker debounces: 20 rapid events → single batch callback. GREEN: Implement.
- RED: Tracker respects max_files_per_tick (8): 20 files → 3 ticks. GREEN: Implement.
- RED: CancellationToken stops the tracker cleanly. GREEN: Implement.

**Integration Tests**:
- RED: Write file to watched dir → tracker fires refresh → cache updated. GREEN: End-to-end.

## Testing Strategy

### Coverage Targets
- **Overall: 80%+ line coverage** measured by `cargo llvm-cov`
- **Integration tests: 20%+ of total tests** (not just unit tests)
- **Every public function** has at least one unit test
- **Every MCP tool** has at least one integration test

### Test Infrastructure

#### wiremock (Mock Ollama API)
All embedding and chat operations go through Ollama HTTP API. wiremock provides a deterministic
mock server that:
- Returns pre-computed embedding vectors for known texts
- Simulates error conditions (context length errors, 500s, timeouts)
- Records request history for assertion (verify batch sizes, retry behavior)

```rust
// Example: Mock Ollama embedding endpoint
let mock_server = MockServer::start().await;
Mock::given(method("POST"))
    .and(path("/api/embed"))
    .respond_with(ResponseTemplate::new(200)
        .set_body_json(json!({
            "embeddings": [[0.1, 0.2, 0.3, /* ... 1024 dims */]]
        })))
    .mount(&mock_server).await;

let client = OllamaClient::new(&format!("http://{}", mock_server.address()));
```

#### tempfile (Isolated Test Directories)
Each test gets an isolated temp directory with a fixture project:
- `tests/fixtures/sample-project/` contains a minimal multi-language project
- Tests copy fixtures to tempdir, run tools, assert output
- Temp dirs auto-cleaned on test completion

#### assert_cmd (CLI Subcommand Tests)
Test the binary's CLI subcommands (init, skeleton, tree):
```rust
Command::cargo_bin("contextplus-rs")?
    .arg("skeleton")
    .arg("tests/fixtures/sample-project/src/main.ts")
    .assert()
    .success()
    .stdout(contains("function"));
```

#### criterion (Benchmark Suite)
Performance regression tests run in CI (not gating, but tracking):
- `cache_load`: mmap 120MB file, build HashMap — target <10ms
- `cosine_scan`: 30K vectors × 1024 dims — target <20ms
- `tree_sitter_parse`: Parse 10 files across languages — target <50ms total
- `search_query`: End-to-end warm semantic search — target <200ms

### Test Fixture Project
`tests/fixtures/sample-project/` contains:
```
sample-project/
  .gitignore              # Ignore node_modules, dist
  src/
    main.ts               # Imports from auth.ts, uses createForm
    auth.ts               # export function verifyToken(), class AuthService
    forms/
      createForm.ts       # export function createForm(), type FormConfig
      validateForm.ts     # export function validateForm()
    utils/
      helpers.py          # def hash_content(text), class CacheManager
      config.rs           # pub fn load_config(), pub struct Config
  docs/
    README.md             # Contains [[src/auth]] wikilinks
    ARCHITECTURE.md       # Contains [[src/forms/createForm]] wikilink
  node_modules/
    some-dep/
      index.js            # Should be ignored by walker
```

### Integration Test Categories

| Category | Count | What it tests |
|----------|-------|---------------|
| MCP protocol | 5 | Initialize, tool list, tool call, error response, resource read |
| Search tools | 6 | semantic_code_search, semantic_identifier_search with various params |
| Analysis tools | 6 | blast_radius, context_tree, file_skeleton, static_analysis |
| Navigation | 3 | semantic_navigate, feature_hub with wikilinks |
| Memory graph | 5 | upsert, create_relation, search, traverse, prune |
| File management | 3 | propose_commit, list_restore_points, undo_change |
| Error handling | 4 | Ollama down, invalid tool args, file not found, permission denied |
| **Total** | **32** | |

### CI Pipeline
```yaml
# .github/workflows/ci.yml
- cargo fmt --check
- cargo clippy -- -D warnings
- cargo test                    # All unit + integration tests
- cargo llvm-cov --lcov         # Coverage report
- cargo bench -- --output-format bencher  # Benchmark (tracking only)
```

## Security Hardening

### Path Traversal Prevention
All tool inputs that accept file paths (`rootDir`, `targetPath`, `filePath`) must be validated:
```rust
fn validate_path(root: &Path, user_path: &str) -> Result<PathBuf> {
    let resolved = root.join(user_path).canonicalize()?;
    if !resolved.starts_with(root.canonicalize()?) {
        return Err(ContextPlusError::PathTraversal(user_path.to_string()));
    }
    Ok(resolved)
}
```
- Applies to: `get_blast_radius`, `get_file_skeleton`, `get_context_tree`, `propose_commit`,
  `undo_change`, `semantic_code_search`, `semantic_identifier_search`
- `rootDir` is set once at server startup and cannot be overridden by tool calls

### Command Injection Prevention
`static_analysis` executes external linters. **Never** pass tool arguments directly to shell:
```rust
// GOOD: Use Command with explicit args
Command::new("npx").arg("tsc").arg("--noEmit").arg("--pretty")
// BAD: shell string interpolation
Command::new("sh").arg("-c").arg(format!("tsc {}", user_input))
```
- Allowlist of supported linters: `tsc`, `eslint`, `cargo check`, `ruff`
- No arbitrary command execution

### mmap Safety
- **Atomic rename for writes**: Build new rkyv file in temp → `fs::rename()` → old mmap remains
  valid for active readers (OS keeps pages mapped until last reference dropped)
- **File truncation**: If external process truncates the cache file while mmap'd, reads may SIGBUS.
  Mitigation: use `MAP_PRIVATE` (copy-on-write) and catch SIGBUS as error, not crash.
- **All unsafe mmap code confined to `cache/rkyv_store.rs`** — single audit surface

### Input Validation
- `symbolName` in `blast_radius`: `escapeRegex()` before constructing `Regex` (prevents ReDoS)
- `query` in search tools: truncate to 2000 chars (prevents oversized Ollama requests)
- `topK` capped at 50 (prevents OOM from sorting huge result sets)
- Memory graph node content: truncate to 10,000 chars
- Restore point ID: validate format `rp-{digits}-{alphanumeric}` before path construction

### Supply Chain
- All crates are well-established (rmcp is official MCP SDK, tree-sitter is GitHub's parser,
  ignore is from ripgrep, notify from the Rust ecosystem core)
- `cargo audit` in CI pipeline to catch known vulnerabilities
- Pin major versions in Cargo.toml

## Performance Targets

Baselines reflect the **already-optimized** TS fork (post O1-O5):

| Operation | TS (optimized) | Rust (expected) | Factor |
|-----------|---------------|-----------------|--------|
| Cache load (120MB) | 115ms (VectorStore) | ~2ms (mmap+HashMap) | 57x |
| Cosine scan 30K | ~50ms (JS loop) | ~15ms (LLVM SIMD) | 3x |
| Warm semantic_code_search | ~1.5s | <100ms | 15x |
| Warm semantic_identifier_search | ~1.7s | <200ms | 8x |
| Tree-sitter parse | 5-20ms (WASM, deduped) | 1-5ms (native) | 4x |
| Context tree (full project) | ~2s | <500ms | 4x |
| Memory graph persist | JSON ~8MB | rkyv ~800KB | 10x smaller |
| No-op file refresh | 870ms | <10ms | 87x |
| Full cache rebuild (cold) | ~45s | ~20s | 2x |

Note: The biggest Rust wins are on cache load (mmap eliminates readFile+copy) and no-op refresh
(zero I/O when nothing changed). The TS VectorStore optimization (O2) already brought load from
1,109ms to 115ms — Rust's mmap takes it to ~2ms.

## Env Var Compatibility (same as TS)

OLLAMA_HOST, OLLAMA_EMBED_MODEL, OLLAMA_CHAT_MODEL, OLLAMA_API_KEY,
CONTEXTPLUS_EMBED_BATCH_SIZE, CONTEXTPLUS_EMBED_TRACKER, CONTEXTPLUS_EMBED_TRACKER_DEBOUNCE_MS,
CONTEXTPLUS_EMBED_TRACKER_MAX_FILES, CONTEXTPLUS_IGNORE_DIRS

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| rmcp API instability (v0.16 → v1.x) | Medium | High | Pin exact version, wrap in thin adapter layer |
| Tree-sitter grammar crate version mismatches | Medium | Medium | Pin tree-sitter = "0.25", use `tree-sitter-language` compat crate |
| simsimd f32 cosine returns distance not similarity | High | High | Unit test day 1: verify `1.0 - distance` mapping |
| rkyv 0.8 archive format changes | Low | High | Pin rkyv version, include format version byte in header |
| mmap safety: file truncation → SIGBUS | Low | Critical | MAP_PRIVATE, SIGBUS handler, atomic rename for writes |
| 10 grammar crates = moderate compile times | Medium | Low | Feature flags for optional grammars, incremental builds |
| Ollama API differences (TS `ollama` vs raw HTTP) | Medium | Medium | Test against real Ollama early in Phase 2 |
| hashContent divergence (JS bitwise vs Rust) | Medium | High | Unit test with known TS outputs before any cache work |
| tree-sitter Parser Send/Sync in 0.25+ | Medium | Medium | thread_local! pool, never share Parser across threads |

## Acceptance Criteria

### Functional
- [ ] All 17 MCP tools return identical results to TS version (snapshot tests)
- [ ] 1 resource (`contextplus://instructions`) works
- [ ] Subcommands (init, skeleton, tree) work identically
- [ ] Same env vars, same `.mcp_data` directory layout
- [ ] Drop-in replacement: swap binary in MCP config, all tools work

### Performance
- [ ] Cache load < 10ms (vs 1,109ms TS)
- [ ] Warm semantic_code_search < 200ms (vs ~2s TS)
- [ ] Warm semantic_identifier_search < 400ms (vs ~3.7s TS)
- [ ] No-op file refresh < 50ms (vs 870ms TS)
- [ ] Binary size < 50MB (static, all grammars included)

### Testing
- [ ] 80%+ line coverage (`cargo llvm-cov`)
- [ ] 20%+ of tests are integration tests (35+ integration / 100+ unit)
- [ ] All integration tests use wiremock (no real Ollama dependency)
- [ ] Benchmark suite tracks all performance targets
- [ ] CI pipeline: fmt + clippy + test + coverage + bench

### Quality Gates (per phase)
- [ ] All TDD tests pass (RED confirmed before GREEN)
- [ ] `cargo clippy` clean (no warnings)
- [ ] No `unsafe` outside VectorStore mmap and tree-sitter FFI
- [ ] Error types use thiserror, no `.unwrap()` in tool handlers
- [ ] `cargo audit` clean (no known vulnerabilities)

## Design Decisions

1. **10 grammars, not 36**: Start with TS, TSX, JS, Python, Rust, Go, Java, C, C++, Bash. Add more
   behind feature flags. Most projects use <10 languages.

2. **simsimd for cosine**: Best-in-class SIMD auto-dispatch (AVX-512/AVX2/SSE). Returns distance,
   not similarity — always `1.0 - distance`. ~8.2M ops/s for 1536-dim f32.

3. **No legacy cache reader**: Clean break from TS. First run does a fresh cold embed (~20s with
   Rust's parallel walker). No migration code to maintain. Simpler codebase.

4. **rkyv everywhere**: Both VectorStore caches AND memory graph use rkyv+mmap. Zero-copy load for
   all persistent data. Memory graph with embedded vectors benefits just as much — rkyv ~800KB vs
   JSON ~8MB, load in <1ms vs ~50ms.

5. **petgraph for memory graph**: Provides correct graph algorithms (BFS, neighbor iteration) with
   zero bugs. Manual HashMap+Vec would be ~300 lines of error-prone code for no benefit.

6. **Single crate, not workspace**: The binary is ~3000 lines total. Workspace overhead isn't
   justified until >10K lines.

## Parallel Agent Strategy

Phases can be partially parallelized:

```
Phase 1 (Core) ─────────────────────────┐
                                         ├─> Phase 3 (Search Tools)
Phase 2 (Embedding Engine) ─────────────┤
                                         ├─> Phase 4 (Analysis Tools)
                                         ├─> Phase 5 (Navigation Tools)
                                         ├─> Phase 6 (Memory Graph)
                                         └─> Phase 7 (File Management)
                                                    │
Phase 8 (Embedding Tracker) <───────────────────────┘
```

- Phases 1+2 are sequential (2 depends on 1)
- Phases 3-7 can run in parallel after 1+2 complete (independent tool implementations)
- Phase 8 depends on Phase 3 (refresh functions)

**Suggested agent assignment**: 2 agents for Phase 1+2 (foundation), then 3 agents splitting
Phases 3-7 (Agent A: 3+4, Agent B: 5+6, Agent C: 7), then 1 agent for Phase 8.

## Repository Setup

New repo, based on the architecture and algorithms from the TS fork (`mrsufgi/contextplus`
branch `fix/merge-cache-on-save`). All optimizations (O1-O5, hash-check) are carried forward
into the Rust design — they informed the architecture, not just the TS fixes.

```bash
mkdir contextplus-rs && cd contextplus-rs
cargo init --name contextplus-rs
git init && git add . && git commit -m "chore: init contextplus-rs"
```

- New GitHub repo: `mrsufgi/contextplus-rs`
- Internal crate name: `contextplus_rs` (Rust snake_case)
- MCP server name in protocol: `"contextplus"` (same as TS — drop-in replacement)
- Resource URI: `contextplus://instructions` (same as TS)
- Env vars: same as TS (`CONTEXTPLUS_*`, `OLLAMA_*`) — full API compatibility
- Data dir: `.mcp_data/` (same layout as TS)
- TS fork reference: `/tmp/contextplus-fork/src/` for algorithm porting

## Verification

### Unit Test Verification
```bash
cargo test                          # All tests
cargo test --lib                    # Unit tests only
cargo llvm-cov --html               # Coverage report
```

### Integration Test Verification
```bash
cargo test --test '*'               # Integration tests only
cargo test --test mcp_server        # MCP protocol tests
cargo test --test search_tools      # Search tool tests
```

### Performance Verification
```bash
cargo bench                         # Full benchmark suite
cargo bench -- cache_load           # Single benchmark
```

### Drop-in Replacement Verification
1. Build: `cargo build --release`
2. Update MCP config: replace `node /path/to/contextplus/build/index.js` with `./target/release/contextplus-rs`
3. Open Claude Code, verify all 17 tools work
4. Compare outputs: run same queries against TS and Rust, diff results

### Feature Parity Snapshot Test
1. Run TS fork against `tests/fixtures/sample-project/` → capture all 17 tool outputs as JSON
2. Run Rust binary against same fixture → capture outputs
3. `diff` the outputs (ignore timing fields, normalize paths)
4. Any difference = regression
