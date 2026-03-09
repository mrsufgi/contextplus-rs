# Architecture

This document describes the internal architecture of contextplus-rs, the Rust MCP server for
semantic code analysis.

## Data Flow

A tool call flows through the system in five stages:

```
MCP Request (JSON-RPC over stdio)
  -> server.rs: call_tool() parses request
    -> dispatch() matches tool name to handler
      -> ensure_project_cache() walks filesystem + reads file contents (lazy, TTL-cached)
        -> tool handler (e.g., context_tree, blast_radius) processes cached data
          -> formatted text response returned to client
```

1. **MCP Request** -- The `rmcp` crate handles JSON-RPC framing over stdin/stdout. Each request
   contains a tool name and a JSON arguments map.

2. **Dispatch** -- `dispatch_inner()` matches the tool name string to a handler method via a flat
   `match` expression. Unknown tools return an error immediately.

3. **Project Cache** -- `ensure_project_cache()` builds a `ProjectCache` containing all walked file
   entries and their line contents. This is stored behind `RwLock<Option<Arc<ProjectCache>>>` and
   reused across calls until TTL expires or the file watcher invalidates it.

4. **Tool Handler** -- Each handler extracts its parameters from the JSON args, calls into the
   corresponding `src/tools/*.rs` module, and formats the result as MCP content.

5. **Response** -- Results are returned as `CallToolResult` with text content. Errors are caught at
   the `dispatch()` boundary and converted to error text.

## Caching Strategy

Three independent cache layers minimize redundant work:

### Project Cache

- **Scope:** File entries (paths + metadata) and file line contents for the entire project.
- **Storage:** `RwLock<Option<Arc<ProjectCache>>>` in `SharedState`.
- **Invalidation:** TTL-based (configurable, default 60s). Also invalidated by the file watcher
  (`embedding_tracker`) when files change on disk.
- **Lifecycle:** Built lazily on first tool call via `ensure_project_cache()`. Subsequent calls
  reuse the `Arc` until TTL expires.

### Embedding Cache

- **Scope:** Maps `file_path -> (content_hash, embedding_vector)` for semantic search.
- **Storage:** `RwLock<HashMap<String, CacheEntry>>` in memory, persisted to disk via
  `cache/rkyv_store.rs` as `.rkyv` files in `.mcp_data/`.
- **Invalidation:** Content-hash based. On each search, the file's current content hash is compared
  against the cached hash. Only files with changed hashes are re-embedded.
- **Persistence:** `rkyv` (zero-copy serialization) + `memmap2` for loading. The cache file
  contains a version header followed by rkyv-serialized `CacheData` (dims, keys, hashes, vectors).

### Identifier Index

- **Scope:** Parsed symbols (functions, classes, etc.) from all project files, plus their embedding
  vectors for `semantic_identifier_search`.
- **Storage:** `RwLock<Option<Arc<IdentifierIndex>>>` in `SharedState`.
- **Invalidation:** TTL (300s) + file count change detection. If the number of walked files differs
  from the cached count, the index is rebuilt.
- **Contents:** `Vec<IdentifierDoc>` (symbol metadata), flat `Vec<f32>` vector buffer, and
  dimensionality info.

## Memory Layout

### VectorStore

The `VectorStore` (in `core/embeddings.rs`) holds all embedding vectors in a flat contiguous array
for cache-friendly SIMD scanning:

```
VectorData::Owned(Vec<f32>)    -- heap-allocated, built from fresh embeddings
VectorData::Mmap { ptr, len }  -- zero-copy pointer into rkyv cache file
```

**Layout:** `dims x count` contiguous `f32` elements. Vector `i` occupies
`data[i * dims .. (i + 1) * dims]`.

**Key fields:**
- `dims: usize` -- embedding dimensionality (e.g., 1024 for snowflake-arctic-embed2)
- `keys: Vec<String>` -- file paths, parallel to vector rows
- `hashes: Vec<String>` -- content hashes, parallel to vector rows
- `key_index: HashMap<String, usize>` -- O(1) lookup from file path to row index
- `vectors: VectorData` -- either Owned or Mmap

**Mmap mode:** When loading from disk, `VectorStore::from_mmap()` creates a `VectorData::Mmap` that
points directly into the memory-mapped rkyv file. The `Arc<Mmap>` keeps the mapping alive for the
lifetime of the store. No deserialization or copying occurs -- the f32 data is read in-place.

## Performance Architecture

### SIMD Cosine Similarity

`find_nearest()` uses `simsimd::SpatialSimilarity::cosine()` for pairwise distance computation.
The `simsimd` crate auto-dispatches to the best available instruction set (AVX-512, AVX2, SSE4, or
scalar fallback). On a 30K-vector scan with 1024 dimensions, this completes in ~4ms.

### Rayon Parallelism

For large vector stores (>2K vectors), `find_nearest()` uses `rayon::par_chunks()` to parallelize
the cosine scan across CPU cores. Each chunk computes distances independently, then results are
merged. Below the threshold, sequential iteration avoids thread pool overhead.

### Partial Sort for Top-K

Instead of sorting all N distances, `select_nth_unstable_by()` is used to partition around the k-th
element in O(N) average time, then only the top-k results are sorted. This avoids O(N log N) when
only the top 5-10 results are needed.

### Thread-Local Tree-Sitter Parsers

Tree-sitter `Parser` objects are not `Send`/`Sync`. The parser pool uses `thread_local!` storage so
each Rayon worker thread gets its own parser instance, avoiding mutex contention during parallel
file parsing.

### Content-Hash Cache Invalidation

Each file's content is hashed (via `hash_content()`) before embedding. On subsequent searches, only
files whose hash differs from the cached hash are re-embedded. Unchanged files skip the Ollama API
call entirely, making warm searches near-instant.

## Context Tree Pruning

The `get_context_tree` tool supports two parameters for controlling output size:

- **`depth_limit`** -- Filters entries before tree construction. Only entries with
  `depth <= depth_limit` are included. This is applied before any token-based pruning.
- **`max_tokens`** -- Controls token-aware pruning (default: 50,000). The tree is rendered at
  three detail levels, falling back to lower levels when the output exceeds the budget:
  - **Level 2:** Full content (symbols + headers)
  - **Level 1:** Headers only (symbols pruned)
  - **Level 0:** File names only (headers + symbols pruned)

## Adding a New Tool

1. **Create the tool module** -- Add `src/tools/my_tool.rs` with your pure logic functions and
   unit tests. Keep the module focused: parse inputs, compute results, format output.

2. **Register the module** -- Add `pub mod my_tool;` to `src/tools/mod.rs`.

3. **Add a handler method** -- In `server.rs`, add an `async fn handle_my_tool()` method on
   `ContextPlusServer`. Extract parameters from `args`, call your tool module, and return a
   `CallToolResult`.

4. **Wire up dispatch** -- Add a match arm in `dispatch_inner()`:
   ```rust
   "my_tool" => self.handle_my_tool(args).await,
   ```

5. **Add the tool definition** -- In `server.rs`, add a `Tool` entry in `tool_definitions()` with
   the tool's name, description, and JSON schema for its parameters.

6. **Add tests** -- Unit tests go in `src/tools/my_tool.rs` under `#[cfg(test)] mod tests`.
   Integration tests for dispatch go in the `#[cfg(test)]` block at the bottom of `server.rs`.

7. **Update documentation** -- Add the tool to the Tools table in `README.md`.
