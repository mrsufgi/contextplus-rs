---
title: 'feat(transport): multi-worktree daemon with CoW chunk + memory indexes'
type: feat
status: active
date: 2026-05-06
---

**Target repo:** `mrsufgi/contextplus-rs` (commits land on existing branch
`feat/per-workspace-daemon`, extending PR #70 in place — do not open a new branch).

# feat(transport): multi-worktree daemon with CoW chunk + memory indexes

## Summary

Extend PR #70 in place so a single daemon serves every git worktree of a repo. The socket relocates
to the primary worktree, `SharedState` becomes a registry of per-ref indexes, and both the chunk
index and the memory graph fork copy-on-write from primary. Agents in worktrees join an existing
daemon, pay only the diff-vs-primary embedding cost on first call, and have their memory layer
auto-fold into primary when their branch's commits become ancestors of primary's HEAD.

---

## Problem Frame

PR #70 introduced one daemon per workspace. From the daemon's POV a git worktree IS a separate
workspace: each spawn creates `<worktree>/.mcp_data/contextplus.sock`, a fresh `ContextPlusServer`,
and rebuilds the entire 130 MB+ embedding cache from scratch. With 26 active swarm worktrees and
short-lived agents in each, this turns "agent does small work that needs MCP" into "agent waits
30–120s for warmup or hits `Transport closed`". Worktrees of the same repo share 95%+ of files
byte-for-byte; the duplicated embedding work is pure waste against a throughput-bound Ollama. The PR
also embeds 199 k files in worktrees that are 4.9 GB of node_modules / .yarn / build artifacts
because the walker's ignore set is too narrow.

---

## Requirements

- R1. The Unix socket for any worktree of a repo resolves to the same path under the primary
  worktree's `.mcp_data/`. Bridges spawned from primary, secondary, or `.git/worktrees/<name>/` all
  connect to one daemon.
- R2. The daemon serves any number of refs concurrently from a single `SharedState`. A new ref
  attaches without restarting the daemon.
- R3. A new ref forks copy-on-write from a base ref (default:
  `merge-base HEAD <primary's tracked branch>`); only the diff vs. base gets re-embedded.
- R4. Embeddings are content-addressed: identical chunk text across refs deduplicates to one CAS
  blob.
- R5. Tool inputs and outputs at the session boundary are translated into the caller's
  worktree-relative path space. A tool result returned to bridge B must never contain a path inside
  another bridge's worktree.
- R6. Memory-graph forks CoW from primary: reads chain through to parent on miss, writes go to
  overlay only.
- R7. When the daemon detects that a ref's HEAD has become an ancestor of primary's HEAD (no GitHub
  round-trip required), the ref's memory overlay folds into primary via the merge ladder: clean →
  identical-content → smart-merge → `needs_rebuild`.
- R8. Memory nodes carry references to the chunk hashes they were derived from; when those chunks
  change, nodes are flagged `needs_rebuild` and hidden from search until lazily re-derived on next
  direct access.
- R9. The walker uses `ignore`-crate gitignore semantics (already present) plus a substantially
  expanded `DEFAULT_IGNORE_DIRS` and a hard-coded extension/file blocklist (`*.map`, `*.min.*`,
  `*.pb.go`, `*_pb.ts`, `*.connect.ts`, lockfiles other than `yarn.lock`/`Cargo.lock`/`go.sum`).
- R10. `CONTEXTPLUS_TRANSPORT=auto|stdio|daemon` keeps current semantics; `auto` continues to fall
  back to stdio when `.mcp_data/` is unwritable. Existing stdio-only deployments are unaffected.
- R11. CI returns to green: clippy clean, `tests/daemon_transport.rs` plus new tests pass, codecov
  patch coverage on new transport code clears the 80% target.

---

## Scope Boundaries

- Remote / S3-backed CAS — out
- Hot reload of daemon binary — out
- Merge-back path _from_ primary to active refs (refs see primary's new state via read-through;
  their own deltas stay where authored) — out
- A one-time migrator for old path-keyed `.rkyv` files — out (drop-and-rebuild on first daemon start
  under new format)
- GitHub PR-merge integration — out (auto-merge is local-git ancestor detection only)

### Deferred to Follow-Up Work

- Memory-graph remote sync between machines: separate PR after CAS.
- `needs_rebuild` proactive sweeper (vs. lazy on-access): separate PR; requires usage data first.
- Multi-machine CAS (S3-backed) once content-addressed layout proves stable in single-machine usage.

---

## Context & Research

### Relevant Code and Patterns

- `src/transport/paths.rs` — `daemon_dir(root_dir)` is the seam to relocate the socket to primary
  worktree. Override env var `CONTEXTPLUS_DAEMON_SOCKET_PATH` already supported and must keep
  working.
- `src/transport/daemon.rs` — `run_if_owner` (L424) instantiates
  `ContextPlusServer::new(root_dir, config)` once. `accept_loop` (L293) spawns one task per
  connection. Multi-ref work plugs into the per-connection task.
- `src/transport/dispatch.rs` — currently the under-tested file (53% patch coverage in PR #70).
  Adding the path-translation layer plus its tests here clears the coverage gap as a side effect.
- `src/server.rs:85` — `SharedState` holds a single `root_dir`, `embedding_cache`, `memory_graph`,
  `search_index_cache`, `tracker_handle`. Refactor target: split into shared (config, ollama, drain
  flags, inflight, idle_monitor) and per-ref (cache, tracker, memory).
- `src/cache/rkyv_store.rs:67` — `cache_dir(root_dir)` already calls `worktree_name(root_dir)` and
  isolates per-worktree under a `worktrees/<name>/` subdir. **This is per-ref isolation but not
  CoW** — the foundation is already there; CAS + parent-overlay layer over it.
- `src/core/walker.rs` — already uses ripgrep's `ignore` crate with `git_ignore(true)`,
  `git_exclude(true)`. R9 is mostly an expansion of `DEFAULT_IGNORE_DIRS` in `src/config.rs:105` and
  adding extension globs, NOT a wholesale rewrite to shell out to `git ls-files`.
- `src/core/memory_graph.rs:109` — `MemoryNode` currently has
  `id, label, content, embedding, created_at, last_accessed, access_count, metadata`. CoW +
  needs_rebuild adds `parent_ref` and `references_chunk_hashes` and a `needs_rebuild: bool` flag.
- PR #69 (merged) — graceful drain protocol the daemon already uses on SIGTERM/SIGINT. Auto-merge
  folder must respect drain (no merges accepted while draining).

### Institutional Learnings

- Render-style "hidden file replaces directory" parser strictness (from
  `feedback_render_blueprint_silent_failures.md` adjacent pattern): `.git`-as-file pointer parsing
  must `tracing::warn!` and continue on malformed pointers, never bail the walker.
- `feedback_dashboards_prune_dont_flag.md` analog: stale-worktree entries in cache should be dropped
  on load, not flush — applies to memory-graph node staleness on rebuild.

### External References

- `git rev-parse --git-common-dir` returns the primary worktree's gitdir for any worktree, including
  the primary itself. Most reliable resolution path.
- ripgrep `ignore` crate `WalkBuilder::add_custom_ignore_filename`: lets us add ad-hoc ignore
  entries without modifying user gitignore.
- `sha2` crate (already a transitive dep via various paths) for SHA-256 of chunk text.

---

## Key Technical Decisions

- **Cache addressing key**: `BLAKE3(chunk_text)` (faster than SHA-256, same blast-radius). One blob
  per (model, chunk-hash) pair, regardless of which file or ref it came from.
- **Fork base for new refs**: `git merge-base HEAD <primary's tracked-branch>`. If a sibling ref
  already exists at the same SHA, fork from the sibling instead (cheap content reuse). Primary
  tracked branch defaults to the upstream branch returned by
  `git rev-parse --abbrev-ref --symbolic-full-name @{u}` on primary; falls back to
  `main`/`master`/`HEAD` in that order.
- **Path translation home**: `src/transport/dispatch.rs`. Translation runs at the dispatch boundary
  so every tool gets it for free; tool implementations stay path-agnostic.
- **Single embedding queue**: bounded mpsc channel, 4 concurrent workers feeding Ollama; queue is
  owned by the daemon-shared layer (not per-ref) so a single Ollama is the universal bottleneck. No
  priority lanes in this PR.
- **Memory merge ladder** (in order):
  1. Clean — ref's overlay touched nodes primary hasn't touched since fork → atomic rename-publish.
  2. Identical-content overlap — both sides made the same change → trivial; treat as clean.
  3. Smart-merge eligible — additive only (union of edges, max(access_count), latest last_accessed),
     no contradictory body → merge and publish.
  4. True conflict — flag the merged node as `needs_rebuild`; next direct access re-derives from
     current code state, discarding both contested versions. Logged at `INFO` per node.
- **`needs_rebuild` semantics**: nodes flagged `needs_rebuild` are hidden from search/traversal
  results until rebuilt. Rebuild = re-embed body + re-validate edge endpoints; if rebuild fails
  (referenced symbol gone, etc.), node is pruned via `prune_stale_links` path.
- **Auto-merge trigger**: notify-rs watcher on primary's gitdir. On HEAD advance, daemon runs
  `git merge-base --is-ancestor <ref_head> <primary_head>` for each active ref; matches enter the
  merge ladder. Auto-merge is suppressed during drain.
- **Drop-and-rebuild on format change**: existing `.rkyv` files at old path-keyed paths are deleted
  on first daemon start under the new format. Embeddings rebuild from Ollama; memory graph rebuilds
  empty (no migration of nodes — accept the data loss for v1).

---

## Open Questions

### Resolved During Planning

- **Should auto-merge run during drain?**: No — drain rejects new connections and finishes
  in-flight; treat auto-merge as a write that mustn't start during shutdown. Watcher pauses when
  `state.draining` is set.
- **Should the walker shell out to `git ls-files` instead of using the `ignore` crate?**: No — the
  existing `ignore`-crate path already respects `.gitignore` / `.git/info/exclude` and is faster
  than subprocessing. Just expand defaults.
- **Where does path translation live?**: `src/transport/dispatch.rs`, the existing under-tested
  file. Keeps the boundary in one place and brings dispatch.rs coverage to target as a side effect.
- **Should memory CoW have an explicit promotion API for an agent to "save this to primary"?**: No —
  auto-merge on local branch merge is the only promotion path in this PR. Deferred to follow-up.

### Deferred to Implementation

- Exact `RefIndex` field set: emerges from the SharedState refactor (U3); current `SharedState` has
  12 fields, some clearly per-ref (cache, tracker, search_index_cache, memory_graph) and some
  clearly shared (config, ollama, drain flags, inflight) — the boundary between them is mechanical
  but tedious to enumerate without touching code.
- Whether `tracker_handle` becomes per-ref or shared: depends on how the embedding tracker handles
  multiple roots; if notify-rs supports a single watcher with per-path callbacks, shared is simpler.
  Resolved in U4.
- The exact field layout of the new on-disk format: emerges from U6's CAS work. Plan-time the only
  commitment is "drop and rebuild" for the migration story.
- HNSW per-ref construction strategy (clone parent's index + insert overlay vs. rebuild from union):
  measure during U6 and pick.

---

## High-Level Technical Design

> _This illustrates the intended approach and is directional guidance for review, not implementation
> specification._

### Topology

```
                  ┌────────────────────── per machine ──────────────────────┐
                  │  contextplus daemon (one per primary repo gitdir)        │
                  │   ├─ shared state (config, Ollama, embed queue,          │
                  │   │   drain flags, idle monitor)                         │
                  │   ├─ HashMap<RefId, Arc<RwLock<RefIndex>>>               │
                  │   │     │                                                │
                  │   │     ├─ Primary ref (parent=None)                     │
                  │   │     │    ├─ chunk manifest: path → [(idx, hash)]     │
                  │   │     │    ├─ memory graph (canonical)                 │
                  │   │     │    └─ HEAD watcher                             │
                  │   │     ├─ Worktree ref A (parent=Primary)               │
                  │   │     │    ├─ chunk manifest overlay (diff only)       │
                  │   │     │    └─ memory graph overlay (diff only)         │
                  │   │     └─ Worktree ref B (parent=Primary)               │
                  │   │          └─ ...                                      │
                  │   ├─ CAS: BLAKE3(chunk_text) → embedding (shared)        │
                  │   └─ Unix socket at <primary>/.mcp_data/contextplus.sock │
                  └──────────────────────────────────────────────────────────┘
                            ▲                ▲                ▲
                            │ stdio          │ stdio          │ stdio
                       ┌────┴─────┐    ┌─────┴────┐    ┌──────┴────┐
                       │ bridge A │    │ bridge B │    │ bridge C  │
                       │ Claude   │    │ Codex    │    │ swarm-1   │
                       │ root=ws  │    │ root=ws  │    │ root=wt-X │
                       └──────────┘    └──────────┘    └───────────┘
```

### Bridge↔Daemon handshake (extends PR #70)

```
bridge → daemon: register_session {
   client_root: PathBuf,           // what --root-dir resolved to
   head_sha: GitSha,               // git rev-parse HEAD in client_root
   client_pid: u32,
}

daemon resolves:
   primary = git_common_dir(client_root)
   ref_id  = blake3(canonical(client_root))
   if ref_id not in SharedState.refs:
       fork(parent = ref_at(merge_base) ?? primary, head = head_sha)

daemon → bridge: session_ready {
   session_id: SessionId,
   ref_id: RefId,
   status: Ready | Warming { eta_ms },
}
```

### Memory merge ladder

```
event: primary HEAD advances from <old_head> to <new_head>
  for each active ref R:
    if git merge-base --is-ancestor R.head <new_head>:
        for each (node_id, overlay_node) in R.memory_overlay:
            base = primary.memory.lookup(node_id)
            classify(base, overlay_node, primary_current=primary.memory.lookup(node_id)):
                None (new node)              -> publish to primary
                Equal (same content)         -> noop
                Additive (edges only)        -> smart-merge into primary
                Conflicting bodies           -> mark needs_rebuild on primary's copy
        clear R.memory_overlay
        keep R alive (still serving requests)
```

### CAS lookup chain

```
chunk_lookup(ref_id, file_path, chunk_idx) ->
    ref = refs[ref_id]
    while ref:
        if (file_path, chunk_idx) in ref.manifest:
            chunk_hash = ref.manifest[(file_path, chunk_idx)]
            return cas[chunk_hash]   // shared blob
        ref = ref.parent
    return Miss   // queue for embedding
```

---

## Implementation Units

- U1. **Aggressive walker defaults**

**Goal:** Drop the corpus the walker emits to actual source by expanding `DEFAULT_IGNORE_DIRS` and
adding hard-coded file-extension/file-name globs. No structural change; pure default-tuning.

**Requirements:** R9

**Dependencies:** None — ships first; reduces noise for every later unit's tests.

**Files:**

- Modify: `src/config.rs` (extend `DEFAULT_IGNORE_DIRS`)
- Modify: `src/core/walker.rs` (add hard-coded file-glob blocklist applied after `ignore` crate)
- Test: `src/core/walker.rs` (existing test module; add cases)

**Approach:**

- Expand `DEFAULT_IGNORE_DIRS` to include: `.yarn`, `.pnp`, `.pnp.*` (glob), `.pnpm-store`,
  `vendor`, `.moon`, `.turbo`, `.nx`, `.rush`, `.changeset/.cache`, `out`, `lib`, `_build`,
  `.svelte-kit`, `.astro`, `.vercel`, `.netlify`, `.idea`, `.vscode`, `.fleet`, `.zed`,
  `__pycache__` (already), `.mypy_cache`, `.ruff_cache`, `.pytest_cache`, `.tox`, `.venv`, `venv`,
  `.eggs`, `.bundle`, `bin`, `obj`, `__snapshots__`, `htmlcov`, `.nyc_output`, `.swc`, `.vite`,
  `.esbuild`, `.webpack`, `.rollup.cache`.
- Add file-glob blocklist (applied as a post-filter): `*.map`, `*.min.js`, `*.min.css`,
  `*.bundle.js`, `*.pb.go`, `*_pb.ts`, `*.connect.ts`, `*.snap`, `package-lock.json`,
  `pnpm-lock.yaml`, `*.lockb`, `.DS_Store`. Keep `yarn.lock`, `Cargo.lock`, `go.sum` indexed
  (load-bearing for some semantic queries).
- Preserve `.gitignore` semantics — file-glob blocklist is additive, not replacement.

**Patterns to follow:**

- Existing `should_track` function in `src/core/walker.rs` is segment-based; the file-glob blocklist
  plugs in after it.

**Test scenarios:**

- Happy path: corpus from a 200k-file worktree drops to ~5k after default tuning (use a fixture
  worktree under `tests/fixtures/aggressive_walker/` with realistic structure).
- Edge case: `.yarn/cache/` filter ≠ `.yarn-state.yml` (the file is kept; the dir is dropped).
- Edge case: `*.map` is blocked but `srcmap.ts` is kept (extension match, not substring).
- Edge case: lock-file policy — `package-lock.json` blocked; `Cargo.lock` kept.
- Integration: walker invocation from server's discover path returns the smaller corpus end-to-end.

**Verification:**

- File count drop measured against a fixture; assertion within 10%.
- All existing walker tests still pass.

---

- U2. **Primary-worktree resolution + socket relocation**

**Goal:** Make the daemon socket live next to the primary repo's `.git/`, regardless of which
worktree the bridge was launched from.

**Requirements:** R1

**Dependencies:** None (parallel-able with U1).

**Files:**

- Create: `src/core/git_worktree.rs`
- Modify: `src/transport/paths.rs` (use new resolver in `daemon_dir`)
- Modify: `src/core/mod.rs` (add module)
- Test: `src/core/git_worktree.rs` (unit tests)
- Test: `tests/daemon_transport.rs` (add cross-worktree connect test)

**Approach:**

- New function `resolve_primary_worktree(root: &Path) -> PathBuf`:
  - If `root/.git` is a directory → `root` is primary; return `root`.
  - If `root/.git` is a file → parse `gitdir: <path>` line. The path points into
    `<primary>/.git/worktrees/<name>/`. Read the `commondir` file in that dir; resolve relative to
    gitdir; that's `<primary>/.git/`. Return its parent.
  - If parse fails or no `.git` at all → fall back to `root` with a `tracing::warn!`. Never bail.
- `transport::paths::daemon_dir`: when the override env var is unset, call
  `resolve_primary_worktree(root)` first, then join `.mcp_data`.
- Override env var keeps absolute precedence (existing behavior preserved).

**Patterns to follow:**

- Existing `transport::paths::daemon_dir` structure; same return shape.

**Test scenarios:**

- Happy path: `.git` as directory at root → returns root.
- Happy path: `.git` as file with valid pointer → returns primary repo root.
- Edge case: malformed `gitdir:` line → returns root, emits warn.
- Edge case: `gitdir:` points at a non-existent path → returns root, emits warn.
- Edge case: no `.git` anywhere up the chain → returns root.
- Edge case: relative `gitdir:` paths resolve correctly.
- Integration (in `tests/daemon_transport.rs`): bridge spawned from a secondary worktree connects to
  the same socket as a bridge spawned from primary.
- Edge case: `CONTEXTPLUS_DAEMON_SOCKET_PATH` override still wins over primary resolution.

**Verification:**

- Two bridges from two worktrees of the same repo open exactly one daemon (verified by `lsof` or
  pidfile inspection in test).

---

- U3. **SharedState refactor: split shared vs. per-ref (no behavior change)**

**Goal:** Carve `SharedState` into a top-level shared layer + a `RefIndex` per-ref layer, while
preserving exactly today's single-ref behavior. This is a mechanical refactor that lets U4 add
multi-ref dispatch without a churning diff.

**Requirements:** R2 (foundation only — actual multi-ref dispatch in U4)

**Dependencies:** U2 (so `RefIndex` knows its primary anchor)

**Files:**

- Modify: `src/server.rs` (split `SharedState`; introduce `RefIndex`; preserve all existing
  tool-handler call sites by keeping a `default_ref()` accessor)
- Modify: `src/transport/daemon.rs` (`run_if_owner` constructs SharedState with one `RefIndex` keyed
  at primary's `RefId`)
- Modify: `src/transport/dispatch.rs` (call sites use `default_ref()`)
- Test: `src/server.rs` (rebalance existing tests)

**Approach:**

- New struct `RefIndex` holds: `root_dir`, `canonical_root`, `embedding_cache`, `identifier_index`,
  `search_index_cache`, `memory_graph`, `tracker_handle`, `cache_generation`,
  `parent_ref_id: Option<RefId>`.
- `SharedState` keeps: `config`, `ollama`, `instructions_cache`, `idle_monitor`, `draining`,
  `inflight`, plus `refs: RwLock<HashMap<RefId, Arc<RwLock<RefIndex>>>>` and
  `default_ref_id: RefId`.
- All current tool handlers go through a thin `state.default_ref()` accessor for now. This is the
  seam U4 replaces.
- Behavior is identical to today; only the type layout changes.

**Execution note:** Make this a pure refactor commit. No new behavior, no changed test outcomes.
Reviewability hinges on the diff being mechanical.

**Patterns to follow:**

- Existing `Arc<SharedState>` ownership pattern; `RefIndex` follows the same.
- Existing `RwLock` discipline.

**Test scenarios:**

- All existing tests still pass (this is the primary correctness signal).
- New unit: `SharedState::default_ref()` returns the only entry from a freshly-constructed state.

**Verification:**

- `cargo test` outcomes identical to pre-commit.
- No new clippy warnings.

---

- U4. **`register_session` protocol + multi-ref fanout**

**Goal:** Bridges register a session with their `client_root` and `head_sha`; daemon resolves to a
`RefId`, attaches a new `RefIndex` if needed (lazily), and dispatches all subsequent tool calls in
that session through the chosen ref. Single-ref CONTEXTPLUS_TRANSPORT=stdio path remains unchanged.

**Requirements:** R2

**Dependencies:** U3 (`RefIndex`/`SharedState` split must exist).

**Files:**

- Modify: `src/transport/client.rs` (bridge sends `register_session` after socket connect, before
  forwarding stdio)
- Modify: `src/transport/daemon.rs` (per-connection task reads `register_session`, looks up or
  creates `RefIndex`, holds the ref alive for session lifetime)
- Modify: `src/transport/dispatch.rs` (dispatch uses `session.ref_id` rather than `default_ref()`)
- Modify: `src/server.rs` (add `attach_ref`, `detach_ref` lifecycle on `SharedState`)
- Test: `tests/daemon_transport.rs` (multi-ref scenarios)

**Approach:**

- Bridge handshake: framed JSON (length-prefixed) over the unix socket, single `register_session`
  frame before forwarding raw rmcp stdio.
- Daemon attach: if `ref_id` not present, mint one. Fork base =
  `merge-base HEAD <primary_tracked_branch>`; if a sibling ref already exists at that SHA, fork from
  sibling instead. Embedding of the diff happens off the request thread (status
  `Warming { eta_ms }`).
- Detach: on session disconnect, decrement refcount; when zero, ref enters TTL eviction queue
  (default 24h, configurable via `CONTEXTPLUS_REF_TTL_SECS`).
- TTL eviction = drop in-memory `RefIndex`; on-disk overlay is preserved unless the worktree's
  gitdir is gone.
- Drain integration: while `state.draining` is set, reject new `register_session` with
  `RejectedDraining`.

**Patterns to follow:**

- Existing rmcp serve-per-connection in `daemon.rs:accept_loop` — extend the per-connection task
  with the handshake step before handing the stream to rmcp.
- Existing `process_lifecycle::IdleMonitor` for the TTL pattern.

**Test scenarios:**

- Happy path: bridge from primary registers with `head_sha=X`; daemon attaches single ref; tool call
  returns expected result.
- Happy path: second bridge from worktree registers with `head_sha=Y`; daemon attaches second ref
  forked from primary; tool call returns expected result.
- Edge case: two bridges from the same worktree registering concurrently → one ref created, both
  share it (no race).
- Edge case: bridge registers with `head_sha` that doesn't exist in primary's git → daemon falls
  back to forking from primary (no-op base) and warns.
- Error path: `register_session` arrives during drain → bridge gets `RejectedDraining`, exits 0.
- Integration: TTL eviction — set `CONTEXTPLUS_REF_TTL_SECS=2`, disconnect bridge, verify ref is
  evicted from the registry within 5s.
- Integration: drain — issue SIGTERM mid-session; verify in-flight finishes, ref states evict
  cleanly, socket unlinks.

**Verification:**

- `tests/daemon_transport.rs` `multi_ref_attach_serves_diff_results` passes.
- TTL eviction observable via a daemon-internal stat (added if not present).

---

- U5. **Path translation at the dispatch boundary**

**Goal:** Tool inputs and outputs are translated into the calling session's worktree path-space at
the dispatch layer, so tool implementations stay path-agnostic. The "B can't see A's worktree path"
invariant is enforced here.

**Requirements:** R5

**Dependencies:** U4 (dispatch knows `session.ref_id` and the ref's worktree).

**Files:**

- Modify: `src/transport/dispatch.rs` (add input/output translation around tool invocation)
- Create: `src/core/path_translation.rs` (pure functions; unit-tested)
- Test: `src/transport/dispatch.rs` (cross-session leakage tests)
- Test: `src/core/path_translation.rs` (unit tests for the translation primitives)

**Approach:**

- Inputs: any caller-provided `path` argument matched by tool schema is rewritten — strip caller's
  worktree absolute prefix, store as canonical (repo-relative).
- Outputs: any path field returned by a tool is rewritten — prefix with the calling session's
  worktree absolute path. If a chunk was learned from another worktree's path, the canonical
  (repo-relative) form is the same; the caller-side prefix is what makes the result correct.
- Out-of-tree paths (caller passes a path not under their worktree) → reject with a clear error
  rather than silently rewriting.
- Translation primitives live in `path_translation.rs` for testability.

**Patterns to follow:**

- The "Caller-provided rootDir is outside the server root; ignoring" log path already exists in
  `src/server.rs:1439` — reuse that warning shape for out-of-tree path rejections.

**Test scenarios:**

- Happy path: bridge A calls `semantic_code_search`; output paths are inside A's worktree absolute
  prefix.
- Happy path: bridge B calls `semantic_code_search` against the same content; output paths are
  inside B's worktree absolute prefix (different from A's).
- **The leakage test**: A and B share a chunk; verify B's tool result NEVER contains A's worktree
  path string anywhere in the response JSON. Run with `serde_json::Value` recursive scan asserting
  absence.
- Edge case: tool input contains a path outside the caller's worktree → tool returns a clear error;
  no panic.
- Edge case: tool output contains a path that was learned from primary (`parent_ref`); caller is a
  worktree → path is rewritten to caller's worktree absolute prefix.
- Edge case: stdio mode (no daemon) → translation is a no-op; existing behavior preserved.

**Verification:**

- `dispatch.rs` patch coverage clears 80% (R11).
- `tests/cross_ref_no_path_leakage.rs` (new) green.

---

- U6. **Content-addressed CAS + per-ref CoW chunk index**

**Goal:** Chunk embeddings become content-addressed; per-ref manifests carry only
`(file_path, chunk_idx) → chunk_hash` deltas vs. parent. Forking a new worktree pays only the
diff-vs-base embedding cost.

**Requirements:** R3, R4

**Dependencies:** U3 (`RefIndex` exists), U4 (refs are addressable).

**Files:**

- Create: `src/cache/cas.rs` (content-addressed blob store)
- Modify: `src/cache/rkyv_store.rs` (manifest writes use chunk-hash references; existing
  per-worktree subdir paths become per-ref manifests)
- Modify: `src/core/embeddings.rs` (lookup chain through parent on miss; embed only on CAS miss)
- Modify: `src/core/embedding_tracker.rs` (file-change events resolve to chunk-hash diffs; manifest
  updates atomic)
- Modify: `Cargo.toml` (add `blake3` dep)
- Test: `src/cache/cas.rs` (unit tests for CAS layout)
- Test: `tests/fork_diff_only.rs` (new — measurement test)

**Approach:**

- CAS layout (under primary's `.mcp_data/`):
  ```
  .mcp_data/
    cas/<model_slug>/<ab>/<cdef…>.emb   // BLAKE3-keyed embeddings
    refs/<ref_id>/manifest.rkyv         // path → [(idx, chunk_hash)]
    refs/<ref_id>/parent                // file containing parent's ref_id, or empty
  ```
- Reads chain through parents until a manifest hit; on hit, fetch CAS blob.
- Writes: walker chunks file, computes BLAKE3 of each chunk; if blob exists, manifest update only;
  else, queue for embedding then write blob + manifest entry.
- Forking a new ref: clone `parent`'s manifest by reference (point `parent` file to parent's
  `ref_id`); manifest starts empty; on first walk, only changed files contribute manifest deltas.
- HNSW per ref: clone parent's index in memory + insert overlay vectors; rebuild from scratch only
  on watermark (e.g., overlay > 5% of parent's size).
- Old path-keyed `.rkyv` files: detected on daemon start by absence of `cas/` dir; deleted with a
  tracing::info! and rebuilt fresh.

**Patterns to follow:**

- Existing rkyv save/load pattern in `src/cache/rkyv_store.rs`.
- Existing `embed_cache_name` / `model_slug` conventions.

**Test scenarios:**

- Happy path: fresh daemon, primary indexes a fixture repo (~200 files) → CAS blobs exist, primary's
  manifest has all entries.
- Happy path: fork a worktree at the same SHA as primary → manifest is empty; tool calls succeed via
  parent fallback.
- Happy path: fork a worktree with 5 file diffs → only 5 files appear in worktree's manifest; CAS
  blob count grows by ≤(5 × max_chunks_per_file).
- **Measurement test (`fork_diff_only.rs`)**: instrument an embed-call counter on the Ollama mock;
  fork a worktree at base+10-file-diff; assert `embed_calls ≤ chunks_in_diffed_files`, never
  `chunks_in_full_repo`.
- Edge case: fork from a sibling ref at the same SHA → no embed calls beyond what the sibling
  already paid.
- Edge case: identical file content under different paths in the same ref → one CAS blob, two
  manifest entries.
- Error path: CAS blob missing on disk for a manifest entry → tool returns a clear error, ref is
  flagged for re-walk on next access.
- Edge case: drop-and-rebuild detection — start daemon with old-format `.rkyv` present and `cas/`
  absent → daemon deletes old files, logs INFO, rebuilds fresh.

**Verification:**

- `fork_diff_only.rs` measures embed-call count below threshold.
- Disk usage of N-worktree daemon ≤ (primary's chunk count × blob size) + (N × small manifest size),
  not N × full embedding cache.

---

- U7. **Memory-graph CoW + auto-merge + `needs_rebuild`**

**Goal:** Memory graph forks CoW from primary. On primary HEAD advance, refs whose HEAD becomes an
ancestor auto-merge their overlay into primary via the merge ladder. Nodes whose referenced chunks
change get `needs_rebuild` and are hidden from search until rebuilt.

**Requirements:** R6, R7, R8

**Dependencies:** U6 (chunk-hash references), U4 (ref lifecycle).

**Files:**

- Modify: `src/core/memory_graph.rs` (add CoW overlay; add `needs_rebuild`,
  `references_chunk_hashes` fields to `MemoryNode`)
- Create: `src/core/memory_merge.rs` (merge ladder + `needs_rebuild` rebuild logic)
- Create: `src/core/head_watcher.rs` (notify-rs subscription on primary's gitdir; HEAD-advance event
  source)
- Modify: `src/transport/daemon.rs` (start head watcher; on event, dispatch ladder)
- Modify: `src/server.rs` (search/traversal hide nodes flagged `needs_rebuild`; rebuild on direct
  access)
- Test: `src/core/memory_merge.rs` (unit tests for each ladder rung)
- Test: `tests/memory_overlay_auto_merge.rs` (integration — scripted git history)

**Approach:**

- `MemoryGraph` gains `parent: Option<Arc<MemoryGraph>>` and `overlay: HashMap<NodeId, MemoryNode>`
  semantics; reads chain to parent on miss; writes go to overlay.
- `MemoryNode` gains `references_chunk_hashes: Vec<ChunkHash>` (set when the node is created from
  code) and `needs_rebuild: bool` (default false).
- On chunk-hash change for any referenced chunk in U6's tracker → set `needs_rebuild=true` on
  dependent nodes (lazy walk on next access, or eager via a dedicated job in a follow-up).
- Search/traversal: filter out nodes where `needs_rebuild` is true; on direct access
  (`get_node(id)`), trigger rebuild — re-embed body, re-validate edge endpoints; if endpoint refers
  to a missing symbol, prune that edge; if rebuild fails entirely, prune the node.
- Auto-merge: head watcher emits `HeadAdvanced { old, new }` events for primary. For each active
  ref, `git merge-base --is-ancestor <ref_head> <new>` → matches enter the ladder. Per-node
  classification:
  - `lookup(node_id)` returns `None` in primary → publish to primary (clean — new node).
  - `lookup(node_id)` returns Equal content (hash-compared) → noop (identical).
  - `lookup(node_id)` returns Additive-compatible (only edges added in overlay, no body change) →
    smart-merge: union edges, max(access_count), latest last_accessed.
  - Otherwise → set `needs_rebuild=true` on primary's node; log INFO with both contents' hashes for
    diagnostics.
- Auto-merge runs in a serialized task per primary; respects drain.

**Patterns to follow:**

- Existing `prune_stale_links` is the reference for safe edge cleanup.
- Existing `embedding_tracker` event loop is the reference for the head watcher's structure.

**Test scenarios:**

- Happy path (CoW reads): ref A inherits primary's nodes via parent fallback; primary tool sees
  primary's view; A's tool sees primary + A's overlay.
- Happy path (CoW writes): A's `upsert_memory_node` creates a node only visible to A; primary's
  `search_memory_graph` does not return it.
- **Auto-merge clean**: A creates node N; A's branch is rebased onto primary; primary HEAD advances
  past A's HEAD; daemon merges N into primary; primary's `search_memory_graph` now returns N; A's
  overlay is empty post-merge.
- **Auto-merge identical**: A creates node N; primary independently creates same N; merge → noop, no
  duplicate.
- **Auto-merge smart**: A adds edge `N → M`; primary adds edge `N → P` (different target); merge →
  both edges present in primary.
- **Auto-merge conflict**: A edits N's body to "X"; primary edits N's body to "Y" since fork;
  ref-merge fires → primary's N flagged `needs_rebuild`; search hides N until next direct access;
  direct access re-embeds + validates.
- **`needs_rebuild` on chunk change**: node N references chunk hash H; H changes (file edited); N
  flagged `needs_rebuild`; search hides N; direct access triggers rebuild from new chunk.
- **`needs_rebuild` rebuild failure**: node N references chunk hash H pointing at a now-deleted
  symbol; rebuild fails → N pruned.
- Drain: HEAD advance during drain → merge deferred (no merge while draining); next daemon start
  picks up the work.

**Verification:**

- `tests/memory_overlay_auto_merge.rs` exercises clean / identical / smart / conflict cases via
  scripted git history (`git2` or shelling out — TBD in execution).
- Search results never include a `needs_rebuild` node, asserted in cross-cutting tests.

---

- U8. **CI cleanup: clippy + dispatch.rs coverage to ≥80%**

**Goal:** Clear the lint failures and patch-coverage gap that have kept #70 yellow.

**Requirements:** R11

**Dependencies:** U5 (which already brings dispatch.rs coverage up via path-translation tests), U7
(merge tests round out coverage).

**Files:**

- Modify: any files clippy flags
- Modify: `src/transport/dispatch.rs` (top-up tests if still under target)
- Modify: `tests/paths.rs` (fix env-var test isolation flagged in #70 description:
  `socket_override_redirects_paths` mutates `CONTEXTPLUS_DAEMON_SOCKET_PATH` without isolation
  across parallel tests)

**Approach:**

- Run `cargo clippy --all-targets -- -D warnings` and fix until clean.
- Run codecov locally (`cargo tarpaulin --packages contextplus-rs --line --out Stdout` or
  equivalent) to verify dispatch.rs is over 80%; add direct unit tests for edges if not.
- Wrap `socket_override_redirects_paths` in a `Mutex` (or use `serial_test`) to fix flakiness under
  parallel test threads.

**Patterns to follow:**

- Whatever serialization pattern is already used in dispatch.rs's existing env-mutation tests
  (visible from PR #70's diff).

**Test scenarios:**

- Test expectation: none — this unit's job is to bring CI green, not add new behavioral coverage.
  New tests added here are gap-fillers not feature tests.

**Verification:**

- CI green on the `feat/per-workspace-daemon` branch: clippy ✅, codecov patch ≥80%, all jobs ✅.

---

## System-Wide Impact

- **Interaction graph:** the daemon is now a coordinator for N refs. Every tool dispatch goes
  through `SharedState.refs[ref_id]`. The path-translation layer touches every tool. Drain protocol
  now also gates auto-merge.
- **Error propagation:** `RegisterSessionRejected`, `RefNotFound`, `RebuildFailed` are new error
  classes. All must serialize cleanly to MCP error responses; bridges must surface them so MCP
  clients (Claude Code, Codex) print useful diagnostics rather than `Transport closed`.
- **State lifecycle risks:** ref TTL eviction must not race with in-flight tool dispatches
  (refcount). Memory-graph auto-merge must atomically rename-publish; mid-merge crashes leave
  overlay intact (next daemon start re-runs). CAS blobs are append-only — no race.
- **API surface parity:** MCP tool surface is unchanged from the bridge perspective. Internally,
  every tool gains an implicit `ref_id` parameter via dispatch; tool implementations don't see it.
- **Integration coverage:** cross-session path-leakage and fork-diff-only embed-count are the two
  new high-value integration tests; they prove invariants that unit tests can't.
- **Unchanged invariants:** `CONTEXTPLUS_TRANSPORT=stdio` is unchanged. Existing single-workspace
  deployments see no behavior change beyond the aggressive walker (smaller corpus, faster
  everything).

---

## Risks & Dependencies

| Risk                                                                                                                                         | Mitigation                                                                                                                                                                |
| -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SharedState` refactor (U3) breaks an obscure tool's call site that read `state.root_dir` directly                                           | U3 is a pure refactor commit; `state.default_ref().root_dir` accessor preserves the read; reviewers can diff field-by-field.                                              |
| Multi-ref dispatch (U4) introduces a deadlock when a tool holds a `RefIndex` read lock and triggers a memory-merge that wants the write lock | Ladder runs in a serialized task that takes the write lock only between tool calls; reads use try_read with backoff. Documented + tested.                                 |
| Path translation (U5) misses an output path field on a less-traveled tool                                                                    | Test asserts no caller-foreign worktree string anywhere in serialized JSON output for every tool. Recursive `serde_json::Value` scan in test framework.                   |
| CAS rewrite (U6) corrupts on-disk format mid-migration if a daemon crashes during drop-and-rebuild                                           | Drop-and-rebuild marker file (`migration_in_progress`) makes the operation idempotent; on next start, presence triggers cleanup-and-retry.                                |
| Memory auto-merge (U7) loses an agent's notes via mis-classified conflict                                                                    | Conflict path flags `needs_rebuild` rather than dropping; the agent's note becomes recoverable via direct access. Logged at INFO with content hashes for postmortem.      |
| HEAD watcher (U7) misses an event during daemon-internal pause (drain or merge in flight)                                                    | On any wakeup, watcher reconciles current HEAD vs. last-known; missed events are caught up rather than lost.                                                              |
| Aggressive walker (U1) drops a file users actually wanted indexed                                                                            | Defaults are conservative-additive (large dirs, vendored deps); user can override via existing `CONTEXTPLUS_IGNORE_DIRS` env var. Document the new defaults in CHANGELOG. |
| inotify watch budget exhausted by N watchers across N worktrees                                                                              | Single `notify-rs` watcher on primary's path; per-ref subscriptions are in-process subscribers, not OS-level inotify watches.                                             |

---

## Phased Delivery

### Phase 1 — Walker + Path Resolution (low-risk foundation)

- U1 Aggressive walker defaults
- U2 Primary-worktree resolution + socket relocation

These two are independent of each other and of PR #70's existing transport code. Land first;
user-visible benefit (faster indexing, shared socket across worktrees) starts here.

### Phase 2 — Multi-ref plumbing (refactor + dispatch)

- U3 SharedState refactor (no behavior change)
- U4 register_session protocol + multi-ref fanout
- U5 Path translation at the dispatch boundary

Each commit reviewable on its own; U3 is the mechanical refactor everyone can trust.

### Phase 3 — Forking the cache (the core feature)

- U6 Content-addressed CAS + per-ref CoW chunk index
- U7 Memory-graph CoW + auto-merge + needs_rebuild

The two big behavior commits. Land them in order; U7 depends on U6's chunk-hash references for
`needs_rebuild` semantics.

### Phase 4 — CI cleanup

- U8 clippy + coverage

Final sweep before requesting merge of #70.

---

## Documentation / Operational Notes

- Update PR #70 description with the multi-worktree story; the original text only addresses
  single-workspace concurrency.
- Add a `MIGRATION.md` snippet: "first daemon start under v0.2 deletes old path-keyed `.rkyv` caches
  and rebuilds; expect 30–120s warmup on first call per repo."
- Update README's MCP-config section to document `CONTEXTPLUS_REF_TTL_SECS` and the new
  socket-resolves-to-primary behavior. Note that override env var still wins.
- CHANGELOG entry: list each new env var, the new ignore defaults, and the auto-merge trigger.
- Operational note: with all worktrees sharing one daemon, killing the daemon kills MCP for every
  active session. Drain protocol (PR #69) is the recommended path; document
  `kill -SIGTERM <daemon_pid>` for ops.

---

## Sources & References

- PR #70: https://github.com/mrsufgi/contextplus-rs/pull/70 — the foundation this plan extends.
- PR #69 (merged): graceful drain on parent process death — referenced by U4 and U7 for shutdown
  semantics.
- PR #50 (merged): "fix(cache): sweep stale worktree entries from rkyv cache on load" — predecessor
  of the CoW work; U6 and U7 supersede the sweep approach.
- `src/cache/rkyv_store.rs:67` — existing per-worktree subdir code
  (`worktree_name`/`WORKTREE_SUBDIR`); foundation U6 builds on.
- `src/server.rs:85-128` — `SharedState` definition; U3 splits this.
- `src/transport/paths.rs:34` — `daemon_dir` function; U2 modifies.
- `src/transport/dispatch.rs` — under-tested today (53% patch coverage); U5 + U8 fix.
- ripgrep `ignore` crate docs: https://docs.rs/ignore/0.4 — already a dep; U1 expands defaults.
- Conversation transcript: prior turns established the diagnostic basis (199k → 5k file count, 4.9
  GB worktree, 26 active worktrees, transport-closed root cause).
