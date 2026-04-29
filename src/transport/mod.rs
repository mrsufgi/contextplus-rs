//! Transport modes for the contextplus MCP server.
//!
//! There are three transports:
//!
//! 1. **stdio** — original behavior. Each Claude Code (or other MCP-host)
//!    session spawns its own contextplus process and the two communicate over
//!    `stdin`/`stdout`. Kept for backward compatibility.
//! 2. **daemon** — a single contextplus process per workspace. It binds a
//!    Unix-domain socket at `<root>/.mcp_data/contextplus.sock` and serves any
//!    number of concurrent MCP sessions, all sharing a single
//!    [`crate::server::SharedState`]. The daemon outlives any one client, so
//!    embedding caches, search indices, and tracker state survive
//!    Claude Code reload-mid-call.
//! 3. **client (proxy)** — a tiny stdin↔socket bridge. The MCP host still
//!    spawns one contextplus process per session, but that process holds no
//!    contextplus state — it just shovels JSON-RPC frames between its own
//!    stdio and a connected daemon. If no daemon is running it spawns one
//!    (forking itself with `--daemon`) and waits for the socket to appear.
//!
//! See `paths.rs` for the on-disk layout (lock + pid + socket).

pub mod client;
pub mod daemon;
pub mod paths;
