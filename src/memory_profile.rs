use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use serde_json::{Map, Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::config::{Config, RefWarmupMode, TrackerMode};
use crate::ref_index::{RefId, RefIndex};
use crate::server::ContextPlusServer;

/// Corpus shape of one profile run.
#[derive(Clone, Copy)]
pub struct ProfileSize {
    pub files: usize,
    pub identifiers_per_file: usize,
    pub dims: usize,
}

impl ProfileSize {
    /// About 8.5k files and 102k identifiers at 768 dimensions.
    pub const BERRIES: Self = Self {
        files: 8_500,
        identifiers_per_file: 12,
        dims: 768,
    };
}

/// What one ref answered and which heavy structures it shares with the primary.
pub struct ProfileRef {
    pub name: String,
    pub results: Vec<String>,
    pub marker_result: String,
    pub shares_identifier_base: bool,
    pub shares_project_cache: bool,
    pub shares_lexical_index: bool,
}

struct ProfileSession {
    name: String,
    owner: Arc<RefIndex>,
    server: ContextPlusServer,
}

fn rss_bytes() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                line.strip_prefix("VmRSS:")?
                    .split_whitespace()
                    .next()?
                    .parse::<usize>()
                    .ok()
            })
        })
        .unwrap_or(0)
        .saturating_mul(1024)
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn report_rss(step: &str) {
    let bytes = rss_bytes();
    println!("rss step={step} bytes={bytes} mib={:.1}", mib(bytes));
}

fn write_corpus(root: &Path, size: ProfileSize) {
    for i in 0..size.files {
        let dir = root.join(format!("src/module_{:03}", i % 128));
        std::fs::create_dir_all(&dir).unwrap();
        let content: String = (0..size.identifiers_per_file)
            .map(|j| format!("pub fn synthetic_function_{i}_{j}() -> usize {{ {i} }}\n"))
            .collect();
        std::fs::write(dir.join(format!("file_{i}.rs")), content).unwrap();
    }
}

fn write_worktree_changes(root: &Path, index: usize) {
    std::fs::write(
        root.join(format!("src/worktree_{index}_change.rs")),
        format!("pub fn worktree_marker_{index}() {{}}\n"),
    )
    .unwrap();
    std::fs::write(
        root.join("src/module_000/file_0.rs"),
        format!("pub fn synthetic_function_0_0() -> usize {{ 0 }}\npub fn worktree_edit_{index}() {{}}\n"),
    )
    .unwrap();
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> Option<Value> {
    let mut buffer = Vec::new();
    let mut chunk = vec![0_u8; 64 * 1024];
    loop {
        let read = socket.read(&mut chunk).await.ok()?;
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
        let Some(end) = buffer.windows(4).position(|window| window == b"\r\n\r\n") else {
            continue;
        };
        let headers = String::from_utf8_lossy(&buffer[..end]).to_ascii_lowercase();
        let length = headers
            .lines()
            .find_map(|line| line.strip_prefix("content-length:"))
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(0);
        while buffer.len() < end + 4 + length {
            let read = socket.read(&mut chunk).await.ok()?;
            if read == 0 {
                return None;
            }
            buffer.extend_from_slice(&chunk[..read]);
        }
        return serde_json::from_slice(&buffer[end + 4..end + 4 + length]).ok();
    }
}

async fn mock_embedding_server(dims: usize) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            tokio::spawn(async move {
                let count = read_request(&mut socket)
                    .await
                    .and_then(|body| body["input"].as_array().map(Vec::len))
                    .unwrap_or(1);
                let body =
                    serde_json::to_vec(&json!({ "embeddings": vec![vec![0.5_f32; dims]; count] }))
                        .unwrap();
                let header = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                let _ = socket.write_all(header.as_bytes()).await;
                let _ = socket.write_all(&body).await;
            });
        }
    });
    (format!("http://{address}"), task)
}

fn profile_root() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("contextplus-memory-profile-{nonce}"))
}

fn explore_args(kind: &str, matching: &str, query: &str) -> Map<String, Value> {
    Map::from_iter([
        ("kind".into(), json!(kind)),
        ("match".into(), json!(matching)),
        ("query".into(), json!(query)),
        ("top_k".into(), json!(10)),
    ])
}

async fn explore(session: &ProfileSession, kind: &str, matching: &str, query: &str) -> String {
    let result = session
        .server
        .dispatch("explore", explore_args(kind, matching, query))
        .await;
    format!("{result:?}")
}

async fn run_queries(session: &ProfileSession) -> Vec<String> {
    let mut results = Vec::new();
    for (kind, matching) in [
        ("files", "meaning"),
        ("files", "keywords"),
        ("identifiers", "meaning"),
    ] {
        results.push(explore(session, kind, matching, "synthetic function 42").await);
    }
    println!(
        "queries ref={} meaning={} keywords={} identifiers={}",
        session.name,
        results[0].len(),
        results[1].len(),
        results[2].len()
    );
    results
}

async fn component_report(session: &ProfileSession) {
    let owner = &session.owner;
    let map_bytes =
        |cache: &std::collections::HashMap<String, crate::core::embeddings::CacheEntry>| {
            cache
                .values()
                .map(|entry| entry.hash.capacity() + entry.vector.capacity() * size_of::<f32>())
                .sum::<usize>()
        };
    let identifier_base = match owner.identifier_vectors.get() {
        Some(base) => map_bytes(&*base.read().await),
        None => 0,
    };
    let identifier_overlay = map_bytes(&*owner.identifier_vector_overlay.read().await);
    let identifier_index_vectors = owner
        .identifier_index
        .read()
        .await
        .as_ref()
        .map_or(0, |index| index.vector_buffer.len() * size_of::<f32>());
    let search = owner.search_index_cache.read().await.clone();
    let project_bytes = owner
        .project_cache
        .read()
        .await
        .as_ref()
        .map_or(0, |cache| {
            cache
                .file_content
                .iter()
                .map(|(path, content)| path.capacity() + content.capacity())
                .sum::<usize>()
        });
    let lexical = owner
        .lexical_search_cache
        .read()
        .await
        .as_ref()
        .map_or(0, |entry| entry.index.estimated_resident_bytes());
    println!(
        "components ref={} file_vectors={} identifier_base={} identifier_overlay={} identifier_index_vectors={} hnsw={} lexical={} project={} search_documents={}",
        session.name,
        search
            .as_ref()
            .map_or(0, |search| search.resident_file_vector_bytes()),
        identifier_base,
        identifier_overlay,
        identifier_index_vectors,
        search
            .as_ref()
            .map_or(0, |search| search.estimated_hnsw_bytes()),
        lexical,
        project_bytes,
        search
            .as_ref()
            .map_or(0, |search| search.resident_document_bytes()),
    );
}

async fn apply_delta(session: &ProfileSession, cycle: usize) {
    let before = session
        .owner
        .cache_generation
        .load(std::sync::atomic::Ordering::Acquire);
    std::fs::write(
        session.owner.root_dir.join("src/module_000/file_1.rs"),
        format!("pub fn identifier_delta_{cycle}() {{}}\n"),
    )
    .unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        while session
            .owner
            .cache_generation
            .load(std::sync::atomic::Ordering::Acquire)
            == before
        {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("the tracker did not pick up the delta of cycle {cycle}"));
    explore(session, "identifiers", "meaning", "synthetic function 42").await;
}

async fn shares<T>(
    owner: &RefIndex,
    primary: &RefIndex,
    slot: impl Fn(&RefIndex) -> &tokio::sync::RwLock<Option<Arc<T>>>,
) -> bool {
    match (
        slot(owner).read().await.as_ref(),
        slot(primary).read().await.as_ref(),
    ) {
        (Some(owner), Some(primary)) => Arc::ptr_eq(owner, primary),
        _ => false,
    }
}

/// Builds the primary and each worktree from its own directory, attaches the
/// worktrees through the `attach_worktree` tool and queries every ref through
/// `explore`, reporting RSS after each step.
pub async fn profile(size: ProfileSize, linked_refs: usize, cycles: usize) -> Vec<ProfileRef> {
    println!(
        "profile files={} identifiers={} dims={} linked_refs={linked_refs} cycles={cycles}",
        size.files,
        size.files * size.identifiers_per_file,
        size.dims
    );
    report_rss("start");

    let root = profile_root();
    let primary_root = root.join("primary");
    write_corpus(&primary_root, size);
    let worktree_roots: Vec<_> = (0..linked_refs)
        .map(|i| {
            let path = root.join(format!("worktree-{i}"));
            write_corpus(&path, size);
            write_worktree_changes(&path, i);
            path
        })
        .collect();
    report_rss("corpus-written");

    let (ollama_host, mock_server) = mock_embedding_server(size.dims).await;
    let mut config = Config::from_env();
    config.ollama_host = ollama_host;
    config.embed_tracker_mode = TrackerMode::Lazy;
    config.ref_warmup_mode = RefWarmupMode::Off;
    config.resident_memory_budget_bytes = usize::MAX;
    let server = ContextPlusServer::new(primary_root.canonicalize().unwrap(), config);
    let primary = ProfileSession {
        name: "primary".into(),
        owner: server.state.default_ref().unwrap(),
        server: server.clone(),
    };
    server.ensure_tracker_started().await;
    let mut sessions = vec![primary];
    let primary = &sessions[0];
    let results = run_queries(primary).await;
    let marker_result = explore(primary, "files", "keywords", "worktree marker").await;
    report_rss("primary");
    component_report(primary).await;
    let mut report = vec![ProfileRef {
        name: primary.name.clone(),
        results,
        marker_result,
        shares_identifier_base: true,
        shares_project_cache: true,
        shares_lexical_index: true,
    }];

    for (i, path) in worktree_roots.iter().enumerate() {
        let canonical = path.canonicalize().unwrap();
        server
            .dispatch(
                "attach_worktree",
                Map::from_iter([("path".into(), json!(canonical.to_string_lossy()))]),
            )
            .await;
        let ref_id = RefId::for_canonical_path(&canonical);
        let session = server.with_session(ref_id);
        session.ensure_tracker_started().await;
        let session = ProfileSession {
            name: format!("worktree-{i}"),
            owner: session.current_ref().await,
            server: session,
        };
        let results = run_queries(&session).await;
        let marker_result = explore(&session, "files", "keywords", "worktree marker").await;
        report_rss(&format!("attach-{}", i + 1));
        component_report(&session).await;
        let primary = &sessions[0].owner;
        report.push(ProfileRef {
            name: session.name.clone(),
            results,
            marker_result,
            shares_identifier_base: match (
                session.owner.identifier_vectors.get(),
                primary.identifier_vectors.get(),
            ) {
                (Some(owner), Some(primary)) => Arc::ptr_eq(owner, primary),
                _ => false,
            },
            shares_project_cache: shares(&session.owner, primary, |r| &*r.project_cache).await,
            shares_lexical_index: shares(&session.owner, primary, |r| &*r.lexical_search_cache)
                .await,
        });
        sessions.push(session);
    }

    let delta_session = sessions.get(1).unwrap_or(&sessions[0]);
    let rss_before_cycles = rss_bytes();
    for cycle in 1..=cycles {
        apply_delta(delta_session, cycle).await;
        if cycle == 1 || cycle == cycles {
            report_rss(&format!("delta-{cycle}"));
        }
    }
    let rss_after_cycles = rss_bytes();
    println!(
        "stability cycles={} before_bytes={} after_bytes={} growth_bytes={}",
        cycles,
        rss_before_cycles,
        rss_after_cycles,
        rss_after_cycles.saturating_sub(rss_before_cycles),
    );
    println!(
        "profile_complete rss_bytes={} rss_mib={:.1}",
        rss_bytes(),
        mib(rss_bytes())
    );
    drop(sessions);
    drop(server);
    mock_server.abort();
    let _ = std::fs::remove_dir_all(root);
    report
}

pub async fn run_memory_profile(linked_refs: usize, cycles: usize) {
    profile(ProfileSize::BERRIES, linked_refs, cycles).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lane_m_profile_measures_worktrees_built_from_their_own_trees() {
        let size = ProfileSize {
            files: 24,
            identifiers_per_file: 2,
            dims: 8,
        };

        let report = profile(size, 2, 0).await;

        assert_eq!(report.len(), 3);
        assert!(
            !report[0].marker_result.contains("worktree_"),
            "the primary answered with a worktree's files"
        );
        for (i, worktree) in report[1..].iter().enumerate() {
            assert!(
                worktree
                    .marker_result
                    .contains(&format!("worktree_{i}_change.rs")),
                "{} did not answer from its own tree: {}",
                worktree.name,
                worktree.marker_result
            );
            assert!(
                !worktree
                    .marker_result
                    .contains(&format!("worktree_{}_change.rs", 1 - i)),
                "{} answered with another worktree's file",
                worktree.name
            );
            assert!(
                !worktree.shares_project_cache && !worktree.shares_lexical_index,
                "{} was served the primary's project or lexical cache",
                worktree.name
            );
            assert!(
                worktree.shares_identifier_base,
                "{} did not share the primary's identifier vectors",
                worktree.name
            );
            assert!(worktree.results.iter().all(|result| !result.is_empty()));
        }
    }
}
