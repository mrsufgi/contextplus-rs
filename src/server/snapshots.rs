//! Snapshots of the primary checkout's built indexes (see
//! [`crate::cache::snapshot`]). On start the keyword index and the identifier
//! documents are read from their snapshots and only the files whose content
//! differs from the snapshot's manifest are indexed again, through the same
//! incremental paths a file change takes; the file documents of semantic
//! search reuse the parsed fields of every unchanged file. Anything that does
//! not match falls back to a full build.

use super::*;
use crate::cache::snapshot::{self, Digest, Snapshot, SnapshotWriter};
use crate::tools::semantic_identifiers::IdentifierDoc;
use rayon::prelude::*;
use std::path::Path;

pub(super) const KEYWORDS: &str = "keyword-index";
pub(super) const IDENTIFIERS: &str = "identifier-documents";
pub(super) const FILES: &str = "file-documents";

/// Version of what the builders of each kind derive from a file. Bumped,
/// with a new digest in the golden test below, whenever a builder's output
/// changes, including through a dependency such as a tree-sitter grammar.
const KEYWORDS_DERIVATION: u32 = 1;
const IDENTIFIERS_DERIVATION: u32 = 1;
const FILES_DERIVATION: u32 = 1;

fn derivation_version(kind: &str) -> u32 {
    match kind {
        KEYWORDS => KEYWORDS_DERIVATION,
        IDENTIFIERS => IDENTIFIERS_DERIVATION,
        _ => FILES_DERIVATION,
    }
}

fn fingerprint(kind: &str, config: &Config) -> String {
    // Keyword documents depend on file contents alone; the others belong to
    // an embedding space. Every kind holds symbols of the grammar set, which
    // the golden corpus cannot cover for a language it has no parser for yet.
    let identity = if kind == KEYWORDS {
        String::new()
    } else {
        config.document_cache_identity()
    };
    snapshot::fingerprint(
        kind,
        &format!(
            "d{}|g{}|{identity}",
            derivation_version(kind),
            crate::core::tree_sitter::grammar_identity()
        ),
    )
}

/// Whether `ref_index` persists snapshots: only a primary checkout does; a
/// linked worktree layers over its primary instead.
pub(super) fn enabled(config: &Config, ref_index: &crate::ref_index::RefIndex) -> bool {
    config.snapshots && ref_index.parent_ref_id.is_none()
}

/// Indexed text of `path`: its content, or empty when it could not be read.
fn indexed_text<'a>(files: &'a crate::core::walker::FileContents, path: &str) -> &'a str {
    files.get(path).map_or("", |content| content.as_str())
}

/// Digest of every indexed file of `cache`.
fn file_digests(cache: &ProjectCache) -> HashMap<&str, Digest> {
    let paths: Vec<&str> = cache
        .file_entries
        .iter()
        .filter(|entry| !entry.is_directory)
        .map(|entry| entry.relative_path.as_str())
        .collect();
    STRUCTURAL_POOL.install(|| {
        paths
            .par_iter()
            .map(|&path| {
                (
                    path,
                    snapshot::digest(indexed_text(&cache.file_content, path).as_bytes()),
                )
            })
            .collect()
    })
}

// ---------------------------------------------------------------------------
// Keyword index
// ---------------------------------------------------------------------------

/// Share of deleted slots or unused terms past which a keyword snapshot is
/// written from a compacted copy of the index.
const COMPACT_DEAD_FRACTION: f64 = 0.2;

/// Persists a flat keyword index with the digest of each document's file.
/// An index whose deleted slots or unused terms pass
/// [`COMPACT_DEAD_FRACTION`] is written compacted, so they do not
/// accumulate across restarts.
pub(super) fn write_keywords(
    root: &Path,
    config: &Config,
    cached: &CachedLexicalIndex,
) -> std::io::Result<bool> {
    if cached.base.is_some() {
        return Ok(false);
    }
    let live: Vec<bool> = cached
        .document_paths
        .iter()
        .map(|path| !path.is_empty())
        .collect();
    let dead_slots = live.iter().filter(|live| !**live).count();
    let dead_terms = cached.index.dead_term_count();
    let compact = dead_slots as f64 > live.len() as f64 * COMPACT_DEAD_FRACTION
        || dead_terms as f64 > cached.index.term_count() as f64 * COMPACT_DEAD_FRACTION;
    let compacted;
    let (index, document_paths): (_, Vec<&str>) = if compact {
        tracing::info!(dead_slots, dead_terms, "compacting the keyword snapshot");
        compacted = cached.index.compacted(&live);
        (
            &compacted,
            cached
                .document_paths
                .iter()
                .filter(|path| !path.is_empty())
                .map(String::as_str)
                .collect(),
        )
    } else {
        (
            &cached.index,
            cached.document_paths.iter().map(String::as_str).collect(),
        )
    };
    let files = &cached.project_cache.file_content;
    let digests: Vec<Digest> = STRUCTURAL_POOL.install(|| {
        document_paths
            .par_iter()
            .map(|path| match path.is_empty() {
                true => [0; 16],
                false => snapshot::digest(indexed_text(files, path).as_bytes()),
            })
            .collect()
    });
    let mut out = SnapshotWriter::create(
        &snapshot::snapshot_path(root, KEYWORDS),
        &fingerprint(KEYWORDS, config),
    )?;
    out.strs(document_paths.iter().copied())?;
    for digest in &digests {
        out.digest(digest)?;
    }
    index.write_snapshot(&mut out)?;
    out.commit()?;
    Ok(true)
}

/// The keyword index in the snapshot, its document paths and the digest of
/// each document's file.
pub(super) fn read_keywords(
    root: &Path,
    config: &Config,
) -> Option<(
    crate::tools::lexical_search::LexicalIndex,
    Vec<String>,
    Vec<Digest>,
)> {
    let snapshot = Snapshot::open(
        &snapshot::snapshot_path(root, KEYWORDS),
        &fingerprint(KEYWORDS, config),
    )?;
    let mut input = snapshot.reader();
    let document_paths = input.strings()?;
    let digests = (0..document_paths.len())
        .map(|_| input.digest())
        .collect::<Option<Vec<_>>>()?;
    let index =
        crate::tools::lexical_search::LexicalIndex::read_snapshot(&mut input, &document_paths)?;
    if !input.is_empty() || index.slot_count() != document_paths.len() {
        return None;
    }
    #[cfg(test)]
    test_seams::record_load(root, KEYWORDS);
    Some((index, document_paths, digests))
}

/// The keyword index of `project_cache` from the snapshot, with the files
/// changed, added or deleted since applied as an update. `None` without a
/// usable snapshot or when too much changed for an update. The count tells
/// how many files changed.
pub(super) fn load_keywords(
    root: &Path,
    config: &Config,
    project_cache: &Arc<ProjectCache>,
    generation: u64,
) -> Option<(CachedLexicalIndex, usize)> {
    let started = Instant::now();
    let (mut index, mut document_paths, digests) = read_keywords(root, config)?;
    let load_ms = started.elapsed().as_millis();

    let current = file_digests(project_cache);
    let mut known: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut changed: Vec<String> = Vec::new();
    let mut deleted: Vec<usize> = Vec::new();
    for (i, path) in document_paths.iter().enumerate() {
        if path.is_empty() {
            continue;
        }
        known.insert(path);
        match current.get(path.as_str()) {
            Some(digest) if *digest == digests[i] => {}
            Some(_) => changed.push(path.clone()),
            None => deleted.push(i),
        }
    }
    changed.extend(
        project_cache
            .file_entries
            .iter()
            .filter(|entry| !entry.is_directory && !known.contains(entry.relative_path.as_str()))
            .map(|entry| entry.relative_path.clone()),
    );
    drop(known);
    let delta = changed.len() + deleted.len();
    if delta as f64
        > index.document_count() as f64
            * crate::tools::semantic_search::FULL_REBUILD_CHANGE_FRACTION
    {
        tracing::info!(delta, "keyword snapshot too far behind; building in full");
        return None;
    }
    if delta > 0 {
        let files = &project_cache.file_content;
        let updates = lexical_updates(
            &mut document_paths,
            changed
                .iter()
                .map(|path| (path.as_str(), indexed_text(files, path))),
        );
        for &i in &deleted {
            document_paths[i].clear();
        }
        index.update_documents(updates, &deleted);
    }
    tracing::info!(
        phase = "keyword_snapshot_load",
        load_ms,
        reconcile_ms = started.elapsed().as_millis() - load_ms,
        changed = changed.len(),
        deleted = deleted.len(),
        documents = index.document_count(),
        "cold-start phase"
    );
    Some((
        CachedLexicalIndex {
            index,
            document_paths,
            project_cache: Arc::clone(project_cache),
            generation,
            base: None,
        },
        delta,
    ))
}

/// Writes the keyword snapshot of `ref_index` in the background from
/// whatever keyword index it holds then.
pub(super) fn schedule_keywords(
    config: &Config,
    ref_index: &Arc<crate::ref_index::RefIndex>,
    change: Change,
) {
    if enabled(config, ref_index) {
        change.request(
            &ref_index.snapshots.keywords,
            KEYWORDS,
            keywords_writer(config, ref_index),
        );
    }
}

fn keywords_writer(
    config: &Config,
    ref_index: &crate::ref_index::RefIndex,
) -> impl FnOnce() -> std::io::Result<bool> + Send + 'static {
    let root = ref_index.root_dir.clone();
    let config = config.clone();
    let slot = Arc::downgrade(&ref_index.lexical_search_cache);
    move || {
        let Some(cached) = slot
            .upgrade()
            .and_then(|slot| slot.blocking_read().as_ref().cloned())
        else {
            return Ok(false);
        };
        write_keywords(&root, &config, &cached)
    }
}

/// What a snapshot write follows.
pub(crate) enum Change {
    /// A full build: written soon.
    Full,
    /// An update of `changed` of `documents` files: written once enough has
    /// drifted and the index is idle, or at shutdown.
    Files { changed: usize, documents: usize },
}

impl Change {
    fn request(
        self,
        schedule: &Arc<snapshot::WriteSchedule>,
        kind: &'static str,
        write: impl FnOnce() -> std::io::Result<bool> + Send + 'static,
    ) {
        match self {
            Change::Full => schedule.schedule(kind, write),
            Change::Files { changed, documents } => {
                schedule.schedule_drift(kind, changed, documents, write)
            }
        }
    }
}

/// Writes every snapshot of `ref_index` with unwritten changes, on this
/// thread. For a graceful shutdown.
pub(crate) fn flush(config: &Config, ref_index: &Arc<crate::ref_index::RefIndex>) {
    if !enabled(config, ref_index) {
        return;
    }
    let snapshots = &ref_index.snapshots;
    let _ = snapshots
        .keywords
        .flush(KEYWORDS, keywords_writer(config, ref_index));
    let _ = snapshots
        .identifiers
        .flush(IDENTIFIERS, identifiers_writer(config, ref_index));
    let _ = snapshots
        .files
        .flush(FILES, files_writer(config, ref_index));
}

// ---------------------------------------------------------------------------
// Identifier documents
// ---------------------------------------------------------------------------

/// Persists the identifier documents of every file of `cache`, the project
/// cache `index` was built from.
pub(super) fn write_identifiers(
    root: &Path,
    config: &Config,
    index: &IdentifierIndex,
    cache: &ProjectCache,
) -> std::io::Result<bool> {
    let digests = file_digests(cache);
    let mut paths: Vec<&str> = digests.keys().copied().collect();
    paths.sort_unstable();
    let mut out = SnapshotWriter::create(
        &snapshot::snapshot_path(root, IDENTIFIERS),
        &fingerprint(IDENTIFIERS, config),
    )?;
    out.usize(paths.len())?;
    for path in paths {
        out.str(path)?;
        out.digest(&digests[path])?;
        let docs = index
            .docs
            .files
            .get(path)
            .map_or(&[][..], |docs| docs.as_slice());
        out.str(docs.first().map_or("", |doc| doc.header.as_str()))?;
        out.usize(docs.len())?;
        for doc in docs {
            out.str(&doc.name)?;
            out.str(&doc.kind)?;
            out.usize(doc.line)?;
            out.usize(doc.end_line)?;
            out.str(&doc.signature)?;
            out.opt_str(doc.parent_name.as_deref())?;
            for tokens in [
                &doc.name_token_set,
                &doc.signature_token_set,
                &doc.parent_token_set,
            ] {
                let mut tokens: Vec<&str> = tokens.iter().map(String::as_str).collect();
                tokens.sort_unstable();
                out.strs(tokens.into_iter())?;
            }
        }
    }
    out.commit()?;
    Ok(true)
}

/// Identifier documents of the files of `cache` whose content matches the
/// snapshot, and the files to parse: those changed or added since.
pub(super) fn load_identifiers(
    root: &Path,
    config: &Config,
    cache: &ProjectCache,
) -> Option<(Vec<IdentifierDoc>, std::collections::HashSet<String>)> {
    let started = Instant::now();
    let current = file_digests(cache);
    let loaded = read_identifiers(root, config, &current)?;
    tracing::info!(
        phase = "identifier_snapshot_load",
        elapsed_ms = started.elapsed().as_millis(),
        reused = loaded.0.len(),
        to_parse = loaded.1.len(),
        "cold-start phase"
    );
    Some(loaded)
}

/// The identifier documents in the snapshot of the files whose digest in
/// `current` matches, and the files of `current` to parse.
pub(super) fn read_identifiers(
    root: &Path,
    config: &Config,
    current: &HashMap<&str, Digest>,
) -> Option<(Vec<IdentifierDoc>, std::collections::HashSet<String>)> {
    let snapshot = Snapshot::open(
        &snapshot::snapshot_path(root, IDENTIFIERS),
        &fingerprint(IDENTIFIERS, config),
    )?;
    let mut input = snapshot.reader();
    let files = input.usize()?;
    let mut reused = Vec::new();
    let mut seen = std::collections::HashSet::with_capacity(files);
    let read_tokens = |input: &mut snapshot::SnapshotReader<'_>| {
        let len = input.usize()?;
        let mut tokens = std::collections::HashSet::with_capacity(len);
        for _ in 0..len {
            tokens.insert(input.str()?.to_owned());
        }
        Some(tokens)
    };
    for _ in 0..files {
        let path = input.str()?;
        let digest = input.digest()?;
        let header = input.str()?;
        let docs = input.usize()?;
        let keep = current.get(path) == Some(&digest);
        if keep {
            seen.insert(path);
        }
        for _ in 0..docs {
            let name = input.str()?;
            let kind = input.str()?;
            let line = input.usize_value()?;
            let end_line = input.usize_value()?;
            let signature = input.str()?;
            let parent_name = input.opt_str()?;
            let name_tokens = read_tokens(&mut input)?;
            let signature_tokens = read_tokens(&mut input)?;
            let parent_tokens = read_tokens(&mut input)?;
            if keep {
                reused.push(IdentifierDoc::assemble(
                    path,
                    header,
                    name.to_owned(),
                    kind.to_owned(),
                    line,
                    end_line,
                    signature.to_owned(),
                    parent_name.map(str::to_owned),
                    name_tokens,
                    signature_tokens,
                    parent_tokens,
                ));
            }
        }
    }
    if !input.is_empty() {
        return None;
    }
    reused.shrink_to_fit();
    let parse: std::collections::HashSet<String> = current
        .keys()
        .filter(|path| !seen.contains(*path))
        .map(|path| (*path).to_owned())
        .collect();
    #[cfg(test)]
    test_seams::record_load(root, IDENTIFIERS);
    Some((reused, parse))
}

/// The values of `first` and `second` read while `first` stays locked. An
/// identifier index install swaps the source under the index write lock, so
/// the snapshot never pairs one index with another index's source.
fn read_paired<A: Clone, B: Clone>(
    first: &tokio::sync::RwLock<Option<A>>,
    second: &tokio::sync::RwLock<Option<B>>,
) -> (Option<A>, Option<B>) {
    let first = first.blocking_read();
    #[cfg(test)]
    test_seams::between_paired_reads();
    let second = second.blocking_read().as_ref().cloned();
    (first.as_ref().cloned(), second)
}

/// Writes the identifier snapshot of `ref_index` in the background from the
/// identifier index it holds then.
pub(super) fn schedule_identifiers(
    config: &Config,
    ref_index: &Arc<crate::ref_index::RefIndex>,
    change: Change,
) {
    if enabled(config, ref_index) {
        change.request(
            &ref_index.snapshots.identifiers,
            IDENTIFIERS,
            identifiers_writer(config, ref_index),
        );
    }
}

fn identifiers_writer(
    config: &Config,
    ref_index: &Arc<crate::ref_index::RefIndex>,
) -> impl FnOnce() -> std::io::Result<bool> + Send + 'static {
    let root = ref_index.root_dir.clone();
    let config = config.clone();
    let owner = Arc::downgrade(ref_index);
    move || {
        let Some(owner) = owner.upgrade() else {
            return Ok(false);
        };
        let (index, source) = read_paired(&owner.identifier_index, &owner.identifier_source);
        drop(owner);
        match (index, source) {
            (Some(index), Some(source)) if index.dims > 0 => {
                write_identifiers(&root, &config, &index, &source)
            }
            _ => Ok(false),
        }
    }
}

// ---------------------------------------------------------------------------
// File documents
// ---------------------------------------------------------------------------

/// The parsed fields of the file documents in the snapshot of `ref_index`,
/// read once; `None` after a full walk has used them.
pub(crate) fn file_seed(
    config: &Config,
    ref_index: &crate::ref_index::RefIndex,
) -> Option<Arc<crate::tools::semantic_search::DocumentSeeds>> {
    use crate::ref_index::FileSeed;
    if !enabled(config, ref_index) {
        return None;
    }
    let mut seed = ref_index.snapshots.file_seed.lock().unwrap();
    if matches!(*seed, FileSeed::Unread) {
        let started = Instant::now();
        let read = Snapshot::open(
            &snapshot::snapshot_path(&ref_index.root_dir, FILES),
            &fingerprint(FILES, config),
        )
        .and_then(|snapshot| {
            let mut input = snapshot.reader();
            let seeds = crate::tools::semantic_search::read_document_seeds(&mut input)?;
            input.is_empty().then_some(seeds)
        });
        tracing::info!(
            phase = "file_snapshot_load",
            elapsed_ms = started.elapsed().as_millis(),
            documents = read.as_ref().map_or(0, |seeds| seeds.len()),
            "cold-start phase"
        );
        #[cfg(test)]
        if read.is_some() {
            test_seams::record_load(&ref_index.root_dir, FILES);
        }
        *seed = match read {
            Some(seeds) => FileSeed::Read(Arc::new(seeds)),
            None => FileSeed::Used,
        };
    }
    match &*seed {
        FileSeed::Read(seeds) => Some(Arc::clone(seeds)),
        _ => None,
    }
}

/// Drops the snapshot's file documents once a full walk has used them.
pub(crate) fn file_seed_used(ref_index: &crate::ref_index::RefIndex) {
    *ref_index.snapshots.file_seed.lock().unwrap() = crate::ref_index::FileSeed::Used;
}

/// Writes the file-documents snapshot of `ref_index` in the background from
/// the search index it holds then, when that index covers the whole checkout.
pub(crate) fn schedule_files(
    config: &Config,
    ref_index: &Arc<crate::ref_index::RefIndex>,
    change: Change,
) {
    if enabled(config, ref_index) {
        change.request(
            &ref_index.snapshots.files,
            FILES,
            files_writer(config, ref_index),
        );
    }
}

fn files_writer(
    config: &Config,
    ref_index: &crate::ref_index::RefIndex,
) -> impl FnOnce() -> std::io::Result<bool> + Send + 'static {
    let root = ref_index.root_dir.clone();
    let canonical_root = ref_index.canonical_root.clone();
    let config = config.clone();
    let slot = Arc::downgrade(&ref_index.search_index_cache);
    move || {
        let Some(cached) = slot
            .upgrade()
            .and_then(|slot| slot.blocking_read().as_ref().cloned())
        else {
            return Ok(false);
        };
        if cached.search_root() != canonical_root {
            return Ok(false);
        }
        write_files(&root, &config, &cached)
    }
}

/// Persists the parsed fields of the documents of a search index walked from
/// the checkout root.
pub(super) fn write_files(
    root: &Path,
    config: &Config,
    cached: &crate::tools::semantic_search::CachedSearchIndex,
) -> std::io::Result<bool> {
    let mut out = SnapshotWriter::create(
        &snapshot::snapshot_path(root, FILES),
        &fingerprint(FILES, config),
    )?;
    crate::tools::semantic_search::write_document_seeds(&mut out, cached.index.documents())?;
    out.commit()?;
    Ok(true)
}

#[cfg(test)]
pub(crate) mod test_seams {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, OnceLock};

    fn loads() -> &'static Mutex<HashMap<(PathBuf, &'static str), usize>> {
        static LOADS: OnceLock<Mutex<HashMap<(PathBuf, &'static str), usize>>> = OnceLock::new();
        LOADS.get_or_init(Default::default)
    }

    pub(super) fn record_load(root: &Path, kind: &'static str) {
        *loads()
            .lock()
            .unwrap()
            .entry((root.to_path_buf(), kind))
            .or_default() += 1;
    }

    /// Snapshots of `kind` read under `root` so far.
    pub(crate) fn load_count(root: &Path, kind: &'static str) -> usize {
        loads()
            .lock()
            .unwrap()
            .get(&(root.to_path_buf(), kind))
            .copied()
            .unwrap_or(0)
    }

    type Hook = Box<dyn FnOnce()>;

    thread_local! {
        static BETWEEN_PAIRED_READS: std::cell::RefCell<Option<Hook>> =
            const { std::cell::RefCell::new(None) };
    }

    /// Runs `hook` on this thread between the two reads of the next paired
    /// read.
    pub(crate) fn set_between_paired_reads(hook: impl FnOnce() + 'static) {
        BETWEEN_PAIRED_READS.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    pub(super) fn between_paired_reads() {
        if let Some(hook) = BETWEEN_PAIRED_READS.with(|slot| slot.borrow_mut().take()) {
            hook();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write as _;

    /// Digest of each kind's builder output for [`golden_corpus`] at its
    /// derivation version. When a builder's output changes, bump that kind's
    /// `*_DERIVATION` and record the new digest here.
    const GOLDEN: [(&str, u32, &str); 3] = [
        (KEYWORDS, 1, "98f58731ff5fa4e7"),
        (IDENTIFIERS, 1, "30d17ad82e762170"),
        (FILES, 1, "d405d7a42c24b63d"),
    ];

    fn golden_corpus() -> Vec<(&'static str, &'static str)> {
        vec![
            (
                "src/ledger.rs",
                "//! Ledger entries and their balances.\n\
                 use std::collections::HashMap;\n\
                 pub trait Balance { fn total(&self) -> i64; }\n\
                 pub struct Ledger { entries: HashMap<String, i64> }\n\
                 impl Ledger {\n    pub fn record_entry(&mut self, name: &str, amount: i64) {\n        \
                 self.entries.insert(name.to_owned(), amount);\n    }\n}\n\
                 impl Balance for Ledger { fn total(&self) -> i64 { self.entries.values().sum() } }\n",
            ),
            (
                "web/invoiceView.ts",
                "// Invoice screen for the billing flow\n\
                 export interface InvoiceProps { id: string; amountDue: number }\n\
                 export class InvoiceView {\n  render(props: InvoiceProps): string { return formatAmount(props.amountDue); }\n}\n\
                 export function formatAmount(value: number): string { return `$${value}`; }\n\
                 export const refundWindowDays = 30;\n",
            ),
            (
                "tools/report.py",
                "\"\"\"Monthly report generator.\"\"\"\n\
                 class ReportBuilder:\n    def add_row(self, row):\n        self.rows.append(row)\n\n\
                 def build_monthly_report(rows):\n    return ReportBuilder()\n",
            ),
            (
                "docs/guide.md",
                "# Payments guide\n\nHow a refund moves through the ledger.\n\n## Retries\n\nFailed refunds retry twice.\n",
            ),
            (
                "config/app.json",
                "{\n  \"refundWindowDays\": 30,\n  \"currency\": \"usd\"\n}\n",
            ),
            (
                "web/RefundPanel.tsx",
                "// Refund panel\n\
                 export interface PanelProps { amount: number }\n\
                 export function RefundPanel(props: PanelProps) { return <div>{props.amount}</div>; }\n\
                 export class PanelStore { load(id: string): void {} }\n",
            ),
            (
                "web/cart.js",
                "// Shopping cart helpers\n\
                 export class Cart {\n  addItem(item) { this.items.push(item); }\n}\n\
                 export function cartTotal(cart) { return cart.items.length; }\n\
                 var cartLimit = 10;\n",
            ),
            (
                "svc/payout.go",
                "// Package payout schedules payouts.\n\
                 package payout\n\n\
                 type Payout struct { Amount int64 }\n\n\
                 func (p *Payout) Settle(account string) error { return nil }\n\n\
                 func NewPayout(amount int64) *Payout { return &Payout{Amount: amount} }\n",
            ),
            (
                "svc/Invoice.java",
                "// Invoices for accounts\n\
                 public class Invoice {\n    public long total(int count) { return count; }\n}\n\
                 interface Payable { void pay(); }\n\
                 enum InvoiceState { OPEN, PAID }\n",
            ),
            (
                "native/ledger.c",
                "/* Ledger in C */\n\
                 struct entry { int amount; };\n\
                 enum kind { DEBIT, CREDIT };\n\
                 int ledger_sum(struct entry *entries, int count) { return count; }\n",
            ),
            (
                "native/ledger.cpp",
                "// Ledger in C++\n\
                 class Ledger {\n public:\n  int total() { return 0; }\n};\n\
                 struct Row { int amount; };\n\
                 int sum_rows(int count) { return count; }\n",
            ),
            (
                "scripts/deploy.sh",
                "#!/bin/bash\n# Deploy the service\n\
                 build_image() {\n  docker build .\n}\n\
                 function push_image {\n  docker push \"$1\"\n}\n",
            ),
            (
                "lib/billing.rb",
                "# Billing helpers\n\
                 module Billing\n  class Charge\n    def capture(amount)\n      amount\n    end\n\n    \
                 def self.build(id)\n      new\n    end\n  end\nend\n",
            ),
            (
                "app/Refund.php",
                "<?php\n// Refunds\n\
                 interface Refundable { public function refund(int $amount): bool; }\n\
                 class Refund {\n    public function issue(string $reason): void {}\n}\n\
                 function refund_total(array $rows): int { return count($rows); }\n",
            ),
            (
                "app/Account.cs",
                "// Accounts\n\
                 public class Account {\n    public decimal Balance(int days) { return 0; }\n}\n\
                 public interface IAccount { void Close(); }\n\
                 public struct Money { public int Cents; }\n\
                 public enum Tier { Free, Pro }\n",
            ),
            (
                "app/Wallet.kt",
                "// Wallet\n\
                 class Wallet(val owner: String) {\n    fun deposit(amount: Long): Long = amount\n}\n\
                 object WalletRegistry\n\
                 fun openWallet(owner: String): Wallet = Wallet(owner)\n",
            ),
            (
                "site/index.html",
                "<!doctype html>\n<html>\n<head><style>body { margin: 0; }</style></head>\n\
                 <body><div id=\"app\">Invoices</div><script>start();</script></body>\n</html>\n",
            ),
            (
                "site/theme.css",
                "/* Theme */\n\
                 .invoice { color: black; }\n\
                 @media (max-width: 600px) { .invoice { color: gray; } }\n\
                 @keyframes fade { from { opacity: 0; } to { opacity: 1; } }\n",
            ),
        ]
    }

    fn sorted_counts(counts: &crate::tools::lexical_search::DocumentTermCounts) -> String {
        let terms: BTreeMap<&String, &[u32; 4]> = counts.0.iter().collect();
        format!("{terms:?} {:?}", counts.1)
    }

    fn sorted_tokens(tokens: &std::collections::HashSet<String>) -> Vec<&String> {
        let mut tokens: Vec<&String> = tokens.iter().collect();
        tokens.sort_unstable();
        tokens
    }

    /// Digest of what each kind's builders derive from [`golden_corpus`].
    fn builder_digests() -> HashMap<&'static str, String> {
        let tmp = tempfile::tempdir().unwrap();
        for (path, content) in golden_corpus() {
            let full = tmp.path().join(path);
            std::fs::create_dir_all(full.parent().unwrap()).unwrap();
            std::fs::write(full, content).unwrap();
        }
        let cache = load_project_cache(tmp.path(), &Config::from_env(), None, false);
        let mut corpus = golden_corpus();
        corpus.sort_unstable();

        let mut keywords = String::new();
        for (path, _) in &corpus {
            let counts = lexical_term_counts(path, &cache.file_content);
            writeln!(keywords, "{path} {}", sorted_counts(&counts)).unwrap();
        }
        let mut document_paths = vec![corpus[0].0.to_owned()];
        for (slot, doc) in lexical_updates(&mut document_paths, corpus.iter().copied()) {
            let counts = crate::tools::lexical_search::document_term_counts(&(&doc).into());
            writeln!(
                keywords,
                "{slot} {} {:?} {:?} {}",
                doc.path,
                doc.header,
                doc.symbols,
                sorted_counts(&counts)
            )
            .unwrap();
        }

        let mut identifiers = String::new();
        for (path, content) in &corpus {
            let docs = crate::tools::semantic_identifiers::identifier_docs_for_file(path, content)
                .unwrap_or_default();
            for doc in docs {
                writeln!(
                    identifiers,
                    "{path} {:?} {} {} {} {} {:?} {:?} {:?} {:?} {:?}",
                    doc.header,
                    doc.name,
                    doc.kind,
                    doc.line,
                    doc.end_line,
                    doc.signature,
                    doc.parent_name,
                    sorted_tokens(&doc.name_token_set),
                    sorted_tokens(&doc.signature_token_set),
                    sorted_tokens(&doc.parent_token_set),
                )
                .unwrap();
            }
        }

        let mut files = String::new();
        for (path, content) in &corpus {
            let doc = crate::tools::semantic_search::file_document(
                (*path).to_owned(),
                content,
                crate::tools::semantic_search::semantic_embedding_content(path, content),
            );
            writeln!(
                files,
                "{path} {:?} {:?} {:?}",
                doc.header, doc.symbols, doc.symbol_entries
            )
            .unwrap();
        }

        [
            (KEYWORDS, keywords),
            (IDENTIFIERS, identifiers),
            (FILES, files),
        ]
        .into_iter()
        .map(|(kind, text)| {
            (
                kind,
                blake3::hash(text.as_bytes()).to_hex()[..16].to_owned(),
            )
        })
        .collect()
    }

    #[test]
    fn cold_start_builder_output_matches_its_derivation_version() {
        let digests = builder_digests();
        let mismatches: Vec<String> = GOLDEN
            .iter()
            .filter(|(kind, version, digest)| {
                *version != derivation_version(kind) || digests[kind] != *digest
            })
            .map(|(kind, _, _)| {
                format!(
                    "{kind}: derivation v{} now yields {}",
                    derivation_version(kind),
                    digests[kind]
                )
            })
            .collect();
        assert!(
            mismatches.is_empty(),
            "a snapshot builder's output changed; bump the kind's *_DERIVATION and record \
             its new digest in GOLDEN:\n{}",
            mismatches.join("\n")
        );
    }

    #[test]
    fn cold_start_snapshots_name_the_grammar_set_that_parsed_them() {
        let config = Config::from_env();
        let grammars = format!("|g{}|", crate::core::tree_sitter::grammar_identity());
        for kind in [KEYWORDS, IDENTIFIERS, FILES] {
            assert!(
                fingerprint(kind, &config).contains(&grammars),
                "{kind} snapshots outlive a change of grammars"
            );
        }
    }

    #[test]
    fn cold_start_golden_corpus_parses_every_grammar() {
        let corpus = golden_corpus();
        let missing: Vec<&str> = crate::core::tree_sitter::grammar_extensions()
            .filter(|(_, exts)| {
                !corpus.iter().any(|(path, content)| {
                    let ext = path.rsplit('.').next().unwrap_or("");
                    exts.contains(&ext)
                        && !crate::core::tree_sitter::parse_with_tree_sitter(content, ext)
                            .unwrap_or_default()
                            .is_empty()
                })
            })
            .map(|(name, _)| name)
            .collect();
        assert!(
            missing.is_empty(),
            "golden_corpus has no file with symbols for grammars {missing:?}; add one and \
             record the new GOLDEN digests"
        );
    }

    /// Digest of each kind's payload for [`payload_digests`]' fixed contents
    /// at [`snapshot::FORMAT_VERSION`]. When a payload layout changes, bump
    /// `FORMAT_VERSION` and record the new digests here.
    const PAYLOAD_GOLDEN: (u32, [(&str, &str); 3]) = (
        2,
        [
            (KEYWORDS, "f3ca98332e3884e0"),
            (IDENTIFIERS, "54092fb534a037e2"),
            (FILES, "17a7b28000777c56"),
        ],
    );

    /// Digest of the payload each kind writes for fixed contents, independent
    /// of what the builders derive.
    fn payload_digests() -> HashMap<&'static str, String> {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let config = Config::from_env();
        std::fs::create_dir_all(root.join("src")).unwrap();
        for (path, content) in [("src/a.rs", "alpha\n"), ("src/b.rs", "alpha beta\n")] {
            std::fs::write(root.join(path), content).unwrap();
        }
        let cache = Arc::new(load_project_cache(root, &config, None, false));

        // One term per document, so term ids do not depend on hash order.
        let mut index = crate::tools::lexical_search::LexicalIndex::with_capacity(3);
        for (path, term, counts) in [
            ("src/a.rs", "alpha", [1, 0, 0, 2]),
            ("src/b.rs", "alpha", [0, 1, 0, 0]),
            ("src/c.rs", "beta", [0, 0, 3, 1]),
        ] {
            index.push_counted(path, (HashMap::from([(term.to_owned(), counts)]), counts));
        }
        index.finish_build();
        let cached = CachedLexicalIndex {
            index,
            document_paths: ["src/a.rs", "src/b.rs", "src/c.rs"]
                .map(str::to_owned)
                .to_vec(),
            project_cache: Arc::clone(&cache),
            generation: 0,
            base: None,
        };
        assert!(write_keywords(root, &config, &cached).unwrap());

        let tokens = |tokens: &[&str]| tokens.iter().map(|token| (*token).to_owned()).collect();
        let docs = vec![
            IdentifierDoc::assemble(
                "src/a.rs",
                "alpha header",
                "Alpha".to_owned(),
                "struct".to_owned(),
                1,
                4,
                "pub struct Alpha".to_owned(),
                None,
                tokens(&["alpha"]),
                tokens(&["pub", "struct", "alpha"]),
                tokens(&[]),
            ),
            IdentifierDoc::assemble(
                "src/a.rs",
                "alpha header",
                "run".to_owned(),
                "method".to_owned(),
                2,
                3,
                "fn run(&self)".to_owned(),
                Some("Alpha".to_owned()),
                tokens(&["run"]),
                tokens(&["fn", "run", "self"]),
                tokens(&["alpha"]),
            ),
        ];
        let identifiers = IdentifierIndex {
            docs: Segmented::from_files(BTreeMap::from([("src/a.rs".to_owned(), Arc::new(docs))])),
            vectors: IdentifierVectorIndex::empty(),
            dims: 2,
            file_count: 2,
            built_at: Instant::now(),
        };
        assert!(write_identifiers(root, &config, &identifiers, &cache).unwrap());

        let mut seeded = crate::tools::semantic_search::SearchDocument::new(
            "src/a.rs".to_owned(),
            "alpha header".to_owned(),
            vec!["Alpha".to_owned(), "run".to_owned()],
            vec![
                crate::tools::semantic_search::SymbolSearchEntry {
                    name: "Alpha".to_owned(),
                    kind: Some("struct".to_owned()),
                    line: 1,
                    end_line: Some(4),
                    signature: Some("pub struct Alpha".to_owned()),
                },
                crate::tools::semantic_search::SymbolSearchEntry {
                    name: "run".to_owned(),
                    kind: None,
                    line: 2,
                    end_line: None,
                    signature: None,
                },
            ],
            "alpha\n".to_owned(),
        );
        seeded.source_hash = "source-hash".to_owned();
        let mut out = SnapshotWriter::create(
            &snapshot::snapshot_path(root, FILES),
            &fingerprint(FILES, &config),
        )
        .unwrap();
        crate::tools::semantic_search::write_document_seeds(&mut out, &[Arc::new(seeded)]).unwrap();
        out.commit().unwrap();

        [KEYWORDS, IDENTIFIERS, FILES]
            .into_iter()
            .map(|kind| {
                let snapshot = Snapshot::open(
                    &snapshot::snapshot_path(root, kind),
                    &fingerprint(kind, &config),
                )
                .unwrap();
                let digest = blake3::hash(snapshot.reader().rest()).to_hex()[..16].to_owned();
                (kind, digest)
            })
            .collect()
    }

    #[test]
    fn cold_start_snapshot_payload_layout_matches_its_format_version() {
        let digests = payload_digests();
        let (version, golden) = PAYLOAD_GOLDEN;
        let mismatches: Vec<String> = golden
            .iter()
            .filter(|(kind, digest)| {
                version != snapshot::FORMAT_VERSION || digests[kind] != *digest
            })
            .map(|(kind, _)| {
                format!(
                    "{kind}: format v{} now writes {}",
                    snapshot::FORMAT_VERSION,
                    digests[kind]
                )
            })
            .collect();
        assert!(
            mismatches.is_empty(),
            "a snapshot payload layout changed; bump FORMAT_VERSION and record the new \
             digests in PAYLOAD_GOLDEN:\n{}",
            mismatches.join("\n")
        );
    }

    /// Writes a snapshot of every kind for a one-file checkout at `root`.
    fn write_every_kind(root: &Path, config: &Config) {
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/lib.rs"), "pub fn shared() -> u32 { 1 }\n").unwrap();
        let cache = Arc::new(load_project_cache(root, config, None, false));
        let (index, document_paths) =
            build_lexical_index(["src/lib.rs"].into_iter(), &cache.file_content);
        let cached = CachedLexicalIndex {
            index,
            document_paths,
            project_cache: cache,
            generation: 0,
            base: None,
        };
        assert!(write_keywords(root, config, &cached).unwrap());
        for kind in [IDENTIFIERS, FILES] {
            let path = snapshot::snapshot_path(root, kind);
            let mut out = SnapshotWriter::create(&path, &fingerprint(kind, config)).unwrap();
            out.usize(0).unwrap();
            out.commit().unwrap();
        }
    }

    /// Whether the keyword, identifier and file snapshots at `root` load.
    fn loadable(root: &Path, config: &Config) -> [bool; 3] {
        let files = Snapshot::open(
            &snapshot::snapshot_path(root, FILES),
            &fingerprint(FILES, config),
        )
        .and_then(|snapshot| {
            crate::tools::semantic_search::read_document_seeds(&mut snapshot.reader())
        })
        .is_some();
        [
            read_keywords(root, config).is_some(),
            read_identifiers(root, config, &HashMap::new()).is_some(),
            files,
        ]
    }

    #[test]
    fn cold_start_bumped_derivation_or_other_config_invalidates_a_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::from_env();
        write_every_kind(tmp.path(), &config);
        assert_eq!(loadable(tmp.path(), &config), [true; 3]);

        let mut other = config.clone();
        other.embed_doc_prefix.push_str("other: ");
        assert_eq!(
            loadable(tmp.path(), &other),
            [true, false, false],
            "only keyword documents are independent of the embedding configuration"
        );

        for kind in [KEYWORDS, IDENTIFIERS, FILES] {
            let version = derivation_version(kind);
            let bumped = fingerprint(kind, &config)
                .replace(&format!("|d{version}|"), &format!("|d{}|", version + 1));
            let path = snapshot::snapshot_path(tmp.path(), kind);
            let mut out = SnapshotWriter::create(&path, &bumped).unwrap();
            out.usize(0).unwrap();
            out.commit().unwrap();
            assert!(Snapshot::open(&path, &bumped).is_some());
            assert!(
                Snapshot::open(&path, &fingerprint(kind, &config)).is_none(),
                "{kind} loaded a snapshot of another derivation version"
            );
        }
    }

    #[test]
    fn cold_start_keyword_snapshot_takes_path_priors_from_the_current_code() {
        let tmp = tempfile::tempdir().unwrap();
        let config = Config::from_env();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::write(tmp.path().join("src/lib.rs"), "pub fn shared() {}\n").unwrap();
        let cache = Arc::new(load_project_cache(tmp.path(), &config, None, false));
        let (index, _) = build_lexical_index(["src/lib.rs"].into_iter(), &cache.file_content);
        assert_eq!(index.document_prior(0), (1.0, false));
        // A document classified by other code than the current: its path now
        // reads as a generated test file.
        let cached = CachedLexicalIndex {
            index,
            document_paths: vec!["src/generated/lib.test.rs".to_owned()],
            project_cache: cache,
            generation: 0,
            base: None,
        };
        assert!(write_keywords(tmp.path(), &config, &cached).unwrap());

        let (loaded, _, _) = read_keywords(tmp.path(), &config).unwrap();
        let (prior, is_test) = loaded.document_prior(0);
        assert!(
            is_test,
            "the snapshot kept the test flag of the code that wrote it"
        );
        assert!(
            prior < 1.0,
            "the snapshot kept the prior of the code that wrote it"
        );
    }

    #[test]
    fn cold_start_one_unusable_snapshot_leaves_the_other_kinds_loading() {
        let config = Config::from_env();
        for (unusable, kind) in [KEYWORDS, IDENTIFIERS, FILES].into_iter().enumerate() {
            let tmp = tempfile::tempdir().unwrap();
            write_every_kind(tmp.path(), &config);
            std::fs::write(snapshot::snapshot_path(tmp.path(), kind), b"stale").unwrap();
            let mut expected = [true; 3];
            expected[unusable] = false;
            assert_eq!(loadable(tmp.path(), &config), expected, "{kind} unusable");
        }
    }

    #[test]
    fn cold_start_identifier_snapshot_never_pairs_an_old_index_with_a_new_source() {
        let index = Arc::new(tokio::sync::RwLock::new(Some(1_u32)));
        let source = Arc::new(tokio::sync::RwLock::new(Some(1_u32)));
        let installer = Arc::new(std::sync::Mutex::new(None));
        {
            let (index, source, installer) = (
                Arc::clone(&index),
                Arc::clone(&source),
                Arc::clone(&installer),
            );
            test_seams::set_between_paired_reads(move || {
                // An install as `install_identifier_index_if_current` does it:
                // the source is swapped under the index write lock.
                let handle = std::thread::spawn(move || {
                    let mut index = index.blocking_write();
                    *source.blocking_write() = Some(2);
                    *index = Some(2);
                });
                std::thread::sleep(std::time::Duration::from_millis(200));
                *installer.lock().unwrap() = Some(handle);
            });
        }

        let pair = read_paired(&index, &source);
        installer.lock().unwrap().take().unwrap().join().unwrap();
        assert!(
            pair == (Some(1), Some(1)) || pair == (Some(2), Some(2)),
            "the snapshot paired an index with another index's source: {pair:?}"
        );
    }

    #[test]
    fn cold_start_keyword_snapshot_after_heavy_deletions_is_compacted() {
        let tmp = tempfile::tempdir().unwrap();
        for i in 0..40 {
            let path = tmp.path().join(format!("src/file{i}.rs"));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(
                path,
                format!("pub fn shared_{i}() -> u32 {{ uniqueterm{i} + common }}\n"),
            )
            .unwrap();
        }
        let config = Config::from_env();
        let cache = Arc::new(load_project_cache(tmp.path(), &config, None, false));
        let mut paths: Vec<&str> = cache
            .file_entries
            .iter()
            .filter(|entry| !entry.is_directory)
            .map(|entry| entry.relative_path.as_str())
            .collect();
        paths.sort_unstable();
        let (mut index, mut document_paths) =
            build_lexical_index(paths.into_iter(), &cache.file_content);
        let deleted: Vec<usize> = (0..document_paths.len()).filter(|i| i % 4 != 0).collect();
        for &i in &deleted {
            document_paths[i].clear();
        }
        index.update_documents(Vec::new(), &deleted);
        let cached = CachedLexicalIndex {
            index,
            document_paths,
            project_cache: Arc::clone(&cache),
            generation: 0,
            base: None,
        };

        assert!(write_keywords(tmp.path(), &config, &cached).unwrap());
        let (loaded, loaded_paths, _) = read_keywords(tmp.path(), &config).unwrap();

        let live: Vec<&String> = cached
            .document_paths
            .iter()
            .filter(|path| !path.is_empty())
            .collect();
        assert_eq!(
            loaded.slot_count(),
            live.len(),
            "deleted slots were written"
        );
        assert_eq!(loaded_paths.iter().collect::<Vec<_>>(), live);
        assert_eq!(loaded.dead_term_count(), 0, "unused terms were written");
        let query = "common uniqueterm0 uniqueterm4 uniqueterm5";
        let by_path = |index: &crate::tools::lexical_search::LexicalIndex, paths: &[String]| {
            index
                .search(query, 10)
                .into_iter()
                .map(|(doc, score)| (paths[doc].clone(), score))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            by_path(&loaded, &loaded_paths),
            by_path(&cached.index, &cached.document_paths)
        );
    }
}
