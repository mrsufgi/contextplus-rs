//! Built-state snapshots that let a restarted server skip rebuilding its
//! indexes.
//!
//! A snapshot is one file under `.mcp_data/snapshots/`: a magic tag, a
//! fingerprint string, a payload written by the index that owns it, and a
//! BLAKE3 checksum of everything before it. The fingerprint names the index
//! kind, the snapshot format version, the version of what the kind's builders
//! derive, and the configuration that shapes it; a snapshot whose
//! fingerprint differs, whose checksum fails or whose payload does not decode
//! is ignored and the index is built as before. Files are written to a
//! temporary path and renamed into place, so an interrupted write leaves the
//! previous snapshot readable. The existing `.rkyv` caches are never touched.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Bumped whenever the payload layout of any kind changes: nothing else
/// keeps a snapshot of the old layout from being read.
pub const FORMAT_VERSION: u32 = 1;
const MAGIC: [u8; 8] = *b"CPSNAPSH";
const CHECKSUM_LEN: usize = 32;
const SNAPSHOT_DIR: &str = "snapshots";

/// How long after a build its snapshot is written, so a burst of rebuilds
/// writes once.
const WRITE_DEBOUNCE: Duration = Duration::from_secs(2);
/// Fewest seconds between two writes of one snapshot.
const WRITE_MIN_INTERVAL: Duration = Duration::from_secs(30);
/// How long an index must go without an update before the drift of its
/// incremental updates is written.
const WRITE_IDLE: Duration = Duration::from_secs(5 * 60);

/// Content digest a snapshot manifest keys each file by.
pub type Digest = [u8; 16];

pub fn digest(bytes: &[u8]) -> Digest {
    let hash = blake3::hash(bytes);
    let mut out = [0; 16];
    out.copy_from_slice(&hash.as_bytes()[..16]);
    out
}

/// Fingerprint of a snapshot of `kind` built under `config_identity`, which
/// carries the version of what the kind's builders derive. It names no code
/// identity, so a release that changes neither the payload layout nor a
/// kind's builder output reads the previous release's snapshot of that kind.
pub fn fingerprint(kind: &str, config_identity: &str) -> String {
    format!("{kind}|v{FORMAT_VERSION}|{config_identity}")
}

/// Where the snapshot of `kind` for the checkout at `root` lives.
pub fn snapshot_path(root: &Path, kind: &str) -> PathBuf {
    crate::cache::rkyv_store::cache_dir(root)
        .join(SNAPSHOT_DIR)
        .join(format!("{kind}.snap"))
}

/// Age past which a snapshot temporary file belongs to no live writer: a
/// write takes seconds.
pub const STALE_TEMP_AGE: Duration = Duration::from_secs(10 * 60);

/// Temporary files of a writer killed before its commit: a hidden
/// `.*.tmp` under `snapshots/` older than `older_than`.
/// Removes them and returns how many it removed. Younger ones may belong to
/// a write in progress in another process and are kept.
pub fn remove_stale_temp_files(root: &Path, older_than: Duration) -> usize {
    let dir = crate::cache::rkyv_store::cache_dir(root).join(SNAPSHOT_DIR);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return 0;
    };
    let mut removed = 0;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !(name.starts_with('.') && name.ends_with(".tmp")) {
            continue;
        }
        let stale = entry
            .metadata()
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| modified.elapsed().ok())
            .is_some_and(|age| age >= older_than);
        if stale && std::fs::remove_file(entry.path()).is_ok() {
            removed += 1;
        }
    }
    if removed > 0 {
        tracing::info!(removed, dir = %dir.display(), "removed orphaned snapshot temporary files");
    }
    removed
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Streams a snapshot to a temporary file; [`commit`](Self::commit) renames it
/// into place. Dropped uncommitted, it removes the temporary file and leaves
/// the previous snapshot as it was.
pub struct SnapshotWriter {
    file: io::BufWriter<std::fs::File>,
    hasher: blake3::Hasher,
    temp: PathBuf,
    path: PathBuf,
    committed: bool,
}

impl SnapshotWriter {
    pub fn create(path: &Path, fingerprint: &str) -> io::Result<Self> {
        let dir = path
            .parent()
            .ok_or_else(|| io::Error::other("snapshot path has no parent"))?;
        std::fs::create_dir_all(dir)?;
        let temp = dir.join(format!(
            ".{}.{}.{}.tmp",
            path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("snapshot"),
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let file = io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&temp)?);
        let mut writer = Self {
            file,
            hasher: blake3::Hasher::new(),
            temp,
            path: path.to_path_buf(),
            committed: false,
        };
        writer.raw(&MAGIC)?;
        writer.str(fingerprint)?;
        Ok(writer)
    }

    fn raw(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.hasher.update(bytes);
        self.file.write_all(bytes)
    }

    pub fn u8(&mut self, value: u8) -> io::Result<()> {
        self.raw(&[value])
    }

    pub fn u32(&mut self, value: u32) -> io::Result<()> {
        self.raw(&value.to_le_bytes())
    }

    pub fn u64(&mut self, value: u64) -> io::Result<()> {
        self.raw(&value.to_le_bytes())
    }

    pub fn usize(&mut self, value: usize) -> io::Result<()> {
        self.u64(value as u64)
    }

    pub fn f64(&mut self, value: f64) -> io::Result<()> {
        self.raw(&value.to_le_bytes())
    }

    pub fn bool(&mut self, value: bool) -> io::Result<()> {
        self.u8(value as u8)
    }

    pub fn str(&mut self, value: &str) -> io::Result<()> {
        self.usize(value.len())?;
        self.raw(value.as_bytes())
    }

    pub fn opt_str(&mut self, value: Option<&str>) -> io::Result<()> {
        match value {
            Some(value) => {
                self.u8(1)?;
                self.str(value)
            }
            None => self.u8(0),
        }
    }

    pub fn strs<'a>(&mut self, values: impl ExactSizeIterator<Item = &'a str>) -> io::Result<()> {
        self.usize(values.len())?;
        for value in values {
            self.str(value)?;
        }
        Ok(())
    }

    pub fn u32s(&mut self, values: &[u32]) -> io::Result<()> {
        self.usize(values.len())?;
        let mut bytes = Vec::with_capacity(values.len().min(1 << 16) * 4);
        for chunk in values.chunks(1 << 16) {
            bytes.clear();
            bytes.extend(chunk.iter().flat_map(|value| value.to_le_bytes()));
            self.raw(&bytes)?;
        }
        Ok(())
    }

    /// A run of `len` values from `values`, read back with
    /// [`SnapshotReader::u32_seq`]. Streams without collecting the values.
    pub fn u32_seq(&mut self, len: usize, values: impl Iterator<Item = u32>) -> io::Result<()> {
        self.usize(len)?;
        let mut written = 0;
        let mut bytes = Vec::with_capacity(1 << 18);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
            written += 1;
            if bytes.len() >= 1 << 18 {
                self.raw(&bytes)?;
                bytes.clear();
            }
        }
        self.raw(&bytes)?;
        if written != len {
            return Err(io::Error::other("snapshot sequence length mismatch"));
        }
        Ok(())
    }

    pub fn digest(&mut self, value: &Digest) -> io::Result<()> {
        self.raw(value)
    }

    /// Seals the snapshot with its checksum and moves it into place.
    pub fn commit(mut self) -> io::Result<()> {
        let checksum = *self.hasher.finalize().as_bytes();
        self.file.write_all(&checksum)?;
        self.file.flush()?;
        self.file.get_ref().sync_all()?;
        std::fs::rename(&self.temp, &self.path)?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for SnapshotWriter {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_file(&self.temp);
        }
    }
}

/// A verified snapshot, mapped read-only.
pub struct Snapshot {
    map: memmap2::Mmap,
    payload: usize,
}

impl Snapshot {
    /// The snapshot at `path` when it is intact and was written under
    /// `fingerprint`; otherwise `None`.
    pub fn open(path: &Path, fingerprint: &str) -> Option<Self> {
        let file = std::fs::File::open(path).ok()?;
        // SAFETY: read-only map of a file only ever replaced by rename, never
        // written in place.
        let map = unsafe { memmap2::Mmap::map(&file) }.ok()?;
        if map.len() < MAGIC.len() + 8 + CHECKSUM_LEN || map[..MAGIC.len()] != MAGIC {
            return None;
        }
        let body = map.len() - CHECKSUM_LEN;
        let mut reader = SnapshotReader {
            bytes: &map[..body],
            pos: MAGIC.len(),
        };
        if reader.str()? != fingerprint {
            return None;
        }
        let payload = reader.pos;
        if blake3::hash(&map[..body]).as_bytes() != &map[body..] {
            tracing::warn!(path = %path.display(), "snapshot checksum mismatch; ignoring it");
            return None;
        }
        Some(Self { map, payload })
    }

    pub fn reader(&self) -> SnapshotReader<'_> {
        SnapshotReader {
            bytes: &self.map[..self.map.len() - CHECKSUM_LEN],
            pos: self.payload,
        }
    }
}

/// Reads a payload; every read is `None` past the end of the payload.
pub struct SnapshotReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> SnapshotReader<'a> {
    fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(len)?;
        let bytes = self.bytes.get(self.pos..end)?;
        self.pos = end;
        Some(bytes)
    }

    fn array<const N: usize>(&mut self) -> Option<[u8; N]> {
        self.take(N)?.try_into().ok()
    }

    pub fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    pub fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.array()?))
    }

    pub fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.array()?))
    }

    /// A length or count; never more than the bytes left, so a corrupt count
    /// cannot drive a huge allocation.
    pub fn usize(&mut self) -> Option<usize> {
        let value = usize::try_from(self.u64()?).ok()?;
        (value <= self.bytes.len() - self.pos).then_some(value)
    }

    /// A plain number, such as a line.
    pub fn usize_value(&mut self) -> Option<usize> {
        usize::try_from(self.u64()?).ok()
    }

    pub fn f64(&mut self) -> Option<f64> {
        Some(f64::from_le_bytes(self.array()?))
    }

    pub fn bool(&mut self) -> Option<bool> {
        match self.u8()? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }

    pub fn str(&mut self) -> Option<&'a str> {
        let len = self.usize()?;
        std::str::from_utf8(self.take(len)?).ok()
    }

    pub fn opt_str(&mut self) -> Option<Option<&'a str>> {
        match self.u8()? {
            0 => Some(None),
            1 => Some(Some(self.str()?)),
            _ => None,
        }
    }

    pub fn strings(&mut self) -> Option<Vec<String>> {
        let len = self.usize()?;
        let mut strings = Vec::with_capacity(len);
        for _ in 0..len {
            strings.push(self.str()?.to_owned());
        }
        Some(strings)
    }

    pub fn u32s(&mut self) -> Option<Vec<u32>> {
        let mut values = self.u32_seq()?;
        let mut out = Vec::with_capacity(values.len());
        while let Some(value) = values.next_value() {
            out.push(value);
        }
        Some(out)
    }

    /// A run written by [`SnapshotWriter::u32_seq`], read in place.
    pub fn u32_seq(&mut self) -> Option<U32Seq<'a>> {
        let len = self.usize()?;
        let bytes = self.take(len.checked_mul(4)?)?;
        Some(U32Seq { bytes })
    }

    pub fn digest(&mut self) -> Option<Digest> {
        self.array()
    }

    /// Whether the whole payload has been read.
    pub fn is_empty(&self) -> bool {
        self.pos == self.bytes.len()
    }
}

/// Values of a `u32` run, decoded as they are taken.
pub struct U32Seq<'a> {
    bytes: &'a [u8],
}

impl U32Seq<'_> {
    pub fn len(&self) -> usize {
        self.bytes.len() / 4
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The next value; `None` once the run is used up.
    pub fn next_value(&mut self) -> Option<u32> {
        let (head, tail) = self.bytes.split_first_chunk::<4>()?;
        self.bytes = tail;
        Some(u32::from_le_bytes(*head))
    }
}

/// Whether `changed` files since the last write are worth rewriting a
/// snapshot of an index of `documents`: 2% of them, at most 200.
pub fn drift_due(changed: usize, documents: usize) -> bool {
    changed >= (documents / 50).clamp(1, 200)
}

/// Unwritten-change count of a full build whose snapshot is not yet written.
const FULL_BUILD: usize = usize::MAX;

/// Writes one snapshot in the background: soon after a full build, and after
/// incremental updates only once enough has drifted and the index is idle.
/// At most once per interval, one write at a time, on a blocking thread.
pub struct WriteSchedule {
    ticket: AtomicU64,
    last_write: std::sync::Mutex<Option<tokio::time::Instant>>,
    writing: tokio::sync::Mutex<()>,
    debounce: Duration,
    min_interval: Duration,
    idle: Duration,
    /// Files changed since the last write, or [`FULL_BUILD`].
    unwritten: AtomicUsize,
}

impl Default for WriteSchedule {
    fn default() -> Self {
        Self::new(WRITE_DEBOUNCE, WRITE_MIN_INTERVAL)
    }
}

impl WriteSchedule {
    pub fn new(debounce: Duration, min_interval: Duration) -> Self {
        Self {
            ticket: AtomicU64::new(0),
            last_write: std::sync::Mutex::new(None),
            writing: tokio::sync::Mutex::new(()),
            debounce,
            min_interval,
            idle: WRITE_IDLE,
            unwritten: AtomicUsize::new(0),
        }
    }

    /// Writes after a full build: once the debounce has passed with no newer
    /// request and the minimum interval since the last write has elapsed. A
    /// newer request supersedes this one. Without a Tokio runtime nothing is
    /// written.
    pub fn schedule(
        self: &Arc<Self>,
        kind: &'static str,
        write: impl FnOnce() -> io::Result<bool> + Send + 'static,
    ) {
        self.unwritten.store(FULL_BUILD, Ordering::Release);
        self.request(kind, self.debounce, write);
    }

    fn request(
        self: &Arc<Self>,
        kind: &'static str,
        delay: Duration,
        write: impl FnOnce() -> io::Result<bool> + Send + 'static,
    ) {
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let ticket = self.ticket.fetch_add(1, Ordering::AcqRel) + 1;
        let schedule = Arc::clone(self);
        runtime.spawn(async move {
            tokio::time::sleep(delay).await;
            if schedule.ticket.load(Ordering::Acquire) != ticket {
                return;
            }
            let _writing = schedule.writing.lock().await;
            let wait = schedule
                .last_write
                .lock()
                .unwrap()
                .map(|last| schedule.min_interval.saturating_sub(last.elapsed()));
            if let Some(wait) = wait.filter(|wait| !wait.is_zero()) {
                tokio::time::sleep(wait).await;
                if schedule.ticket.load(Ordering::Acquire) != ticket {
                    return;
                }
            }
            let started = Instant::now();
            let taken = schedule.unwritten.swap(0, Ordering::AcqRel);
            let result = tokio::task::spawn_blocking(write).await;
            *schedule.last_write.lock().unwrap() = Some(tokio::time::Instant::now());
            if !matches!(result, Ok(Ok(_))) {
                schedule.add_unwritten(taken);
            }
            match result {
                Ok(Ok(true)) => tracing::info!(
                    kind,
                    elapsed_ms = started.elapsed().as_millis(),
                    "snapshot written"
                ),
                Ok(Ok(false)) => tracing::debug!(kind, "snapshot skipped: state was replaced"),
                Ok(Err(error)) => tracing::warn!(kind, %error, "snapshot write failed"),
                Err(error) => tracing::warn!(kind, %error, "snapshot write task failed"),
            }
        });
    }

    /// Records an incremental update of `changed` files to an index of
    /// `documents`. Its snapshot is written once the files changed since the
    /// last write reach [`drift_due`] and no update has followed for the idle
    /// period; smaller drift waits for [`flush`](Self::flush) at shutdown and
    /// is otherwise absorbed by the startup reconcile.
    pub fn schedule_drift(
        self: &Arc<Self>,
        kind: &'static str,
        changed: usize,
        documents: usize,
        write: impl FnOnce() -> io::Result<bool> + Send + 'static,
    ) {
        let unwritten = self.add_unwritten(changed);
        if unwritten == FULL_BUILD {
            self.request(kind, self.debounce, write);
        } else if drift_due(unwritten, documents) {
            self.request(kind, self.idle, write);
        }
    }

    /// Writes now, on this thread, when anything is unwritten, and drops
    /// every pending request. For a graceful shutdown.
    pub fn flush(
        &self,
        kind: &'static str,
        write: impl FnOnce() -> io::Result<bool>,
    ) -> io::Result<bool> {
        self.ticket.fetch_add(1, Ordering::AcqRel);
        let taken = self.unwritten.swap(0, Ordering::AcqRel);
        if taken == 0 {
            return Ok(false);
        }
        let result = write();
        match &result {
            Ok(true) => tracing::info!(kind, "snapshot written at shutdown"),
            Ok(false) => {}
            Err(error) => {
                self.add_unwritten(taken);
                tracing::warn!(kind, %error, "snapshot write at shutdown failed");
            }
        }
        result
    }

    fn add_unwritten(&self, changed: usize) -> usize {
        let previous = self
            .unwritten
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |unwritten| {
                Some(unwritten.saturating_add(changed))
            })
            .unwrap_or_default();
        previous.saturating_add(changed)
    }

    /// Drops every pending request.
    #[cfg(test)]
    pub(crate) fn cancel(&self) {
        self.ticket.fetch_add(1, Ordering::AcqRel);
    }

    #[cfg(test)]
    pub(crate) async fn idle(&self) {
        let _writing = self.writing.lock().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, fingerprint: &str, values: &[u32]) {
        let mut writer = SnapshotWriter::create(path, fingerprint).unwrap();
        writer.str("payload").unwrap();
        writer.u32s(values).unwrap();
        writer.commit().unwrap();
    }

    fn read(path: &Path, fingerprint: &str) -> Option<Vec<u32>> {
        let snapshot = Snapshot::open(path, fingerprint)?;
        let mut reader = snapshot.reader();
        assert_eq!(reader.str()?, "payload");
        let values = reader.u32s()?;
        reader.is_empty().then_some(values)
    }

    #[test]
    fn cold_start_snapshot_round_trips_its_payload() {
        let dir = tempfile::tempdir().unwrap();
        let path = snapshot_path(dir.path(), "kind");
        write(&path, "fp", &[1, 2, 3]);
        assert_eq!(read(&path, "fp"), Some(vec![1, 2, 3]));
    }

    #[test]
    fn cold_start_snapshot_from_another_config_or_version_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let path = snapshot_path(dir.path(), "kind");
        write(&path, &fingerprint("kind", "model-a"), &[1]);
        assert!(read(&path, &fingerprint("kind", "model-b")).is_none());
        let older = fingerprint("kind", "model-a").replace(
            &format!("|v{FORMAT_VERSION}|"),
            &format!("|v{}|", FORMAT_VERSION - 1),
        );
        write(&path, &older, &[1]);
        assert!(read(&path, &fingerprint("kind", "model-a")).is_none());
    }

    #[test]
    fn cold_start_corrupt_or_truncated_snapshot_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let path = snapshot_path(dir.path(), "kind");
        write(&path, "fp", &[7; 64]);
        let mut bytes = std::fs::read(&path).unwrap();
        let middle = bytes.len() / 2;
        bytes[middle] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();
        assert!(read(&path, "fp").is_none(), "a flipped byte must fail");
        bytes[middle] ^= 0xff;
        std::fs::write(&path, &bytes[..bytes.len() - 5]).unwrap();
        assert!(read(&path, "fp").is_none(), "a truncated file must fail");
        std::fs::write(&path, b"garbage").unwrap();
        assert!(read(&path, "fp").is_none());
    }

    #[test]
    fn cold_start_interrupted_write_leaves_the_previous_snapshot_readable() {
        let dir = tempfile::tempdir().unwrap();
        let path = snapshot_path(dir.path(), "kind");
        write(&path, "fp", &[1, 2]);
        {
            let mut writer = SnapshotWriter::create(&path, "fp").unwrap();
            writer.str("payload").unwrap();
            writer.u32s(&[9; 1024]).unwrap();
            // Dropped before commit: the writer was interrupted.
        }
        assert_eq!(read(&path, "fp"), Some(vec![1, 2]));
        let leftovers: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(leftovers.len(), 1, "temporary files remain: {leftovers:?}");
    }

    #[test]
    fn cold_start_reader_rejects_counts_past_the_payload() {
        let dir = tempfile::tempdir().unwrap();
        let path = snapshot_path(dir.path(), "kind");
        let mut writer = SnapshotWriter::create(&path, "fp").unwrap();
        writer.u64(u64::MAX / 2).unwrap();
        writer.commit().unwrap();
        let snapshot = Snapshot::open(&path, "fp").unwrap();
        assert!(snapshot.reader().u32s().is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn cold_start_scheduled_writes_are_debounced_and_bounded() {
        let schedule = Arc::new(WriteSchedule::new(
            Duration::from_secs(2),
            Duration::from_secs(30),
        ));
        let writes = Arc::new(AtomicU64::new(0));
        let request = |schedule: &Arc<WriteSchedule>| {
            let writes = Arc::clone(&writes);
            schedule.schedule("test", move || {
                writes.fetch_add(1, Ordering::SeqCst);
                Ok(true)
            });
        };
        for _ in 0..5 {
            request(&schedule);
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        tokio::time::sleep(Duration::from_secs(3)).await;
        schedule.idle().await;
        assert_eq!(writes.load(Ordering::SeqCst), 1, "a burst writes once");

        request(&schedule);
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert_eq!(
            writes.load(Ordering::SeqCst),
            1,
            "a second write waits out the minimum interval"
        );
        tokio::time::sleep(Duration::from_secs(30)).await;
        schedule.idle().await;
        assert_eq!(writes.load(Ordering::SeqCst), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn cold_start_incremental_updates_write_only_after_drift_and_idle() {
        let schedule = Arc::new(WriteSchedule::new(
            Duration::from_secs(2),
            Duration::from_secs(30),
        ));
        let writes = Arc::new(AtomicU64::new(0));
        let written = || writes.load(Ordering::SeqCst);
        let counter = || {
            let writes = Arc::clone(&writes);
            move || {
                writes.fetch_add(1, Ordering::SeqCst);
                Ok(true)
            }
        };
        // Of 1000 documents, 2% is 20 changed files.
        for _ in 0..19 {
            schedule.schedule_drift("test", 1, 1000, counter());
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        tokio::time::sleep(WRITE_IDLE * 2).await;
        schedule.idle().await;
        assert_eq!(written(), 0, "drift below the threshold is not written");

        schedule.schedule_drift("test", 1, 1000, counter());
        tokio::time::sleep(WRITE_IDLE - Duration::from_secs(60)).await;
        assert_eq!(written(), 0, "the index was not idle long enough");
        schedule.schedule_drift("test", 1, 1000, counter());
        tokio::time::sleep(WRITE_IDLE - Duration::from_secs(60)).await;
        assert_eq!(written(), 0, "a newer update restarts the idle wait");
        tokio::time::sleep(Duration::from_secs(120)).await;
        schedule.idle().await;
        assert_eq!(
            written(),
            1,
            "drift past the threshold is written once idle"
        );

        assert!(
            !schedule.flush("test", counter()).unwrap(),
            "nothing is unwritten after a write"
        );
        schedule.schedule_drift("test", 1, 1000, counter());
        assert!(
            schedule.flush("test", counter()).unwrap(),
            "shutdown writes what is unwritten"
        );
        assert_eq!(written(), 2);
        tokio::time::sleep(WRITE_IDLE * 2).await;
        schedule.idle().await;
        assert_eq!(written(), 2);
    }

    #[test]
    fn cold_start_stale_temp_files_are_removed_at_startup() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            remove_stale_temp_files(dir.path(), Duration::from_secs(600)),
            0
        );
        let path = snapshot_path(dir.path(), "kind");
        write(&path, "fp", &[1]);
        let snapshots = path.parent().unwrap();
        let stale = snapshots.join(".kind.snap.4242.0.tmp");
        let fresh = snapshots.join(".kind.snap.4243.0.tmp");
        let unrelated = snapshots.join("notes.tmp");
        let hour_ago = std::time::SystemTime::now() - Duration::from_secs(3600);
        for file in [&stale, &fresh, &unrelated] {
            std::fs::write(file, b"partial").unwrap();
        }
        for file in [&stale, &unrelated] {
            std::fs::File::options()
                .write(true)
                .open(file)
                .unwrap()
                .set_modified(hour_ago)
                .unwrap();
        }

        assert_eq!(
            remove_stale_temp_files(dir.path(), Duration::from_secs(600)),
            1
        );
        assert!(!stale.exists(), "an orphaned temporary file remains");
        assert!(fresh.exists(), "a write in progress lost its file");
        assert!(unrelated.exists());
        assert_eq!(read(&path, "fp"), Some(vec![1]));
    }

    #[test]
    fn cold_start_snapshot_from_a_release_with_other_code_still_loads() {
        // What another release writes for the same kind, format version and
        // configuration, whatever else its code changed.
        let dir = tempfile::tempdir().unwrap();
        let path = snapshot_path(dir.path(), "kind");
        write(&path, &format!("kind|v{FORMAT_VERSION}|model-a"), &[1]);
        assert_eq!(read(&path, &fingerprint("kind", "model-a")), Some(vec![1]));
    }
}
