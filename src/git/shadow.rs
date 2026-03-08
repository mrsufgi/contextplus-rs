// File-based restore point system for safe AI file writes
// Creates backups in .mcp_data/backups/ with manifest tracking

use crate::error::{ContextPlusError, Result};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;

const MAX_RESTORE_POINTS: usize = 100;
const BACKUP_DIR: &str = "backups";
const DATA_DIR: &str = ".mcp_data";
const MANIFEST_FILE: &str = "restore-points.json";

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RestorePoint {
    pub id: String,
    pub timestamp: u64,
    pub files: Vec<FileBackup>,
    pub description: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FileBackup {
    pub original_path: String,
    pub backup_path: String,
}

/// Validates restore point ID format: `rp-{digits}-{6 alphanumeric chars}`
pub fn validate_restore_point_id(id: &str) -> bool {
    let re = regex::Regex::new(r"^rp-\d+-[a-z0-9]{6}$").expect("valid regex");
    re.is_match(id)
}

/// Generates a unique restore point ID: `rp-{timestamp_ms}-{random_6}`
pub fn generate_restore_point_id() -> String {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let mut rng = rand::thread_rng();
    let random: String = (0..6)
        .map(|_| {
            let idx: u8 = rng.gen_range(0..36);
            if idx < 10 {
                (b'0' + idx) as char
            } else {
                (b'a' + idx - 10) as char
            }
        })
        .collect();
    format!("rp-{}-{}", timestamp, random)
}

fn data_dir(root_dir: &Path) -> PathBuf {
    root_dir.join(DATA_DIR)
}

fn manifest_path(root_dir: &Path) -> PathBuf {
    data_dir(root_dir).join(MANIFEST_FILE)
}

fn backup_dir(root_dir: &Path) -> PathBuf {
    data_dir(root_dir).join(BACKUP_DIR)
}

/// Encodes a relative file path into a flat backup filename.
/// Replaces `/` and `\` with `__` so the backup is a flat directory.
fn encode_backup_filename(relative_path: &str) -> String {
    relative_path.replace(['/', '\\'], "__")
}

async fn ensure_data_dir(root_dir: &Path) -> Result<()> {
    fs::create_dir_all(data_dir(root_dir)).await?;
    Ok(())
}

async fn load_manifest(root_dir: &Path) -> Result<Vec<RestorePoint>> {
    let path = manifest_path(root_dir);
    match fs::read_to_string(&path).await {
        Ok(content) => serde_json::from_str(&content)
            .map_err(|e| ContextPlusError::Serialization(format!("manifest parse: {}", e))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(e.into()),
    }
}

async fn save_manifest(root_dir: &Path, points: &[RestorePoint]) -> Result<()> {
    ensure_data_dir(root_dir).await?;
    let content = serde_json::to_string_pretty(points)
        .map_err(|e| ContextPlusError::Serialization(format!("manifest serialize: {}", e)))?;
    fs::write(manifest_path(root_dir), content).await?;
    Ok(())
}

/// Creates a restore point by backing up the specified files before overwrite.
///
/// `files_with_content` is a slice of `(relative_path, new_content)` tuples.
/// For each file, if it currently exists on disk, its original content is saved
/// to `.mcp_data/backups/{restore_point_id}/{encoded_filename}`.
///
/// Returns the created `RestorePoint` (before writing new content).
pub async fn create_restore_point(
    root_dir: &Path,
    files: &[&str],
    description: &str,
) -> Result<RestorePoint> {
    let id = generate_restore_point_id();
    let rp_backup_dir = backup_dir(root_dir).join(&id);
    fs::create_dir_all(&rp_backup_dir).await?;

    let mut file_backups = Vec::new();

    for &relative_path in files {
        let full_path = root_dir.join(relative_path);
        let backup_filename = encode_backup_filename(relative_path);
        let backup_path = rp_backup_dir.join(&backup_filename);

        // Only back up if the original file exists
        match fs::read(&full_path).await {
            Ok(content) => {
                fs::write(&backup_path, &content).await?;
                file_backups.push(FileBackup {
                    original_path: relative_path.to_string(),
                    backup_path: backup_filename,
                });
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // New file — no backup needed, but record the path so restore can delete it
                file_backups.push(FileBackup {
                    original_path: relative_path.to_string(),
                    backup_path: String::new(), // empty means "file didn't exist"
                });
            }
            Err(e) => return Err(e.into()),
        }
    }

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let point = RestorePoint {
        id,
        timestamp,
        files: file_backups,
        description: description.to_string(),
    };

    let mut manifest = load_manifest(root_dir).await?;
    manifest.push(point.clone());

    // Cap at MAX_RESTORE_POINTS, removing oldest first
    if manifest.len() > MAX_RESTORE_POINTS {
        let excess = manifest.len() - MAX_RESTORE_POINTS;
        manifest.drain(0..excess);
    }

    save_manifest(root_dir, &manifest).await?;
    Ok(point)
}

/// Restores files from a previously created restore point.
/// Returns the list of relative paths that were restored.
pub async fn restore_from_point(root_dir: &Path, restore_point_id: &str) -> Result<Vec<String>> {
    if !validate_restore_point_id(restore_point_id) {
        return Err(ContextPlusError::PathTraversal(format!(
            "invalid restore point ID: {}",
            restore_point_id
        )));
    }

    let manifest = load_manifest(root_dir).await?;
    let point = manifest
        .iter()
        .find(|p| p.id == restore_point_id)
        .ok_or_else(|| {
            ContextPlusError::Other(format!(
                "restore point {} not found",
                restore_point_id
            ))
        })?;

    let rp_backup_dir = backup_dir(root_dir).join(restore_point_id);
    let mut restored_files = Vec::new();

    for file_backup in &point.files {
        let target_path = root_dir.join(&file_backup.original_path);

        if file_backup.backup_path.is_empty() {
            // File didn't exist before — remove it to truly restore
            let _ = fs::remove_file(&target_path).await;
            restored_files.push(file_backup.original_path.clone());
            continue;
        }

        let backup_file = rp_backup_dir.join(&file_backup.backup_path);
        match fs::read(&backup_file).await {
            Ok(content) => {
                if let Some(parent) = target_path.parent() {
                    fs::create_dir_all(parent).await?;
                }
                fs::write(&target_path, &content).await?;
                restored_files.push(file_backup.original_path.clone());
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Backup file missing — skip silently (matches TS behavior)
            }
            Err(e) => return Err(e.into()),
        }
    }

    Ok(restored_files)
}

/// Lists all restore points from the manifest.
pub async fn list_restore_points(root_dir: &Path) -> Result<Vec<RestorePoint>> {
    load_manifest(root_dir).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn generate_id_matches_format() {
        for _ in 0..20 {
            let id = generate_restore_point_id();
            assert!(
                validate_restore_point_id(&id),
                "generated ID '{}' failed validation",
                id
            );
        }
    }

    #[test]
    fn validate_id_rejects_invalid() {
        assert!(!validate_restore_point_id(""));
        assert!(!validate_restore_point_id("rp-abc-123456"));
        assert!(!validate_restore_point_id("rp-123-12345")); // 5 chars
        assert!(!validate_restore_point_id("rp-123-1234567")); // 7 chars
        assert!(!validate_restore_point_id("xx-123-abcdef"));
        assert!(!validate_restore_point_id("rp-123-ABCDEF")); // uppercase
        assert!(!validate_restore_point_id("../rp-123-abcdef")); // path traversal
    }

    #[test]
    fn validate_id_accepts_valid() {
        assert!(validate_restore_point_id("rp-1234567890-abc123"));
        assert!(validate_restore_point_id("rp-0-000000"));
        assert!(validate_restore_point_id("rp-99-z9a8b7"));
    }

    #[test]
    fn encode_backup_filename_works() {
        assert_eq!(encode_backup_filename("src/main.rs"), "src__main.rs");
        assert_eq!(
            encode_backup_filename("a/b/c/d.ts"),
            "a__b__c__d.ts"
        );
        assert_eq!(
            encode_backup_filename("path\\to\\file.js"),
            "path__to__file.js"
        );
    }

    #[tokio::test]
    async fn create_and_list_restore_point() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        // Create a test file
        let test_file = root.join("src/hello.rs");
        fs::create_dir_all(test_file.parent().unwrap())
            .await
            .unwrap();
        fs::write(&test_file, "original content").await.unwrap();

        // Create restore point
        let rp = create_restore_point(root, &["src/hello.rs"], "test backup")
            .await
            .unwrap();
        assert!(validate_restore_point_id(&rp.id));
        assert_eq!(rp.description, "test backup");
        assert_eq!(rp.files.len(), 1);
        assert_eq!(rp.files[0].original_path, "src/hello.rs");

        // List should return the point
        let points = list_restore_points(root).await.unwrap();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].id, rp.id);
    }

    #[tokio::test]
    async fn create_and_restore() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        // Create a test file
        let test_file = root.join("file.txt");
        fs::write(&test_file, "original").await.unwrap();

        // Create restore point
        let rp = create_restore_point(root, &["file.txt"], "before edit")
            .await
            .unwrap();

        // Overwrite file
        fs::write(&test_file, "modified").await.unwrap();
        assert_eq!(fs::read_to_string(&test_file).await.unwrap(), "modified");

        // Restore
        let restored = restore_from_point(root, &rp.id).await.unwrap();
        assert_eq!(restored, vec!["file.txt"]);
        assert_eq!(fs::read_to_string(&test_file).await.unwrap(), "original");
    }

    #[tokio::test]
    async fn restore_new_file_deletes_it() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        // Create restore point for a file that doesn't exist yet
        let rp = create_restore_point(root, &["new_file.txt"], "before create")
            .await
            .unwrap();

        // Now create the file
        fs::write(root.join("new_file.txt"), "new content")
            .await
            .unwrap();
        assert!(root.join("new_file.txt").exists());

        // Restore should remove the file
        let restored = restore_from_point(root, &rp.id).await.unwrap();
        assert_eq!(restored, vec!["new_file.txt"]);
        assert!(!root.join("new_file.txt").exists());
    }

    #[tokio::test]
    async fn manifest_capped_at_100() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        // Create a test file to back up
        fs::write(root.join("f.txt"), "x").await.unwrap();

        // Create 105 restore points
        for i in 0..105 {
            create_restore_point(root, &["f.txt"], &format!("rp {}", i))
                .await
                .unwrap();
        }

        let points = list_restore_points(root).await.unwrap();
        assert_eq!(points.len(), MAX_RESTORE_POINTS);

        // First point should be rp 5 (0-4 were trimmed)
        assert_eq!(points[0].description, "rp 5");
        assert_eq!(points[99].description, "rp 104");
    }

    #[tokio::test]
    async fn restore_invalid_id_rejected() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        let result = restore_from_point(root, "../evil-path").await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ContextPlusError::PathTraversal(_)));
    }

    #[tokio::test]
    async fn restore_nonexistent_id_errors() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        let result = restore_from_point(root, "rp-999-abc123").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn multiple_files_in_restore_point() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        fs::create_dir_all(root.join("src")).await.unwrap();
        fs::write(root.join("a.txt"), "content a").await.unwrap();
        fs::write(root.join("src/b.rs"), "content b").await.unwrap();

        let rp = create_restore_point(root, &["a.txt", "src/b.rs"], "multi")
            .await
            .unwrap();
        assert_eq!(rp.files.len(), 2);

        // Overwrite both
        fs::write(root.join("a.txt"), "changed a").await.unwrap();
        fs::write(root.join("src/b.rs"), "changed b").await.unwrap();

        // Restore
        let restored = restore_from_point(root, &rp.id).await.unwrap();
        assert_eq!(restored.len(), 2);
        assert_eq!(
            fs::read_to_string(root.join("a.txt")).await.unwrap(),
            "content a"
        );
        assert_eq!(
            fs::read_to_string(root.join("src/b.rs")).await.unwrap(),
            "content b"
        );
    }

    #[tokio::test]
    async fn empty_manifest_returns_empty_vec() {
        let dir = TempDir::new().unwrap();
        let points = list_restore_points(dir.path()).await.unwrap();
        assert!(points.is_empty());
    }
}
