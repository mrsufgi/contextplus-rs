// Process lifecycle helpers for resilient MCP stdio shutdown behavior handling.
// Ports the TypeScript process-lifecycle.ts to async Rust with tokio.

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::Notify;
use tokio::time::{Duration, sleep};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const DEFAULT_IDLE_TIMEOUT_MS: u64 = 15 * 60 * 1000; // 15 min
pub const MIN_IDLE_TIMEOUT_MS: u64 = 60 * 1000; // 1 min
pub const DEFAULT_PARENT_POLL_MS: u64 = 5 * 1000; // 5 s
pub const MIN_PARENT_POLL_MS: u64 = 1 * 1000; // 1 s

const DISABLED_VALUES: &[&str] = &["0", "false", "off", "disabled", "none"];

// ---------------------------------------------------------------------------
// Config parsing helpers
// ---------------------------------------------------------------------------

/// Parse idle shutdown timeout from an env-var string.
/// Returns 0 (disabled) for "0"/"false"/"off"/"disabled"/"none".
/// Otherwise clamps to at least `MIN_IDLE_TIMEOUT_MS`.
pub fn get_idle_shutdown_ms(value: Option<&str>) -> u64 {
    let normalized = match value {
        Some(v) => v.trim().to_lowercase(),
        None => return DEFAULT_IDLE_TIMEOUT_MS,
    };
    if normalized.is_empty() {
        return DEFAULT_IDLE_TIMEOUT_MS;
    }
    if DISABLED_VALUES.contains(&normalized.as_str()) {
        return 0;
    }
    match normalized.parse::<u64>() {
        Ok(v) => v.max(MIN_IDLE_TIMEOUT_MS),
        Err(_) => DEFAULT_IDLE_TIMEOUT_MS,
    }
}

/// Parse parent-poll interval from an env-var string.
/// Clamps to at least `MIN_PARENT_POLL_MS`.
pub fn get_parent_poll_ms(value: Option<&str>) -> u64 {
    let raw = match value {
        Some(v) => v.trim().parse::<u64>().unwrap_or(DEFAULT_PARENT_POLL_MS),
        None => DEFAULT_PARENT_POLL_MS,
    };
    raw.max(MIN_PARENT_POLL_MS)
}

// ---------------------------------------------------------------------------
// Broken pipe detection
// ---------------------------------------------------------------------------

/// Returns `true` if the IO error represents a broken pipe / connection reset.
pub fn is_broken_pipe_error(err: &io::Error) -> bool {
    matches!(
        err.kind(),
        io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset
    )
}

// ---------------------------------------------------------------------------
// Process alive check (Unix)
// ---------------------------------------------------------------------------

/// Check whether a process with `pid` is still alive using `kill(pid, 0)`.
///
/// Returns `false` for pid == 0.
#[cfg(unix)]
pub fn is_process_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    // SAFETY: kill with signal 0 does not send a signal; it only checks existence.
    let ret = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if ret == 0 {
        return true;
    }
    // ESRCH = no such process -> not alive.  Other errors (e.g. EPERM) -> alive.
    let errno = io::Error::last_os_error().raw_os_error().unwrap_or(0);
    errno != libc::ESRCH
}

#[cfg(not(unix))]
pub fn is_process_alive(_pid: u32) -> bool {
    // Non-Unix: conservatively assume alive; parent monitor is a no-op.
    true
}

// ---------------------------------------------------------------------------
// Idle shutdown monitor
// ---------------------------------------------------------------------------

/// Handle returned by [`create_idle_monitor`].
/// Call `touch()` on every tool invocation to reset the timer.
/// Call `stop()` (or drop) to cancel the background task.
#[derive(Clone)]
pub struct IdleMonitor {
    notify: Arc<Notify>,
    stopped: Arc<AtomicBool>,
}

impl IdleMonitor {
    /// Reset the idle timer (e.g. on each tool call).
    pub fn touch(&self) {
        if !self.stopped.load(Ordering::Relaxed) {
            self.notify.notify_one();
        }
    }

    /// Stop the idle monitor. The background task will exit.
    pub fn stop(&self) {
        self.stopped.store(true, Ordering::Relaxed);
        self.notify.notify_one();
    }
}

/// Creates an idle monitor that calls `on_idle` after `timeout_ms` of inactivity.
///
/// If `timeout_ms == 0`, returns a no-op monitor (no background task spawned).
pub fn create_idle_monitor<F>(timeout_ms: u64, on_idle: F) -> IdleMonitor
where
    F: Fn() + Send + Sync + 'static,
{
    let notify = Arc::new(Notify::new());
    let stopped = Arc::new(AtomicBool::new(false));

    if timeout_ms > 0 {
        let n = notify.clone();
        let s = stopped.clone();
        let timeout = Duration::from_millis(timeout_ms);

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = sleep(timeout) => {
                        if !s.load(Ordering::Relaxed) {
                            on_idle();
                        }
                        return;
                    }
                    _ = n.notified() => {
                        if s.load(Ordering::Relaxed) {
                            return;
                        }
                        // Timer reset -- loop again with fresh sleep.
                    }
                }
            }
        });
    } else {
        // Mark as stopped immediately -- touch/stop are no-ops.
        stopped.store(true, Ordering::Relaxed);
    }

    IdleMonitor { notify, stopped }
}

// ---------------------------------------------------------------------------
// Parent PID monitor
// ---------------------------------------------------------------------------

/// Handle returned by [`start_parent_monitor`]. Call `stop()` to cancel polling.
#[derive(Clone)]
pub struct ParentMonitorHandle {
    stopped: Arc<AtomicBool>,
}

impl ParentMonitorHandle {
    pub fn stop(&self) {
        self.stopped.store(true, Ordering::Relaxed);
    }
}

/// Start a background task that polls whether the parent process is still alive.
///
/// Guards:
/// - `parent_pid <= 1` -> no-op (PID 1 is init)
/// - `parent_pid == current PID` -> no-op
///
/// Calls `on_parent_exit` once if the parent dies or we get re-parented.
pub fn start_parent_monitor<F>(
    parent_pid: u32,
    poll_interval_ms: u64,
    on_parent_exit: F,
) -> ParentMonitorHandle
where
    F: Fn() + Send + Sync + 'static,
{
    let stopped = Arc::new(AtomicBool::new(false));

    let self_pid = std::process::id();
    if parent_pid <= 1 || parent_pid == self_pid {
        // Return already-stopped handle.
        stopped.store(true, Ordering::Relaxed);
        return ParentMonitorHandle { stopped };
    }

    let interval = Duration::from_millis(poll_interval_ms.max(MIN_PARENT_POLL_MS));
    let s = stopped.clone();

    tokio::spawn(async move {
        loop {
            sleep(interval).await;
            if s.load(Ordering::Relaxed) {
                return;
            }

            let reparented = {
                #[cfg(unix)]
                {
                    std::os::unix::process::parent_id() != parent_pid
                }
                #[cfg(not(unix))]
                {
                    false
                }
            };

            if reparented || !is_process_alive(parent_pid) {
                s.store(true, Ordering::Relaxed);
                on_parent_exit();
                return;
            }
        }
    });

    ParentMonitorHandle { stopped }
}

// ---------------------------------------------------------------------------
// Graceful cleanup
// ---------------------------------------------------------------------------

/// Options for [`run_cleanup`].
pub struct CleanupOptions<FCe, FSt, FSrv, FTr, FMo>
where
    FCe: FnOnce(),
    FSt: FnOnce(),
    FSrv: FnOnce() -> tokio::task::JoinHandle<()>,
    FTr: FnOnce() -> tokio::task::JoinHandle<()>,
    FMo: FnOnce(),
{
    pub cancel_embeddings: Option<FCe>,
    pub stop_tracker: FSt,
    pub close_server: FSrv,
    pub close_transport: FTr,
    pub stop_monitors: Option<FMo>,
}

/// Run graceful cleanup: cancel embeddings -> stop monitors -> stop tracker ->
/// close server + transport concurrently.
pub async fn run_cleanup<FCe, FSt, FSrv, FTr, FMo>(
    options: CleanupOptions<FCe, FSt, FSrv, FTr, FMo>,
) where
    FCe: FnOnce(),
    FSt: FnOnce(),
    FSrv: FnOnce() -> tokio::task::JoinHandle<()>,
    FTr: FnOnce() -> tokio::task::JoinHandle<()>,
    FMo: FnOnce(),
{
    if let Some(cancel) = options.cancel_embeddings {
        cancel();
    }
    if let Some(stop_mon) = options.stop_monitors {
        stop_mon();
    }
    (options.stop_tracker)();
    let server_handle = (options.close_server)();
    let transport_handle = (options.close_transport)();
    let _ = tokio::join!(server_handle, transport_handle);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    // -----------------------------------------------------------------------
    // Config parsing
    // -----------------------------------------------------------------------

    #[test]
    fn idle_shutdown_defaults() {
        assert_eq!(get_idle_shutdown_ms(None), DEFAULT_IDLE_TIMEOUT_MS);
        assert_eq!(get_idle_shutdown_ms(Some("")), DEFAULT_IDLE_TIMEOUT_MS);
    }

    #[test]
    fn idle_shutdown_disabled_values() {
        for v in &["0", "false", "off", "disabled", "none", " OFF ", " False "] {
            assert_eq!(get_idle_shutdown_ms(Some(v)), 0, "should disable for {v:?}");
        }
    }

    #[test]
    fn idle_shutdown_clamps_to_min() {
        assert_eq!(get_idle_shutdown_ms(Some("1000")), MIN_IDLE_TIMEOUT_MS);
        assert_eq!(get_idle_shutdown_ms(Some("60000")), MIN_IDLE_TIMEOUT_MS);
    }

    #[test]
    fn idle_shutdown_accepts_valid_value() {
        assert_eq!(get_idle_shutdown_ms(Some("120000")), 120_000);
        assert_eq!(get_idle_shutdown_ms(Some("900000")), 900_000);
    }

    #[test]
    fn idle_shutdown_invalid_string_uses_default() {
        assert_eq!(get_idle_shutdown_ms(Some("abc")), DEFAULT_IDLE_TIMEOUT_MS);
    }

    #[test]
    fn parent_poll_defaults() {
        assert_eq!(get_parent_poll_ms(None), DEFAULT_PARENT_POLL_MS);
    }

    #[test]
    fn parent_poll_clamps_to_min() {
        assert_eq!(get_parent_poll_ms(Some("500")), MIN_PARENT_POLL_MS);
    }

    #[test]
    fn parent_poll_accepts_valid() {
        assert_eq!(get_parent_poll_ms(Some("3000")), 3000);
    }

    // -----------------------------------------------------------------------
    // Broken pipe detection
    // -----------------------------------------------------------------------

    #[test]
    fn broken_pipe_detected() {
        let err = io::Error::new(io::ErrorKind::BrokenPipe, "broken");
        assert!(is_broken_pipe_error(&err));
    }

    #[test]
    fn connection_reset_detected() {
        let err = io::Error::new(io::ErrorKind::ConnectionReset, "reset");
        assert!(is_broken_pipe_error(&err));
    }

    #[test]
    fn other_error_not_broken_pipe() {
        let err = io::Error::new(io::ErrorKind::NotFound, "not found");
        assert!(!is_broken_pipe_error(&err));
    }

    // -----------------------------------------------------------------------
    // Process alive check
    // -----------------------------------------------------------------------

    #[cfg(unix)]
    #[test]
    fn pid_zero_not_alive() {
        assert!(!is_process_alive(0));
    }

    #[cfg(unix)]
    #[test]
    fn current_process_is_alive() {
        assert!(is_process_alive(std::process::id()));
    }

    #[cfg(unix)]
    #[test]
    fn bogus_pid_not_alive() {
        // Very high PID -- almost certainly not running.
        assert!(!is_process_alive(4_000_000));
    }

    // -----------------------------------------------------------------------
    // Idle monitor
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn idle_monitor_fires_after_timeout() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let _monitor = create_idle_monitor(50, move || {
            c.fetch_add(1, Ordering::SeqCst);
        });

        // Wait longer than timeout
        sleep(Duration::from_millis(120)).await;
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn idle_monitor_resets_on_touch() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let monitor = create_idle_monitor(80, move || {
            c.fetch_add(1, Ordering::SeqCst);
        });

        // Touch before timeout
        sleep(Duration::from_millis(50)).await;
        monitor.touch();
        sleep(Duration::from_millis(50)).await;
        // Should NOT have fired yet (80ms from last touch)
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        // Now wait for it to fire
        sleep(Duration::from_millis(80)).await;
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn idle_monitor_stop_prevents_firing() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let monitor = create_idle_monitor(50, move || {
            c.fetch_add(1, Ordering::SeqCst);
        });

        monitor.stop();
        sleep(Duration::from_millis(120)).await;
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn idle_monitor_zero_timeout_is_noop() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let monitor = create_idle_monitor(0, move || {
            c.fetch_add(1, Ordering::SeqCst);
        });

        sleep(Duration::from_millis(50)).await;
        assert_eq!(counter.load(Ordering::SeqCst), 0);
        // touch/stop should not panic
        monitor.touch();
        monitor.stop();
    }

    // -----------------------------------------------------------------------
    // Parent monitor
    // -----------------------------------------------------------------------

    #[cfg(unix)]
    #[tokio::test]
    async fn parent_monitor_detects_dead_parent() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        // Use a very high PID that shouldn't exist.
        // Poll interval is clamped to MIN_PARENT_POLL_MS (1000ms),
        // so we must wait longer than that.
        let _handle = start_parent_monitor(4_000_000, 1000, move || {
            c.fetch_add(1, Ordering::SeqCst);
        });

        // Wait long enough for at least one poll cycle (1000ms) plus overhead
        sleep(Duration::from_millis(1500)).await;
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn parent_monitor_pid_1_is_noop() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let handle = start_parent_monitor(1, 50, move || {
            c.fetch_add(1, Ordering::SeqCst);
        });

        sleep(Duration::from_millis(100)).await;
        assert_eq!(counter.load(Ordering::SeqCst), 0);
        assert!(handle.stopped.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn parent_monitor_self_pid_is_noop() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let handle = start_parent_monitor(std::process::id(), 50, move || {
            c.fetch_add(1, Ordering::SeqCst);
        });

        sleep(Duration::from_millis(100)).await;
        assert_eq!(counter.load(Ordering::SeqCst), 0);
        assert!(handle.stopped.load(Ordering::Relaxed));
    }
}
