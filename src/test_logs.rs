use std::cell::RefCell;
use std::io::Write;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock};

type Logs = Arc<Mutex<Vec<u8>>>;

thread_local! {
    static CAPTURE: RefCell<Option<Logs>> = const { RefCell::new(None) };
}

pub(crate) struct CaptureGuard {
    previous: Option<Logs>,
    _same_thread: PhantomData<Rc<()>>,
}

impl Drop for CaptureGuard {
    fn drop(&mut self) {
        CAPTURE.with(|capture| capture.replace(self.previous.take()));
    }
}

struct CapturedWriter(Option<Logs>);

impl Write for CapturedWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if let Some(logs) = &self.0 {
            logs.lock().unwrap().extend_from_slice(buf);
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

// Captures belong to synchronous tests or current-thread Tokio runtimes. Keep
// interest global and stable even when a callsite is first used on another thread.
pub(crate) fn captured_info_logs() -> (Logs, CaptureGuard) {
    static SUBSCRIBER: OnceLock<()> = OnceLock::new();
    SUBSCRIBER.get_or_init(|| {
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_ansi(false)
            .without_time()
            .with_target(false)
            .with_writer(|| CapturedWriter(CAPTURE.with(|capture| capture.borrow().clone())))
            .finish();
        tracing::subscriber::set_global_default(subscriber).unwrap();
    });
    let logs = Arc::new(Mutex::new(Vec::new()));
    let previous = CAPTURE.with(|capture| capture.replace(Some(Arc::clone(&logs))));
    (
        logs,
        CaptureGuard {
            previous,
            _same_thread: PhantomData,
        },
    )
}

pub(crate) fn logs_as_string(logs: &Logs) -> String {
    String::from_utf8(logs.lock().unwrap().clone()).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_callsite_first_used_on_another_thread() {
        const CHILD: &str = "CONTEXTPLUS_LOG_CAPTURE_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "test_logs::tests::captures_callsite_first_used_on_another_thread",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        fn emit() {
            tracing::info!("capture regression event");
        }
        let (logs, _guard) = captured_info_logs();
        std::thread::spawn(emit).join().unwrap();
        assert!(
            tracing::level_filters::LevelFilter::current()
                >= tracing::level_filters::LevelFilter::INFO
        );
        emit();
        assert_eq!(
            logs_as_string(&logs)
                .matches("capture regression event")
                .count(),
            1
        );
        std::thread::spawn(tracing::callsite::rebuild_interest_cache)
            .join()
            .unwrap();
        assert!(
            tracing::level_filters::LevelFilter::current()
                >= tracing::level_filters::LevelFilter::INFO
        );
        emit();
        assert_eq!(
            logs_as_string(&logs)
                .matches("capture regression event")
                .count(),
            2
        );
    }

    #[test]
    fn parallel_captures_are_isolated_and_restore_nested_buffers() {
        let barrier = std::sync::Barrier::new(16);
        std::thread::scope(|scope| {
            for id in 0..16 {
                let barrier = &barrier;
                scope.spawn(move || {
                    let (outer, _outer_guard) = captured_info_logs();
                    barrier.wait();
                    for _ in 0..32 {
                        let (inner, guard) = captured_info_logs();
                        tracing::info!(id, "inner event");
                        drop(guard);
                        let inner = logs_as_string(&inner);
                        assert_eq!(inner.lines().count(), 1);
                        assert!(inner.contains(&format!("id={id}")));
                        tracing::info!(id, "outer event");
                    }
                    let outer = logs_as_string(&outer);
                    assert_eq!(outer.lines().count(), 32);
                    assert!(!outer.contains("inner event"));
                    assert!(
                        outer
                            .lines()
                            .all(|line| line.ends_with(&format!("id={id}")))
                    );
                });
            }
        });
    }
}
