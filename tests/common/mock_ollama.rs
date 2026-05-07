//! Shared mock-Ollama harness for warmup throttling tests.
//!
//! Stands up a local HTTP server on `127.0.0.1:0` (OS-assigned port) using
//! `wiremock`. Every POST to `/api/embed` returns a deterministic 768-dim
//! embedding and atomically increments two counters:
//!
//! - `total_calls` — lifetime sum of all embedding inputs received.
//! - `peak_concurrent` — highest in-flight count observed at any instant.
//!
//! An optional `delay_ms` injects latency into each response so concurrency
//! windows are wide enough for the atomics to register overlapping requests.
//!
//! # Usage
//!
//! ```rust,ignore
//! let mock = MockOllamaServer::start(50 /* delay_ms */).await;
//! let host = mock.uri();          // "http://127.0.0.1:<port>"
//! // ... run your code that calls Ollama ...
//! assert_eq!(mock.total_calls(), 10);
//! assert!(mock.peak_concurrent() <= 2);
//! ```

use std::sync::{
    Arc,
    atomic::{AtomicI64, AtomicUsize, Ordering},
};

use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

/// Dimensionality of every synthetic embedding vector returned by the mock.
const MOCK_EMBED_DIMS: usize = 768;

/// Produce a deterministic 768-dim unit vector seeded by `seed`.
pub fn synthetic_embedding(seed: usize) -> Vec<f32> {
    let scale = 1.0 / (MOCK_EMBED_DIMS as f32).sqrt();
    (0..MOCK_EMBED_DIMS)
        .map(|i| ((seed + i) % 17) as f32 * scale * 0.1)
        .collect()
}

// ---------------------------------------------------------------------------
// Responder that tracks concurrency
// ---------------------------------------------------------------------------

struct TrackingResponder {
    total_calls: Arc<AtomicUsize>,
    in_flight: Arc<AtomicI64>,
    peak_concurrent: Arc<AtomicUsize>,
    delay_ms: u64,
}

impl Respond for TrackingResponder {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        // Count how many inputs are in this batch.
        let body: serde_json::Value =
            serde_json::from_slice(&req.body).unwrap_or(serde_json::Value::Null);
        let n_inputs = body
            .get("input")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(1);

        // Increment in-flight counter and record peak.
        let prev = self.in_flight.fetch_add(1, Ordering::AcqRel);
        let current = (prev + 1) as usize;
        // Update peak using a compare-and-swap loop.
        let mut old_peak = self.peak_concurrent.load(Ordering::Acquire);
        while old_peak < current {
            match self.peak_concurrent.compare_exchange(
                old_peak,
                current,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(actual) => old_peak = actual,
            }
        }

        // Accumulate total calls.
        self.total_calls.fetch_add(n_inputs, Ordering::Relaxed);

        // Inject optional delay (blocking — wiremock calls respond() in a sync context).
        if self.delay_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(self.delay_ms));
        }

        // Decrement in-flight.
        self.in_flight.fetch_sub(1, Ordering::AcqRel);

        // Build synthetic embeddings response.
        let embeddings: Vec<Vec<f32>> = (0..n_inputs).map(synthetic_embedding).collect();
        ResponseTemplate::new(200).set_body_json(serde_json::json!({ "embeddings": embeddings }))
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// A mock Ollama HTTP server with concurrency instrumentation.
pub struct MockOllamaServer {
    /// Underlying `wiremock` server — must stay alive while the mock is in use.
    _server: MockServer,
    /// Cumulative count of embedding *inputs* processed (not batch requests).
    total_calls: Arc<AtomicUsize>,
    /// Maximum observed concurrent in-flight batch calls.
    peak_concurrent: Arc<AtomicUsize>,
    /// Base URI for this server (e.g. `"http://127.0.0.1:51234"`).
    uri: String,
}

impl MockOllamaServer {
    /// Start a new mock server.
    ///
    /// `delay_ms` — per-response artificial latency in milliseconds. Use ≥ 50
    /// to make concurrent overlap measurable; use 0 for fastest throughput.
    pub async fn start(delay_ms: u64) -> Self {
        let server = MockServer::start().await;

        let total_calls = Arc::new(AtomicUsize::new(0));
        let in_flight = Arc::new(AtomicI64::new(0));
        let peak_concurrent = Arc::new(AtomicUsize::new(0));

        let responder = TrackingResponder {
            total_calls: Arc::clone(&total_calls),
            in_flight: Arc::clone(&in_flight),
            peak_concurrent: Arc::clone(&peak_concurrent),
            delay_ms,
        };

        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(responder)
            .mount(&server)
            .await;

        let uri = server.uri();

        Self {
            _server: server,
            total_calls,
            peak_concurrent,
            uri,
        }
    }

    /// URI of the mock server (pass as `OLLAMA_HOST` to the server under test).
    pub fn uri(&self) -> &str {
        &self.uri
    }

    /// Total number of embedding inputs processed since start (or last reset).
    pub fn total_calls(&self) -> usize {
        self.total_calls.load(Ordering::Acquire)
    }

    /// Peak number of simultaneous in-flight batch calls observed.
    pub fn peak_concurrent(&self) -> usize {
        self.peak_concurrent.load(Ordering::Acquire)
    }

    /// Reset both counters to zero (useful between test phases).
    #[allow(dead_code)]
    pub fn reset_counters(&self) {
        self.total_calls.store(0, Ordering::Release);
        self.peak_concurrent.store(0, Ordering::Release);
    }
}
