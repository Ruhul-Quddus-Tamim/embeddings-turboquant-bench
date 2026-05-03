//! Median ms/query TurboTiming + optional FAISS PQ FastScan subprocess (`benchmarks/faiss_npy_time.py`).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use ndarray::ArrayView2;
use serde::Deserialize;

use crate::index::{SearchConfig, SearchObjective};
use crate::simd::Scorer;
use crate::TurboQuantIndex;

/// Median ms/query over `runs` timed batch searches (warmup + timed runs), `RAYON_NUM_THREADS=1` expected.
pub fn median_tq_ms_per_query(
    idx: &TurboQuantIndex,
    queries: ArrayView2<f32>,
    k: usize,
    n_query: usize,
    timed_runs: usize,
) -> f64 {
    let cfg = SearchConfig {
        objective: SearchObjective::Cosine,
        scorer: Scorer::Wide,
    };
    let _ = idx.search_topk_indices(queries, k, cfg);
    let mut samples = Vec::with_capacity(timed_runs);
    for _ in 0..timed_runs {
        let t0 = Instant::now();
        let _ = idx.search_topk_indices(queries, k, cfg);
        samples.push(t0.elapsed().as_secs_f64() * 1000.0 / n_query as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_i = (timed_runs / 2).min(timed_runs.saturating_sub(1));
    samples[median_i]
}

#[derive(Deserialize)]
pub struct FaissTimings {
    pub faiss_ms_per_query: f64,
    pub faiss_pq_backend: String,
}

pub fn faiss_time_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../benchmarks/faiss_npy_time.py")
}

pub fn try_faiss_median_ms(
    train: &Path,
    test: &Path,
    k: usize,
    bits: u8,
    runs: usize,
) -> Result<FaissTimings, String> {
    let script = faiss_time_script();
    if !script.is_file() {
        return Err(format!("FAISS helper missing: {}", script.display()));
    }
    let out = Command::new("python3.10")
        .arg(&script)
        .arg(train)
        .arg(test)
        .arg(k.to_string())
        .arg(bits.to_string())
        .arg(runs.to_string())
        .arg("st")
        .output()
        .map_err(|e| format!("python3.10: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!("faiss_npy_time.py failed: {stderr}"));
    }
    let line = String::from_utf8_lossy(&out.stdout);
    let trimmed = line.trim();
    serde_json::from_str::<FaissTimings>(trimmed)
        .map_err(|e| format!("parse faiss JSON: {e}: {trimmed}"))
}

pub fn eprint_if_turboquant_slower_than_faiss(tq_ms: f64, faiss_ms: f64) {
    if tq_ms > faiss_ms * 1.5 {
        eprintln!(
            "note: `tq_ms_per_query` uses the **same timing protocol** as `benchmarks/benchmark_speed_glove200.py` \
             (warmup + 5 runs, median ms/query, ST). It measures **`turboquant_index`** search latency; **`TurboQuantIndex`** \
             timings from Python wrappers can differ wall‑clock‑wise even when FAISS ms matches."
        );
    }
}
