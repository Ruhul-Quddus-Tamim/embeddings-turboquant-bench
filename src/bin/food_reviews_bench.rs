//! Bench: Fine Food embeddings CSV — compression, disk size, Recall@k (bits 2 & 4), timings.
//!
//! Stderr ( **`bits=4`** path, same protocol as [`glove_angular_bench`]): **`tq_ms_per_query`**, optional **`faiss_ms_per_query`**, then exact-vs-TurboQuant recall block. **`--no-faiss`** skips Python FAISS. JSON on stdout is unchanged.

use std::path::PathBuf;

use ndarray::{Array2, ArrayView2, Axis};
use ndarray_npy::write_npy;
use serde::Serialize;

use turboquant_index::csv_dataset::{self, FINE_FOOD_EMBEDDING_DIM};
use turboquant_index::index::{SearchConfig, SearchObjective};
use turboquant_index::packing::packed_row_bytes;
use turboquant_index::recall_report::report_recall_for_index;
use turboquant_index::latency::{
    eprint_if_turboquant_slower_than_faiss, median_tq_ms_per_query, try_faiss_median_ms,
};
use turboquant_index::simd::Scorer;
use turboquant_index::TurboQuantIndex;

const DIM: usize = FINE_FOOD_EMBEDDING_DIM;

#[derive(Clone, Serialize)]
struct SplitInfo {
    train_fraction: f64,
    split_seed: u64,
    n_total: usize,
    n_db: usize,
    n_query: usize,
}

#[derive(Clone, Serialize)]
struct BytesBreakdown {
    packed_codes: u64,
    norms: u64,
    rotation_q: u64,
    scalar_quant_boundaries: u64,
    scalar_quant_centroids: u64,
    index_structure_total: u64,
}

#[derive(Clone, Serialize)]
struct ReferenceSearchBench {
    /// Exact cosine top‑`k`: full linear scan over **dense fp32 DB** (`n_db × dim` dot-products per query).
    brute_dense_cosine_topk_batch_ms: f64,
}

/// Same protocol as [`glove_angular_bench`] / `benchmarks/faiss_npy_time.py`: warmup + median over `timed_runs` batch searches, `RAYON_NUM_THREADS=1`, bits=4 TurboQuant vs FAISS PQ FastScan.
#[derive(Clone, Serialize)]
struct ReferenceLatencyMs {
    timed_runs: usize,
    tq_ms_per_query: f64,
    faiss_ms_per_query: Option<f64>,
    faiss_pq_backend: Option<String>,
    /// Set when FAISS was not run (`--no-faiss`) or the subprocess failed (see message).
    faiss_skipped_reason: Option<String>,
}

#[derive(Clone, Serialize)]
struct TimingsMs {
    build_index: f64,
    search_wide_batch: f64,
    search_scalar_batch: f64,
}

#[derive(Clone, Serialize)]
struct RecallStats {
    mean: f32,
    std_sample: f32,
    min: f32,
}

#[derive(Clone, Serialize)]
struct BitConfigReport {
    bits: u8,
    index_seed: u64,
    lloyd_iterations: usize,
    theoretical_packed_row_bytes: usize,
    actual_packed_row_bytes_observed: usize,
    bytes: BytesBreakdown,
    bytes_dense_db_vectors_only: u64,
    bytes_quantized_corpus_payload: u64,
    compression_ratio_dense_over_quantized_corpus: f64,
    compression_ratio_dense_db_vs_index_structure: f64,
    disk_snapshot_bytes_tempfile: Option<u64>,
    extrapolated_dense_vectors_bytes_n10m: u64,
    extrapolated_quantized_corpus_payload_bytes_n10m: u64,
    extrapolated_index_structure_bytes_n10m: u64,
    timings_ms: TimingsMs,
    /// `reference_search.brute_* / timings_ms.search_*` (>1 → TurboQuant path faster wall-clock for this workload).
    speedup_wide_vs_brute_dense_batch: f64,
    speedup_scalar_vs_brute_dense_batch: f64,
    recall_at_k: RecallStats,
    scorer_wide: &'static str,
}

#[derive(Serialize)]
struct FoodBenchReport {
    dataset_path: String,
    dim: usize,
    k: usize,
    index_lloyd_iterations: usize,
    split: SplitInfo,
    notes: &'static str,
    reference_search: ReferenceSearchBench,
    /// TurboQuant vs FAISS median ms/query (also echoed on stderr).
    reference_latency_ms_per_query: ReferenceLatencyMs,
    configs: Vec<BitConfigReport>,
}

fn arg_string(args: &[String], i: usize, default: &str) -> String {
    args.get(i).cloned().unwrap_or_else(|| default.to_string())
}

fn arg_usize(args: &[String], i: usize, default: usize) -> usize {
    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn arg_u64(args: &[String], i: usize, default: u64) -> u64 {
    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn arg_f64(args: &[String], i: usize, default: f64) -> f64 {
    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn cosine_row(db: ArrayView2<f32>, row: usize, q: &[f32]) -> f32 {
    let r = db.index_axis(Axis(0), row);
    let s = r.as_slice().unwrap();
    s.iter().zip(q.iter()).map(|(&a, &b)| a * b).sum()
}

fn brute_topk_cosine(db: ArrayView2<f32>, q: &[f32], k: usize) -> Vec<usize> {
    let mut scores: Vec<(usize, f32)> =
        (0..db.shape()[0]).map(|i| (i, cosine_row(db, i, q))).collect();
    scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    scores.into_iter().take(k).map(|(i, _)| i).collect()
}

/// Wall time to run exact brute cosine top‑`k` for **all** query rows (baseline for search latency).
fn benchmark_brute_dense_batch(db: &Array2<f32>, queries: &Array2<f32>, k: usize) -> f64 {
    let t = std::time::Instant::now();
    for qi in 0..queries.nrows() {
        let qrow_own = queries.index_axis(Axis(0), qi).into_owned();
        let qslice = qrow_own.as_slice().unwrap();
        let _ = brute_topk_cosine(db.view(), qslice, k);
    }
    t.elapsed().as_secs_f64() * 1000.0
}

fn recall_overlap(gt: &[usize], approx: &[usize]) -> f32 {
    let mut hits = 0usize;
    for &i in approx {
        if gt.contains(&i) {
            hits += 1;
        }
    }
    hits as f32 / gt.len().max(1) as f32
}

fn recall_stats(samples: &[f32]) -> (f32, f32, f32) {
    if samples.is_empty() {
        return (0., 0., 0.);
    }
    let n = samples.len() as f32;
    let mean = samples.iter().sum::<f32>() / n;
    let min = samples.iter().copied().fold(f32::INFINITY, f32::min);
    let std_sample = if samples.len() < 2 {
        0f32
    } else {
        let var: f32 = samples
            .iter()
            .map(|x| {
                let d = *x - mean;
                d * d
            })
            .sum::<f32>()
            / (samples.len() - 1) as f32;
        var.sqrt()
    };
    (mean, std_sample, min)
}

fn bytes_breakdown(idx: &TurboQuantIndex) -> BytesBreakdown {
    let sq = idx.scalar_quant_tables();
    let packed = idx.packed_codes().len() as u64;
    let norms = (idx.norms().len() * std::mem::size_of::<f32>()) as u64;
    let rq = idx.rotation().len() * std::mem::size_of::<f32>();
    let rq = rq as u64;
    let b = (sq.boundaries.len() * std::mem::size_of::<f32>()) as u64;
    let c = (sq.centroids.len() * std::mem::size_of::<f32>()) as u64;
    BytesBreakdown {
        packed_codes: packed,
        norms,
        rotation_q: rq,
        scalar_quant_boundaries: b,
        scalar_quant_centroids: c,
        index_structure_total: packed + norms + rq + b + c,
    }
}

fn disk_snapshot_size(idx: &TurboQuantIndex) -> Result<u64, std::io::Error> {
    let mut tmp = std::env::temp_dir();
    tmp.push(format!(
        "turboquant_food_bench_{}.tq",
        std::process::id()
    ));
    idx.write_disk(&tmp)?;
    let sz = std::fs::metadata(&tmp)?.len();
    std::fs::remove_file(tmp).ok();
    Ok(sz)
}

fn extrapolate_dense_bytes(d: usize, n: usize) -> u64 {
    (n * d * std::mem::size_of::<f32>()) as u64
}

fn extrapolate_index_structure_bytes(dim: usize, n: usize, bits: u8) -> u64 {
    let n_u = n as u64;
    let row = packed_row_bytes(dim, bits) as u64;
    let k = (1usize << bits) as u64;
    let bsz = std::mem::size_of::<f32>() as u64;
    let boundaries = k.saturating_sub(1).saturating_mul(bsz);
    let centroids = k.saturating_mul(bsz);
    let q = (dim as u64)
        .saturating_mul(dim as u64)
        .saturating_mul(bsz);
    n_u.saturating_mul(row)
        .saturating_add(n_u.saturating_mul(bsz))
        .saturating_add(q)
        .saturating_add(boundaries)
        .saturating_add(centroids)
}

fn run_bits_config(
    bits: u8,
    index_seed: u64,
    lloyd_iters: usize,
    db: &Array2<f32>,
    queries: &Array2<f32>,
    k: usize,
    bytes_dense_db: u64,
    brute_dense_batch_ms: f64,
) -> Result<(BitConfigReport, TurboQuantIndex), Box<dyn std::error::Error>> {
    let n_db = db.nrows();
    let dim = DIM;

    let t_build = std::time::Instant::now();
    let mut idx =
        TurboQuantIndex::with_lloyd_iterations(dim, bits, index_seed, lloyd_iters);
    idx.add_vectors(db.view());
    let build_ms = t_build.elapsed().as_secs_f64() * 1000.0;

    let row_theory = packed_row_bytes(dim, bits);
    let row_observed = idx.packed_codes().len() / n_db;
    debug_assert_eq!(row_observed, row_theory);

    let bytes = bytes_breakdown(&idx);
    let disk_bytes = disk_snapshot_size(&idx).ok();

    let extrap_n = 10_000_000_usize;
    let extrap_n_u = extrap_n as u64;
    let bsz_u = std::mem::size_of::<f32>() as u64;
    let extrap_corpus_quant = extrap_n_u
        .saturating_mul(row_theory as u64)
        .saturating_add(extrap_n_u.saturating_mul(bsz_u));

    let mut recalls = Vec::with_capacity(queries.nrows());
    let cfg_wide = SearchConfig {
        objective: SearchObjective::Cosine,
        scorer: Scorer::Wide,
    };

    let t_search_w = std::time::Instant::now();
    let approx = idx.search_topk_indices(queries.view(), k, cfg_wide);
    let wide_ms = t_search_w.elapsed().as_secs_f64() * 1000.0;

    let cfg_scalar = SearchConfig {
        objective: SearchObjective::Cosine,
        scorer: Scorer::Scalar,
    };
    let t_search_s = std::time::Instant::now();
    let _approx_s = idx.search_topk_indices(queries.view(), k, cfg_scalar);
    let scalar_ms = t_search_s.elapsed().as_secs_f64() * 1000.0;

    for (qi, aq) in approx.iter().enumerate() {
        let qrow_own = queries.index_axis(Axis(0), qi).into_owned();
        let qrow = qrow_own.as_slice().unwrap();
        let gt = brute_topk_cosine(db.view(), qrow, k);
        recalls.push(recall_overlap(&gt, aq));
    }

    let (mean, std_sample, min) = recall_stats(&recalls);

    let ratio =
        bytes_dense_db as f64 / bytes.index_structure_total.max(1) as f64;
    let corpus_payload = bytes.packed_codes + bytes.norms;
    let ratio_corpus = bytes_dense_db as f64 / corpus_payload.max(1) as f64;

    let eps_ms = 1e-6_f64;
    let speedup_wide = brute_dense_batch_ms / wide_ms.max(eps_ms);
    let speedup_scalar = brute_dense_batch_ms / scalar_ms.max(eps_ms);

    let report = BitConfigReport {
        bits,
        index_seed,
        lloyd_iterations: lloyd_iters,
        theoretical_packed_row_bytes: row_theory,
        actual_packed_row_bytes_observed: row_observed,
        bytes,
        bytes_dense_db_vectors_only: bytes_dense_db,
        bytes_quantized_corpus_payload: corpus_payload,
        compression_ratio_dense_over_quantized_corpus: ratio_corpus,
        compression_ratio_dense_db_vs_index_structure: ratio,
        disk_snapshot_bytes_tempfile: disk_bytes,
        extrapolated_dense_vectors_bytes_n10m: extrapolate_dense_bytes(dim, extrap_n),
        extrapolated_quantized_corpus_payload_bytes_n10m: extrap_corpus_quant,
        extrapolated_index_structure_bytes_n10m: extrapolate_index_structure_bytes(
            dim, extrap_n, bits,
        ),
        timings_ms: TimingsMs {
            build_index: build_ms,
            search_wide_batch: wide_ms,
            search_scalar_batch: scalar_ms,
        },
        speedup_wide_vs_brute_dense_batch: speedup_wide,
        speedup_scalar_vs_brute_dense_batch: speedup_scalar,
        recall_at_k: RecallStats {
            mean,
            std_sample,
            min,
        },
        scorer_wide: "Wide",
    };
    Ok((report, idx))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("RAYON_NUM_THREADS", "1");

    let mut raw: Vec<String> = std::env::args().skip(1).collect();
    let skip_faiss = raw.iter().any(|s| s == "--no-faiss");
    raw.retain(|s| s != "--no-faiss");
    let positional: Vec<String> = raw.into_iter().filter(|s| !s.starts_with('-')).collect();

    let csv_path = PathBuf::from(arg_string(
        &positional,
        0,
        "fine_food_reviews_with_embeddings_1k.csv",
    ));
    let k = arg_usize(&positional, 1, 10);
    let train_fraction = arg_f64(&positional, 2, 0.85);
    let split_seed = arg_u64(&positional, 3, 42);
    let index_seed = arg_u64(&positional, 4, 99);
    let lloyd_iters = arg_usize(&positional, 5, 80);

    if k < 1 {
        return Err("k must be >= 1".into());
    }

    let mut embeddings = csv_dataset::load_embedding_csv(&csv_path, DIM)?;
    csv_dataset::l2_normalize_rows(&mut embeddings);

    let n = embeddings.nrows();
    let (db_ix, q_ix) = csv_dataset::split_train_query(n, train_fraction, split_seed);
    let n_db = db_ix.len();
    let n_q = q_ix.len();

    if k > n_db {
        return Err(format!(
            "k={k} exceeds database size n_db={n_db}; lower k or train_fraction"
        )
        .into());
    }

    const REFERENCE_N_DB: usize = 100_000;
    if n_db < REFERENCE_N_DB {
        eprintln!(
            "note: Fine Food split uses n_db={n_db} (GloVe reference speed scripts use ~{REFERENCE_N_DB} train rows); \
             FAISS vs TQ ordering here is for matched protocol, not the large-corpus regime."
        );
    }

    let db = csv_dataset::gather_rows(embeddings.view(), &db_ix);
    let queries = csv_dataset::gather_rows(embeddings.view(), &q_ix);
    drop(embeddings);

    let bytes_dense_db = extrapolate_dense_bytes(DIM, n_db);

    let brute_batch_ms = benchmark_brute_dense_batch(&db, &queries, k);

    let mut configs = Vec::new();
    let mut idx4_opt: Option<TurboQuantIndex> = None;
    for &bits in &[2u8, 4u8] {
        let (rep, idx) = run_bits_config(
            bits,
            index_seed,
            lloyd_iters,
            &db,
            &queries,
            k,
            bytes_dense_db,
            brute_batch_ms,
        )?;
        configs.push(rep);
        if bits == 4 {
            idx4_opt = Some(idx);
        }
    }

    let idx4 = idx4_opt.expect("4-bit config");
    let label = format!(
        "{} ({}×{} train, {}×{} query, angular, bits=4 latency below)",
        csv_path.display(),
        n_db,
        DIM,
        n_q,
        DIM,
    );

    let timed_runs = 5usize;
    let tq_ms = median_tq_ms_per_query(&idx4, queries.view(), k, n_q, timed_runs);

    let mut faiss_ms: Option<f64> = None;
    let mut faiss_backend: Option<String> = None;
    let mut faiss_skipped_reason: Option<String> = None;

    eprintln!(
        "speed_median_ms_per_query (warmup + {timed_runs} timed runs, RAYON_NUM_THREADS=1): tq_ms_per_query={tq_ms:.4}  (bits=4)",
    );
    if !skip_faiss {
        let pid = std::process::id();
        let train_npy = std::env::temp_dir().join(format!("turboquant_food_{pid}_train.npy"));
        let test_npy = std::env::temp_dir().join(format!("turboquant_food_{pid}_test.npy"));
        write_npy(&train_npy, &db)?;
        write_npy(&test_npy, &queries)?;
        match try_faiss_median_ms(&train_npy, &test_npy, k, 4, timed_runs) {
            Ok(f) => {
                faiss_ms = Some(f.faiss_ms_per_query);
                faiss_backend = Some(f.faiss_pq_backend.clone());
                eprintln!(
                    "faiss_ms_per_query={:.4}  backend={}  (same protocol as benchmarks/benchmark_speed_glove200.py)",
                    f.faiss_ms_per_query,
                    f.faiss_pq_backend,
                );
                eprint_if_turboquant_slower_than_faiss(tq_ms, f.faiss_ms_per_query);
            }
            Err(e) => {
                faiss_skipped_reason = Some(e.clone());
                eprintln!("faiss_ms_per_query=(skipped)  reason: {e}");
            }
        }
        let _ = std::fs::remove_file(&train_npy);
        let _ = std::fs::remove_file(&test_npy);
    } else {
        faiss_skipped_reason = Some("--no-faiss".to_string());
        eprintln!("faiss_ms_per_query=(skipped)  --no-faiss");
    }

    report_recall_for_index(&idx4, db.view(), queries.view(), k, &label);

    eprintln!(
        "=== latency (bits=4): tq_ms_per_query={tq_ms:.4}  faiss_ms_per_query={} ===",
        faiss_ms
            .map(|x| format!("{x:.4}"))
            .unwrap_or_else(|| "null".to_string())
    );

    let report = FoodBenchReport {
        dataset_path: csv_path.display().to_string(),
        dim: DIM,
        k,
        index_lloyd_iterations: lloyd_iters,
        split: SplitInfo {
            train_fraction,
            split_seed,
            n_total: n,
            n_db,
            n_query: n_q,
        },
        notes: "Mean/min/std of Recall@k vs brute-force cosine on dense f32 DB. \
                Memory: bytes_quantized_corpus_payload vs bytes_dense_db_vectors_only; full index includes rotation Q. \
                Search: reference_search.brute_dense_cosine_topk_batch_ms = dense fp32 linear scan for all queries (exact top-k). \
                speedup_wide_vs_brute_dense_batch = brute_ms / turboquant_wide_ms: > 1 means TurboQuant Wide batch is faster wall-clock; \
                < 1 means dense brute was faster on this run (common at modest n where float dot-products are very cheap vs unpack/LUT). \
                speedup_scalar_vs_brute_dense_batch is the same using Scalar batch timings. \
                Extrapolation uses n=10M for corpus vs index structure formulas. \
                reference_latency_ms_per_query matches glove_angular_bench / benchmarks/faiss_npy_time.py (median ms/query after warmup).",
        reference_search: ReferenceSearchBench {
            brute_dense_cosine_topk_batch_ms: brute_batch_ms,
        },
        reference_latency_ms_per_query: ReferenceLatencyMs {
            timed_runs,
            tq_ms_per_query: tq_ms,
            faiss_ms_per_query: faiss_ms,
            faiss_pq_backend: faiss_backend,
            faiss_skipped_reason: faiss_skipped_reason,
        },
        configs,
    };

    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
