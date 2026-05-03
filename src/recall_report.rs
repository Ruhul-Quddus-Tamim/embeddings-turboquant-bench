//! Shared brute-force vs TurboQuant recall reporting for benchmark binaries.

use ndarray::{ArrayView2, Axis};
use std::time::Instant;

use crate::index::{SearchConfig, SearchObjective};
use crate::simd::{flatten_query_lut, scores_for_query, Scorer};
use crate::TurboQuantIndex;

pub fn cosine_row(db: ArrayView2<f32>, row: usize, q: &[f32]) -> f32 {
    let r = db.index_axis(Axis(0), row);
    let s = r.as_slice().unwrap();
    s.iter().zip(q.iter()).map(|(&a, &b)| a * b).sum()
}

pub fn brute_topk_cosine(db: ArrayView2<f32>, q: &[f32], k: usize) -> Vec<usize> {
    let mut scores: Vec<(usize, f32)> =
        (0..db.shape()[0]).map(|i| (i, cosine_row(db, i, q))).collect();
    scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    scores.into_iter().take(k).map(|(i, _)| i).collect()
}

pub fn recall_at_k(gt: &[usize], approx: &[usize]) -> f32 {
    let mut hits = 0usize;
    for &i in approx {
        if gt.contains(&i) {
            hits += 1;
        }
    }
    hits as f32 / gt.len().max(1) as f32
}

fn report_recall_inner(
    idx: &TurboQuantIndex,
    db: ArrayView2<f32>,
    queries: ArrayView2<f32>,
    k: usize,
    dataset_label: &str,
) {
    let dim = idx.dim();
    let bits = idx.bits();
    let n_db = idx.len();
    let n_queries = queries.nrows();
    let index_seed = idx.seed();
    let lloyd_iters = idx.lloyd_iterations();

    let cfg_wide = SearchConfig {
        objective: SearchObjective::Cosine,
        scorer: Scorer::Wide,
    };

    let t_brute = Instant::now();
    let mut gts: Vec<Vec<usize>> = Vec::with_capacity(n_queries);
    for qi in 0..n_queries {
        let qrow_own = queries.index_axis(Axis(0), qi).into_owned();
        let qrow = qrow_own.as_slice().unwrap();
        gts.push(brute_topk_cosine(db, qrow, k));
    }
    let brute_ms = t_brute.elapsed().as_secs_f64() * 1000.0;

    let t_turbo = Instant::now();
    let approx = idx.search_topk_indices(queries.view(), k, cfg_wide);
    let turbo_ms = t_turbo.elapsed().as_secs_f64() * 1000.0;

    let eps_ms = 1e-6_f64;
    let speedup_wide = brute_ms / turbo_ms.max(eps_ms);

    let mut avg = 0f32;
    for (qi, aq) in approx.iter().enumerate() {
        avg += recall_at_k(&gts[qi], aq);
    }
    avg /= n_queries as f32;

    let q0_own = queries.index_axis(Axis(0), 0).into_owned();
    let q0 = q0_own.as_slice().unwrap();
    let qnorm_unit = q0
        .iter()
        .map(|&x| x * x)
        .sum::<f32>()
        .sqrt()
        .max(f32::MIN_POSITIVE);
    let max_diff = {
        let q_unit = ndarray::Array1::from_shape_fn(dim, |i| q0[i] / qnorm_unit);
        let q_rot_vec = crate::rotation::apply_rotation(idx.rotation(), q_unit.view());
        let q_rot = q_rot_vec.as_slice().unwrap().to_vec();
        let lut = flatten_query_lut(&q_rot, idx.centroids(), dim);

        let s_scalar = scores_for_query(
            idx.packed_codes(),
            idx.norms(),
            dim,
            bits,
            n_db,
            idx.row_packed_bytes(),
            &lut,
            idx.num_quant_levels(),
            qnorm_unit,
            false,
            Scorer::Scalar,
        );
        let s_wide = scores_for_query(
            idx.packed_codes(),
            idx.norms(),
            dim,
            bits,
            n_db,
            idx.row_packed_bytes(),
            &lut,
            idx.num_quant_levels(),
            qnorm_unit,
            false,
            Scorer::Wide,
        );
        s_scalar
            .iter()
            .zip(s_wide.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
    };

    eprintln!("dataset={dataset_label}");
    eprintln!(
        "dim={dim} bits={bits} n_db={n_db} n_queries={n_queries} k={k} index_seed={index_seed} lloyd_iters={lloyd_iters}",
    );
    eprintln!(
        "search_batch_ms: brute_dense_cosine(top-k all queries)={brute_ms:.4} turboquant_wide(top-k all queries)={turbo_ms:.4}",
    );
    eprintln!(
        "speedup_wide_vs_brute_dense_batch={speedup_wide:.4} (>1 ⇒ TurboQuant wide batch faster on this machine; often <1 for small n_db)",
    );
    eprintln!(
        "mean Recall@{} (overlap with exact cosine brute force): {:.4}",
        k, avg
    );
    eprintln!(
        "max_abs_diff(scalar SIMD vs wide SIMD one query scores): {:.6}",
        max_diff
    );
}

/// Build index from `db`, then print recall / timing report.
pub fn report_recall(
    dim: usize,
    bits: u8,
    n_db: usize,
    _n_queries: usize,
    k: usize,
    index_seed: u64,
    lloyd_iters: usize,
    db: ArrayView2<f32>,
    queries: ArrayView2<f32>,
    dataset_label: &str,
) {
    let mut idx = TurboQuantIndex::with_lloyd_iterations(dim, bits, index_seed, lloyd_iters);
    idx.add_vectors(db);
    debug_assert_eq!(idx.len(), n_db);
    report_recall_for_index(&idx, db, queries, k, dataset_label);
}

/// Recall / timing report using an existing index (vectors already added).
pub fn report_recall_for_index(
    idx: &TurboQuantIndex,
    db: ArrayView2<f32>,
    queries: ArrayView2<f32>,
    k: usize,
    dataset_label: &str,
) {
    report_recall_inner(idx, db, queries, k, dataset_label);
}
