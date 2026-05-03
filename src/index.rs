#[cfg(target_arch = "aarch64")]
use std::cell::RefCell;
use std::sync::{Arc, Mutex, RwLock};

use ndarray::{ArrayView2, Axis};
use ndarray::linalg::general_mat_mul;

use crate::lloyd_max::{build_gaussian_lloyd_max, GaussianQuantizer};
use crate::packing::{pack_row, packed_row_bytes, unpack_all_rows};
use crate::rotation::{apply_rotation, random_orthogonal_matrix};
use crate::simd::{
    flatten_query_lut, scores_for_query, scores_for_query_decoded, Scorer,
};

/// Scratch buffers reused across searches: L2‑normalize queries and `queries_unit @ rotationᵀ` without
/// allocating new `nq×dim` arrays on every [`TurboQuantIndex::search_topk_indices`] call (latency-critical).
#[derive(Debug)]
pub(crate) struct SearchScratch {
    normed_queries: ndarray::Array2<f32>,
    q_rot: ndarray::Array2<f32>,
    query_norms: Vec<f32>,
    #[cfg(target_arch = "aarch64")]
    neon: RefCell<crate::search_blocked_neon::Aarch64NeonSearchReuse>,
}

impl SearchScratch {
    pub(crate) fn new() -> Self {
        Self {
            normed_queries: ndarray::Array2::<f32>::zeros((1usize, 1usize)),
            q_rot: ndarray::Array2::<f32>::zeros((1usize, 1usize)),
            query_norms: Vec::new(),
            #[cfg(target_arch = "aarch64")]
            neon: RefCell::new(crate::search_blocked_neon::Aarch64NeonSearchReuse::new()),
        }
    }

    #[inline]
    fn ensure_queries_shape(mat: &mut ndarray::Array2<f32>, nq: usize, dim: usize) {
        let (r, c) = (mat.nrows(), mat.ncols());
        if r != nq || c != dim {
            *mat = ndarray::Array2::<f32>::zeros((nq, dim));
        }
    }

    /// Copies `queries_row_major`, L2‑normalizes rows, writes **`normed @ rotationᵀ`** into **`q_rot`**,
    /// and fills **`query_norms`** (`rotationᵀ` is `rotation_q.t()`).
    #[inline]
    pub(crate) fn fill_queries_rotated(
        &mut self,
        queries_row_major: &[f32],
        nq: usize,
        dim: usize,
        rotation_t: ndarray::ArrayView2<'_, f32>,
    ) {
        debug_assert_eq!(queries_row_major.len(), nq * dim);

        Self::ensure_queries_shape(&mut self.normed_queries, nq, dim);
        Self::ensure_queries_shape(&mut self.q_rot, nq, dim);

        let unit = &mut self.normed_queries;
        unit.as_slice_mut().unwrap()[..nq * dim].copy_from_slice(queries_row_major);
        normalize_rows_inplace_collect_norms(unit, &mut self.query_norms);
        debug_assert_eq!(self.query_norms.len(), nq);
        general_mat_mul(1.0_f32, &unit.view(), &rotation_t, 0.0_f32, &mut self.q_rot.view_mut());
    }
}

/// TurboQuant MSE index: Gaussian-marginal Lloyd–Max (`bits` per coordinate), random orthogonal `Q`,
/// bit-packed codes, LUT search ([`crate::simd`]).
#[derive(Debug)]
pub struct TurboQuantIndex {
    pub(crate) dim: usize,
    /// Total bits per coordinate for MSE quantization (`b` in [`TurboQuantIndex::new`]).
    pub(crate) bits: u8,
    pub(crate) seed: u64,
    lloyd_iterations: usize,
    rotation_q: ndarray::Array2<f32>,
    scalar_quant: GaussianQuantizer,
    norms: Vec<f32>,
    packed: Vec<u8>,
    decoded_flat: Vec<u8>,
    row_packed_bytes: usize,
    /// Lazily built blocked-code layout for NEON nibble‑LUT search (4‑bit lanes); cleared on [`Self::add_vectors`].
    blocked_codes_neon: RwLock<Option<Arc<Vec<u8>>>>,
    /// Query rotation + aarch64 NEON scratch reused across [`Self::search_topk_indices`] (avoids per-call `nq×dim` allocations).
    search_workspace: Mutex<SearchScratch>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchObjective {
    InnerProduct,
    Cosine,
}

#[derive(Clone, Copy, Debug)]
pub struct SearchConfig {
    pub objective: SearchObjective,
    pub scorer: Scorer,
}

impl TurboQuantIndex {
    pub fn new(dim: usize, bits: u8, seed: u64) -> Self {
        Self::with_lloyd_iterations(dim, bits, seed, 80)
    }

    pub fn with_lloyd_iterations(dim: usize, bits: u8, seed: u64, lloyd_iterations: usize) -> Self {
        assert!(dim > 1);
        assert!((1..=8).contains(&bits));
        let sigma = (1.0f32 / dim as f32).sqrt();
        let scalar_quant = build_gaussian_lloyd_max(bits, sigma, lloyd_iterations);
        let rotation_q = random_orthogonal_matrix(dim, seed);
        let row_packed_bytes = packed_row_bytes(dim, bits);
        Self {
            dim,
            bits,
            seed,
            lloyd_iterations,
            rotation_q,
            scalar_quant,
            norms: Vec::new(),
            packed: Vec::new(),
            decoded_flat: Vec::new(),
            row_packed_bytes,
            blocked_codes_neon: RwLock::new(None),
            search_workspace: Mutex::new(SearchScratch::new()),
        }
    }

    /// Reserved for API compatibility; indexing does not require a separate warm-up step.
    #[inline]
    pub fn prepare(&self) {}

    #[inline]
    pub fn len(&self) -> usize {
        self.norms.len()
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub fn bits(&self) -> u8 {
        self.bits
    }

    #[inline]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    #[inline]
    pub fn rotation(&self) -> ndarray::ArrayView2<'_, f32> {
        self.rotation_q.view()
    }

    pub fn rotation_matrix(&self) -> ndarray::Array2<f32> {
        self.rotation_q.clone()
    }

    #[inline]
    pub fn mse_bits_per_coordinate(&self) -> u8 {
        self.scalar_quant.bits
    }

    #[inline]
    pub fn centroids(&self) -> &[f32] {
        &self.scalar_quant.centroids
    }

    #[inline]
    pub fn norms(&self) -> &[f32] {
        &self.norms
    }

    #[inline]
    pub fn packed_codes(&self) -> &[u8] {
        &self.packed
    }

    #[inline]
    pub fn row_packed_bytes(&self) -> usize {
        self.row_packed_bytes
    }

    #[inline]
    pub fn num_quant_levels(&self) -> usize {
        self.scalar_quant.num_levels()
    }

    #[inline]
    pub fn scalar_quant_tables(&self) -> &GaussianQuantizer {
        &self.scalar_quant
    }

    #[inline]
    pub(crate) fn lloyd_iterations(&self) -> usize {
        self.lloyd_iterations
    }

    pub(crate) fn from_loaded(
        dim: usize,
        bits: u8,
        seed: u64,
        lloyd_iterations: usize,
        rotation_q: ndarray::Array2<f32>,
        scalar_quant: GaussianQuantizer,
        norms: Vec<f32>,
        packed: Vec<u8>,
    ) -> Self {
        assert_eq!(
            scalar_quant.bits, bits,
            "MSE snapshot: quantizer bits must match index bits"
        );
        let row_packed_bytes = packed_row_bytes(dim, bits);
        assert_eq!(
            norms.len() * row_packed_bytes,
            packed.len(),
            "packed length mismatch"
        );
        let mut decoded_flat = Vec::new();
        unpack_all_rows(
            &packed,
            norms.len(),
            dim,
            bits,
            row_packed_bytes,
            &mut decoded_flat,
        );
        Self {
            dim,
            bits,
            seed,
            lloyd_iterations,
            rotation_q,
            scalar_quant,
            norms,
            packed,
            decoded_flat,
            row_packed_bytes,
            blocked_codes_neon: RwLock::new(None),
            search_workspace: Mutex::new(SearchScratch::new()),
        }
    }

    fn blocked_codes_neon_cache(&self) -> Arc<Vec<u8>> {
        let n = self.len();
        let dim = self.dim;
        debug_assert!(crate::search_blocked_neon::neon_mse_blocked_eligible(
            self.bits,
            dim,
            self.decoded_flat.len(),
            n
        ));
        {
            let guard = self.blocked_codes_neon.read().unwrap();
            if let Some(ref arc) = *guard {
                return arc.clone();
            }
        }
        let mut guard = self.blocked_codes_neon.write().unwrap();
        if let Some(ref arc) = *guard {
            return arc.clone();
        }
        let v = Arc::new(crate::search_blocked_neon::build_blocked_codes_from_decoded(
            &self.decoded_flat,
            n,
            dim,
            self.bits,
        ));
        *guard = Some(v.clone());
        v
    }

    /// Append row-major `[n, dim]` vectors (`f32`).
    pub fn add_vectors(&mut self, rows: ArrayView2<f32>) {
        assert_eq!(rows.ncols(), self.dim, "row width must equal index dimension");
        if rows.is_empty() {
            return;
        }
        let q = self.rotation_q.view();
        for row in rows.axis_iter(Axis(0)) {
            let slice = row.as_slice().expect("row-major contiguous rows");
            let (norm, slab, codes) =
                encode_row(slice, self.dim, self.bits, q, &self.scalar_quant);
            self.norms.push(norm);
            self.packed.extend_from_slice(&slab);
            self.decoded_flat.extend_from_slice(&codes);
        }
        *self.blocked_codes_neon.write().unwrap() = None;
    }

    /// Top‑`k` approximate neighbors per query (higher score is better).
    pub fn search_topk_indices(
        &self,
        queries: ArrayView2<f32>,
        k: usize,
        cfg: SearchConfig,
    ) -> Vec<Vec<usize>> {
        assert_eq!(queries.ncols(), self.dim);
        let db_n = self.len();
        assert!(k > 0, "k must be positive");
        assert!(db_n >= k, "k cannot exceed database size {db_n}");

        let centroids_slice = self.scalar_quant.centroids.as_slice();
        let nl = centroids_slice.len();
        let dim = self.dim;
        let mse_bits = self.scalar_quant.bits;
        let scale_by_norms = cfg.objective == SearchObjective::InnerProduct;
        let rotation = self.rotation_q.view();

        let nq = queries.nrows();
        let queries_row_major = queries
            .as_slice()
            .expect("search_topk_indices: queries must be contiguous row-major f32 slice");
        let mut ws = self
            .search_workspace
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        ws.fill_queries_rotated(queries_row_major, nq, dim, rotation.t());
        let q_rot_all = &ws.q_rot;
        let query_norms_batch = ws.query_norms.as_slice();

        #[cfg(target_arch = "aarch64")]
        {
            let use_neon = crate::search_blocked_neon::neon_mse_blocked_eligible(
                self.bits,
                dim,
                self.decoded_flat.len(),
                db_n,
            ) && matches!(cfg.scorer, Scorer::Wide);
            if use_neon {
                let blocked = self.blocked_codes_neon_cache();
                let codes_per_byte = (8 / self.bits) as usize;
                let nbg = dim / codes_per_byte;
                let bits_u = self.bits as usize;
                let mut out = Vec::with_capacity(nq);
                let mut neon_reuse = ws.neon.borrow_mut();
                let n = &mut *neon_reuse;
                if k < db_n {
                    n.topk4.ensure_k(k);
                    n.topk1.ensure_k(k);
                }
                let mut qi = 0usize;
                while qi < nq {
                    let batch = (nq - qi).min(4);
                    if batch == 4 {
                        let r0 = q_rot_lut_row(q_rot_all, qi);
                        let r1 = q_rot_lut_row(q_rot_all, qi + 1);
                        let r2 = q_rot_lut_row(q_rot_all, qi + 2);
                        let r3 = q_rot_lut_row(q_rot_all, qi + 3);
                        let (s0, b0) =
                            n.lut_scratch_batch[0].build(r0, centroids_slice, bits_u, dim);
                        let (s1, b1) =
                            n.lut_scratch_batch[1].build(r1, centroids_slice, bits_u, dim);
                        let (s2, b2) =
                            n.lut_scratch_batch[2].build(r2, centroids_slice, bits_u, dim);
                        let (s3, b3) =
                            n.lut_scratch_batch[3].build(r3, centroids_slice, bits_u, dim);
                        let lut_u8: [&[u8]; 4] = [
                            n.lut_scratch_batch[0].uint8_luts(),
                            n.lut_scratch_batch[1].uint8_luts(),
                            n.lut_scratch_batch[2].uint8_luts(),
                            n.lut_scratch_batch[3].uint8_luts(),
                        ];
                        let scales = [s0, s1, s2, s3];
                        let biases = [b0, b1, b2, b3];
                        if k >= db_n {
                            let mut scores4 = vec![f32::NEG_INFINITY; 4 * db_n];
                            crate::search_blocked_neon::scores_4x_4bit_blocked_neon(
                                blocked.as_slice(),
                                lut_u8,
                                scales,
                                biases,
                                nbg,
                                &self.norms,
                                db_n,
                                scale_by_norms,
                                &mut scores4,
                            );
                            for off in 0..4 {
                                let qnorm = query_norms_batch[qi + off];
                                let row = &mut scores4[off * db_n..(off + 1) * db_n];
                                if scale_by_norms {
                                    for s in row.iter_mut() {
                                        *s *= qnorm;
                                    }
                                }
                                out.push(topk_argmax_indices(row, k));
                            }
                        } else {
                            let qscale = [
                                if scale_by_norms {
                                    query_norms_batch[qi]
                                } else {
                                    1.0
                                },
                                if scale_by_norms {
                                    query_norms_batch[qi + 1]
                                } else {
                                    1.0
                                },
                                if scale_by_norms {
                                    query_norms_batch[qi + 2]
                                } else {
                                    1.0
                                },
                                if scale_by_norms {
                                    query_norms_batch[qi + 3]
                                } else {
                                    1.0
                                },
                            ];
                            let tops =
                                crate::search_blocked_neon::search_4x_4bit_blocked_neon_topk(
                                    blocked.as_slice(),
                                    lut_u8,
                                    scales,
                                    biases,
                                    nbg,
                                    &self.norms,
                                    db_n,
                                    scale_by_norms,
                                    scale_by_norms,
                                    qscale,
                                    k,
                                    &mut n.topk4,
                                );
                            out.extend(tops);
                        }
                        qi += 4;
                    } else {
                        for off in 0..batch {
                            let qii = qi + off;
                            let qnorm = query_norms_batch[qii];
                            let q_rot_row = q_rot_lut_row(q_rot_all, qii);
                            let (scale, bias) = n
                                .lut_scratch_tail
                                .build(q_rot_row, centroids_slice, bits_u, dim);
                            let lut_sl = n.lut_scratch_tail.uint8_luts();
                            if k >= db_n {
                                let mut scores = vec![f32::NEG_INFINITY; db_n];
                                crate::search_blocked_neon::scores_4bit_blocked_neon(
                                    blocked.as_slice(),
                                    lut_sl,
                                    scale,
                                    bias,
                                    nbg,
                                    &self.norms,
                                    db_n,
                                    scale_by_norms,
                                    &mut scores,
                                );
                                if scale_by_norms {
                                    for s in &mut scores {
                                        *s *= qnorm;
                                    }
                                }
                                out.push(topk_argmax_indices(&scores, k));
                            } else {
                                out.push(
                                    crate::search_blocked_neon::search_1x_4bit_blocked_neon_topk(
                                        blocked.as_slice(),
                                        lut_sl,
                                        scale,
                                        bias,
                                        nbg,
                                        &self.norms,
                                        db_n,
                                        scale_by_norms,
                                        scale_by_norms,
                                        qnorm,
                                        k,
                                        &mut n.topk1,
                                    ),
                                );
                            }
                        }
                        qi += batch;
                    }
                }
                return out;
            }
        }

        (0..nq)
            .map(|qi| {
                    let qnorm = query_norms_batch[qi];
                    let q_rot_slice = q_rot_lut_row(q_rot_all, qi);
                    let lut = flatten_query_lut(q_rot_slice, centroids_slice, dim);
                    let scores = if self.decoded_flat.len() == db_n * dim {
                        scores_for_query_decoded(
                            &self.decoded_flat,
                            &self.norms,
                            dim,
                            db_n,
                            &lut,
                            nl,
                            qnorm,
                            scale_by_norms,
                            cfg.scorer,
                        )
                    } else {
                        scores_for_query(
                            &self.packed,
                            &self.norms,
                            dim,
                            mse_bits,
                            db_n,
                            self.row_packed_bytes,
                            &lut,
                            nl,
                            qnorm,
                            scale_by_norms,
                            cfg.scorer,
                        )
                    };
                    topk_argmax_indices(&scores, k)
                })
                .collect()
    }

    pub fn write_disk(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        crate::disk::write(path.as_ref(), self)
    }

    pub fn read_disk(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        crate::disk::read(path.as_ref())
    }
}

#[cfg(test)]
#[inline]
fn rotated_unit_parts(
    vec: &[f32],
    dim: usize,
    rotation_mat: ndarray::ArrayView2<f32>,
) -> (f32, Vec<f32>) {
    assert_eq!(vec.len(), dim);
    let mut s = 0f32;
    for &x in vec {
        s += x * x;
    }
    let nrm = s.sqrt().max(f32::MIN_POSITIVE);
    let unit = ndarray::Array1::from_shape_fn(dim, |i| vec[i] / nrm);
    let r = apply_rotation(rotation_mat, unit.view());
    (nrm, r.to_vec())
}

#[inline]
fn encode_row(
    vec: &[f32],
    dim: usize,
    bits: u8,
    rotation_mat: ndarray::ArrayView2<f32>,
    quant: &GaussianQuantizer,
) -> (f32, Vec<u8>, Vec<u8>) {
    assert_eq!(vec.len(), dim);
    let mut s = 0f32;
    for &x in vec {
        s += x * x;
    }
    let norm = s.sqrt().max(f32::MIN_POSITIVE);
    let unit = ndarray::Array1::from_shape_fn(dim, |i| vec[i] / norm);
    let rotated = apply_rotation(rotation_mat, unit.view());

    let mut codes = vec![0u8; dim];
    for (i, &z) in rotated.iter().enumerate() {
        codes[i] = quant.encode(z);
    }

    let row_b = packed_row_bytes(dim, bits);
    let mut buf = vec![0u8; row_b];
    pack_row(&codes, dim, bits, &mut buf);
    (norm, buf, codes)
}

fn topk_argmax_indices(scores: &[f32], k: usize) -> Vec<usize> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    /// Max-heap entry: **largest** [`Ord`] = **smallest** score → `peek` is the worst of the top‑`k`.
    #[derive(Clone, Copy, Debug)]
    struct Key(f32, usize);
    impl PartialEq for Key {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0 && self.1 == other.1
        }
    }
    impl Eq for Key {}
    impl PartialOrd for Key {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Key {
        fn cmp(&self, other: &Self) -> Ordering {
            match self.0.total_cmp(&other.0) {
                Ordering::Equal => self.1.cmp(&other.1),
                o => o.reverse(),
            }
        }
    }

    let n = scores.len();
    if k >= n {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        return order;
    }

    let mut heap: BinaryHeap<Key> = BinaryHeap::with_capacity(k + 1);
    for (i, &s) in scores.iter().enumerate() {
        if heap.len() < k {
            heap.push(Key(s, i));
        } else if s > heap.peek().unwrap().0 {
            heap.pop();
            heap.push(Key(s, i));
        }
    }
    let mut best: Vec<(f32, usize)> = heap.into_iter().map(|Key(s, i)| (s, i)).collect();
    best.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    best.into_iter().map(|(_, i)| i).collect()
}

#[inline]
fn q_rot_lut_row(q_rot_rows: &ndarray::Array2<f32>, qi: usize) -> &[f32] {
    let dim = q_rot_rows.ncols();
    let flat = q_rot_rows
        .as_slice()
        .expect("contiguous row-major rotated query matrix");
    let start = qi * dim;
    &flat[start..start + dim]
}

/// L2‑normalize each row **in place** and record per‑row norms in `norms_out` (length set to row count).
fn normalize_rows_inplace_collect_norms(units: &mut ndarray::Array2<f32>, norms_out: &mut Vec<f32>) {
    norms_out.resize(units.nrows(), 0.0_f32);
    for (mut row, slot) in units.axis_iter_mut(Axis(0)).zip(norms_out.iter_mut()) {
        let s = row.iter().fold(0f32, |acc, &x| acc + x * x).sqrt();
        let denom = s.max(f32::MIN_POSITIVE);
        *slot = s;
        row.iter_mut().for_each(|x| *x /= denom);
    }
}

#[cfg(test)]
/// L2‑normalize each row **in place**; returns per‑row norms (parity tests).
fn normalize_rows_inplace_ret_norms(units: &mut ndarray::Array2<f32>) -> Vec<f32> {
    let mut norms = Vec::new();
    normalize_rows_inplace_collect_norms(units, &mut norms);
    norms
}

#[cfg(test)]
/// All rotated unit queries: **`queries_unit @ rotationᵀ`** → **`(nq, dim)` row‑major** (each row matches per‑row [`rotated_unit_parts`]; contiguous rows for NEON LUT + BLAS).
fn batch_rotated_unit_queries(
    queries: ArrayView2<f32>,
    rotation_mat: ndarray::ArrayView2<f32>,
) -> (ndarray::Array2<f32>, Vec<f32>) {
    assert_eq!(queries.ncols(), rotation_mat.nrows());
    let nq = queries.nrows();
    let dout = rotation_mat.ncols();
    let mut units = queries.to_owned();
    let norms = normalize_rows_inplace_ret_norms(&mut units);
    let mut q_rot_rows = ndarray::Array2::<f32>::zeros((nq, dout));
    general_mat_mul(1.0_f32, &units.view(), &rotation_mat.t(), 0.0_f32, &mut q_rot_rows.view_mut());
    (q_rot_rows, norms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayView2, Axis};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    fn l2_normalize_rows(a: &mut ndarray::Array2<f32>) {
        for mut row in a.axis_iter_mut(Axis(0)) {
            let s = row.iter().fold(0f32, |acc, &x| acc + x * x).sqrt();
            let denom = s.max(f32::MIN_POSITIVE);
            row.iter_mut().for_each(|x| *x /= denom);
        }
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

    fn recall_at_k(gt: &[usize], approx: &[usize]) -> f32 {
        let mut hits = 0usize;
        for &i in approx {
            if gt.contains(&i) {
                hits += 1;
            }
        }
        hits as f32 / gt.len().max(1) as f32
    }

    #[test]
    fn topk_heap_matches_full_sort() {
        let mut rng = StdRng::seed_from_u64(404);
        let dist = StandardNormal;
        let scores: Vec<f32> = (0..5000)
            .map(|_| {
                let v: f64 = dist.sample(&mut rng);
                v as f32
            })
            .collect();
        for k in [1usize, 3, 10, 64, 256] {
            let mut order: Vec<usize> = (0..scores.len()).collect();
            order.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]));
            order.truncate(k);
            let heap = topk_argmax_indices(&scores, k);
            let mut exp: Vec<f32> = order.iter().map(|&i| scores[i]).collect();
            let mut got: Vec<f32> = heap.iter().map(|&i| scores[i]).collect();
            exp.sort_by(|a, b| b.total_cmp(a));
            got.sort_by(|a, b| b.total_cmp(a));
            assert_eq!(exp, got, "k={k}");
        }
    }

    #[test]
    fn batch_rotation_matches_per_row() {
        use crate::rotation::random_orthogonal_matrix;
        let d = 32;
        let q =
            ndarray::Array2::from_shape_fn((5, d), |(i, j)| ((i * 17 + j) as f32) * 0.01);
        let rot = random_orthogonal_matrix(d, 9);
        let (batch_r, norms_b) = batch_rotated_unit_queries(q.view(), rot.view());
        for qi in 0..5 {
            let row = q.index_axis(Axis(0), qi);
            let (nref, rref) = rotated_unit_parts(row.as_slice().unwrap(), d, rot.view());
            assert!((norms_b[qi] - nref).abs() < 1e-5);
            for j in 0..d {
                let a = batch_r[(qi, j)];
                let b = rref[j];
                assert!((a - b).abs() < 1e-4, "qi={qi} j={j} {a} {b}");
            }
        }
    }

    #[test]
    fn simd_wide_matches_scalar_scores() {
        let dim = 20;
        let bits = 4u8;
        let mut rng = StdRng::seed_from_u64(303);
        let dist = StandardNormal;
        let mut raw: Vec<f32> = Vec::with_capacity(40 * dim);
        for _ in 0..(40 * dim) {
            let v: f64 = dist.sample(&mut rng);
            raw.push(v as f32);
        }
        let mut db =
            ndarray::Array2::from_shape_vec((40, dim), raw).unwrap();
        l2_normalize_rows(&mut db);

        let mut idx =
            TurboQuantIndex::with_lloyd_iterations(dim, bits, 77, 64);
        idx.add_vectors(db.view());

        let q = db.index_axis(Axis(0), 0).to_owned();
        let qrow = q.as_slice().unwrap();
        let (qnorm, q_rot) = rotated_unit_parts(qrow, dim, idx.rotation());

        let lut = crate::simd::flatten_query_lut(&q_rot, idx.centroids(), dim);

        let s_scalar = crate::simd::scores_for_query(
            idx.packed_codes(),
            idx.norms(),
            dim,
            bits,
            idx.len(),
            idx.row_packed_bytes(),
            &lut,
            idx.num_quant_levels(),
            qnorm,
            false,
            Scorer::Scalar,
        );
        let s_wide = crate::simd::scores_for_query(
            idx.packed_codes(),
            idx.norms(),
            dim,
            bits,
            idx.len(),
            idx.row_packed_bytes(),
            &lut,
            idx.num_quant_levels(),
            qnorm,
            false,
            Scorer::Wide,
        );
        let max_diff = s_scalar
            .iter()
            .zip(&s_wide)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_diff < 1e-4, "{max_diff}");
    }

    #[test]
    fn decoded_lut_scores_match_packed() {
        let dim = 20;
        let bits = 4u8;
        let mut rng = StdRng::seed_from_u64(404);
        let dist = StandardNormal;
        let mut raw: Vec<f32> = Vec::with_capacity(40 * dim);
        for _ in 0..(40 * dim) {
            let v: f64 = dist.sample(&mut rng);
            raw.push(v as f32);
        }
        let mut db =
            ndarray::Array2::from_shape_vec((40, dim), raw).unwrap();
        l2_normalize_rows(&mut db);

        let mut idx =
            TurboQuantIndex::with_lloyd_iterations(dim, bits, 77, 64);
        idx.add_vectors(db.view());

        let mut flat = Vec::new();
        unpack_all_rows(
            idx.packed_codes(),
            idx.len(),
            dim,
            bits,
            idx.row_packed_bytes(),
            &mut flat,
        );

        let q = db.index_axis(Axis(0), 0).to_owned();
        let qrow = q.as_slice().unwrap();
        let (qnorm, q_rot) = rotated_unit_parts(qrow, dim, idx.rotation());

        let lut = crate::simd::flatten_query_lut(&q_rot, idx.centroids(), dim);

        let s_packed = crate::simd::scores_for_query(
            idx.packed_codes(),
            idx.norms(),
            dim,
            bits,
            idx.len(),
            idx.row_packed_bytes(),
            &lut,
            idx.num_quant_levels(),
            qnorm,
            false,
            Scorer::Scalar,
        );
        let s_dec = crate::simd::scores_for_query_decoded(
            &flat,
            idx.norms(),
            dim,
            idx.len(),
            &lut,
            idx.num_quant_levels(),
            qnorm,
            false,
            Scorer::Scalar,
        );
        let max_diff = s_packed
            .iter()
            .zip(&s_dec)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_diff < 1e-4, "{max_diff}");
    }

    #[test]
    fn disk_roundtrip() {
        let dim = 20;
        let mut idx =
            TurboQuantIndex::with_lloyd_iterations(dim, 2, 9001, 50);
        let mut rng = StdRng::seed_from_u64(1);
        let dist = StandardNormal;
        let mut raw: Vec<f32> = Vec::with_capacity(8 * dim);
        for _ in 0..(8 * dim) {
            let v: f64 = dist.sample(&mut rng);
            raw.push(v as f32);
        }
        let mut db = ndarray::Array2::from_shape_vec((8, dim), raw).unwrap();
        l2_normalize_rows(&mut db);
        idx.add_vectors(db.view());

        let mut path = std::env::temp_dir();
        path.push("turboquant_roundtrip_test.tq");
        idx.write_disk(&path).unwrap();
        let loaded = TurboQuantIndex::read_disk(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(idx.len(), loaded.len());
        assert_eq!(idx.packed_codes(), loaded.packed_codes());
        assert_eq!(idx.scalar_quant_tables().centroids.len(), loaded.scalar_quant_tables().centroids.len());
    }

    #[test]
    fn recall_mean_above_random_for_synthetic_sphere() {
        let dim = 48;
        let bits = 4u8;
        let n_db = 800;
        let n_q = 64;
        let k = 8;
        let mut rng = StdRng::seed_from_u64(404);
        let dist = StandardNormal;
        let mut raw_db: Vec<f32> = Vec::with_capacity(n_db * dim);
        for _ in 0..(n_db * dim) {
            let v: f64 = dist.sample(&mut rng);
            raw_db.push(v as f32);
        }
        let mut db =
            ndarray::Array2::from_shape_vec((n_db, dim), raw_db).unwrap();
        l2_normalize_rows(&mut db);

        let mut raw_q: Vec<f32> = Vec::with_capacity(n_q * dim);
        for _ in 0..(n_q * dim) {
            let v: f64 = dist.sample(&mut rng);
            raw_q.push(v as f32);
        }
        let mut queries =
            ndarray::Array2::from_shape_vec((n_q, dim), raw_q).unwrap();
        l2_normalize_rows(&mut queries);

        let mut idx =
            TurboQuantIndex::with_lloyd_iterations(dim, bits, 808, 80);
        idx.add_vectors(db.view());

        let approx = idx.search_topk_indices(
            queries.view(),
            k,
            SearchConfig {
                objective: SearchObjective::Cosine,
                scorer: Scorer::Wide,
            },
        );
        let mut mean = 0f32;
        for (qi, aq) in approx.iter().enumerate() {
            let qrow_own = queries.index_axis(Axis(0), qi).into_owned();
            let qrow = qrow_own.as_slice().unwrap();
            let gt = brute_topk_cosine(db.view(), qrow, k);
            mean += recall_at_k(&gt, aq);
        }
        mean /= n_q as f32;
        assert!(mean > 0.12, "{mean}");
    }
}
