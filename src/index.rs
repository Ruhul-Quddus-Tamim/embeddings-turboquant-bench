use ndarray::{Array1, ArrayView2, Axis};

use crate::lloyd_max::{build_gaussian_lloyd_max, GaussianQuantizer};
use crate::packing::{pack_row, packed_row_bytes};
use crate::qjl::{encode_qjl_signs, apply_gaussian_matvec, random_gaussian_matrix, QJL_INVERSE_NUMERATOR};
use crate::rotation::{apply_rotation, random_orthogonal_matrix};
use crate::simd::{flatten_query_lut, scores_for_query, scores_for_query_prod, Scorer};

#[derive(Debug, Clone)]
pub struct TurboQuantIndex {
    pub(crate) dim: usize,
    /// Total target bits per coordinate: `b` for [`TurboQuantIndex::new`], same `b` for [`TurboQuantIndex::new_prod`]
    /// (MSE stage uses `b - 1` plus one QJL bit per coordinate).
    pub(crate) bits: u8,
    pub(crate) seed: u64,
    lloyd_iterations: usize,
    rotation_q: ndarray::Array2<f32>,
    /// `None` — MSE-only [`TurboQuantIndex::new`]. `Some(S)` — inner-product [`TurboQuantIndex::new_prod`].
    s_matrix: Option<ndarray::Array2<f32>>,
    scalar_quant: GaussianQuantizer,
    norms: Vec<f32>,
    packed: Vec<u8>,
    row_packed_bytes: usize,
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
            s_matrix: None,
            scalar_quant,
            norms: Vec::new(),
            packed: Vec::new(),
            row_packed_bytes,
        }
    }

    /// TurboQuant **Q_prod** (Algorithm 2): `(b−1)`-bit MSE stage plus **QJL** on the residual, for **b** bits/coordinate overall (`b ≥ 2`).
    pub fn new_prod(dim: usize, bits: u8, seed: u64) -> Self {
        Self::new_prod_with_lloyd_iterations(dim, bits, seed, 80)
    }

    pub fn new_prod_with_lloyd_iterations(
        dim: usize,
        bits: u8,
        seed: u64,
        lloyd_iterations: usize,
    ) -> Self {
        assert!(dim > 1);
        assert!(
            (2..=8).contains(&bits),
            "Q_prod requires bits in 2..=8 (need b-1 ≥ 1 for the MSE stage)"
        );
        let sigma = (1.0f32 / dim as f32).sqrt();
        let mse_bits = bits - 1;
        let scalar_quant = build_gaussian_lloyd_max(mse_bits, sigma, lloyd_iterations);
        let rotation_q = random_orthogonal_matrix(dim, seed);
        let s_matrix = random_gaussian_matrix(dim, seed);
        let row_packed_bytes =
            packed_row_bytes(dim, mse_bits) + packed_row_bytes(dim, 1) + 4;
        Self {
            dim,
            bits,
            seed,
            lloyd_iterations,
            rotation_q,
            s_matrix: Some(s_matrix),
            scalar_quant,
            norms: Vec::new(),
            packed: Vec::new(),
            row_packed_bytes,
        }
    }

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

    /// Random Gaussian matrix **S** for QJL (`Q_prod` only).
    #[inline]
    pub fn s_matrix(&self) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.s_matrix.as_ref().map(|s| s.view())
    }

    #[inline]
    pub fn is_prod_quantizer(&self) -> bool {
        self.s_matrix.is_some()
    }

    /// Bits used by the MSE scalar stage (equals [`Self::bits`] for MSE mode, or `bits - 1` for `Q_prod`).
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
        s_matrix: Option<ndarray::Array2<f32>>,
        scalar_quant: GaussianQuantizer,
        norms: Vec<f32>,
        packed: Vec<u8>,
    ) -> Self {
        let row_packed_bytes = if s_matrix.is_some() {
            packed_row_bytes(dim, scalar_quant.bits) + packed_row_bytes(dim, 1) + 4
        } else {
            assert_eq!(
                scalar_quant.bits, bits,
                "MSE snapshot: quantizer bits must match index bits"
            );
            packed_row_bytes(dim, bits)
        };
        if let Some(ref s) = s_matrix {
            assert_eq!(s.nrows(), dim);
            assert_eq!(s.ncols(), dim);
            assert_eq!(
                scalar_quant.bits,
                bits - 1,
                "Q_prod snapshot: MSE stage must use bits-1"
            );
        }
        assert_eq!(
            norms.len() * row_packed_bytes,
            packed.len(),
            "packed length mismatch"
        );
        Self {
            dim,
            bits,
            seed,
            lloyd_iterations,
            rotation_q,
            s_matrix,
            scalar_quant,
            norms,
            packed,
            row_packed_bytes,
        }
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
            let (norm, slab) = if let Some(ref s) = self.s_matrix {
                encode_row_prod(slice, self.dim, self.bits, q, s.view(), &self.scalar_quant)
            } else {
                encode_row(slice, self.dim, self.bits, q, &self.scalar_quant)
            };
            self.norms.push(norm);
            self.packed.extend_from_slice(&slab);
        }
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

        let centroids_slice = self.centroids();
        let nl = centroids_slice.len();
        let dim = self.dim;
        let mse_bits = self.scalar_quant.bits;
        let scale_by_norms = cfg.objective == SearchObjective::InnerProduct;
        let rotation = self.rotation_q.view();
        let qjl_scale = QJL_INVERSE_NUMERATOR / dim as f32;

        queries
            .axis_iter(Axis(0))
            .map(|q_view| {
                let q_row = q_view.as_slice().expect("query row contiguous");
                let (qnorm, q_rot_vec) = rotated_unit_parts(q_row, dim, rotation);
                let lut = flatten_query_lut(&q_rot_vec, centroids_slice, dim);

                let scores = if let Some(ref s) = self.s_matrix {
                    let unit_q =
                        Array1::from_shape_fn(dim, |i| q_row[i] / qnorm.max(f32::MIN_POSITIVE));
                    let s_q = apply_gaussian_matvec(s.view(), unit_q.view());
                    let s_q_sl = s_q.as_slice().unwrap();
                    let mse_b = packed_row_bytes(dim, mse_bits);
                    let qjl_b = packed_row_bytes(dim, 1);
                    scores_for_query_prod(
                        &self.packed,
                        &self.norms,
                        dim,
                        mse_bits,
                        mse_b,
                        qjl_b,
                        self.row_packed_bytes,
                        &lut,
                        nl,
                        s_q_sl,
                        qjl_scale,
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
) -> (f32, Vec<u8>) {
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
    (norm, buf)
}

#[inline]
fn encode_row_prod(
    vec: &[f32],
    dim: usize,
    bits_total: u8,
    rotation_mat: ndarray::ArrayView2<f32>,
    s: ndarray::ArrayView2<f32>,
    quant: &GaussianQuantizer,
) -> (f32, Vec<u8>) {
    assert_eq!(quant.bits, bits_total - 1);
    let mse_bits = quant.bits;
    let mut ssum = 0f32;
    for &x in vec {
        ssum += x * x;
    }
    let norm = ssum.sqrt().max(f32::MIN_POSITIVE);
    let unit = Array1::from_shape_fn(dim, |i| vec[i] / norm);
    let rotated = apply_rotation(rotation_mat, unit.view());

    let mut codes = vec![0u8; dim];
    for (i, &z) in rotated.iter().enumerate() {
        codes[i] = quant.encode(z);
    }

    let y_rec = Array1::from_shape_fn(dim, |i| quant.reconstruct(codes[i]));
    let u_mse = apply_rotation(rotation_mat.t(), y_rec.view());
    let r = &unit - &u_mse;
    let mut gsq = 0f32;
    for &x in r.iter() {
        gsq += x * x;
    }
    let gamma = gsq.sqrt();

    let mse_row_b = packed_row_bytes(dim, mse_bits);
    let qjl_row_b = packed_row_bytes(dim, 1);
    let total_b = mse_row_b + qjl_row_b + 4;
    let mut buf = vec![0u8; total_b];
    pack_row(&codes, dim, mse_bits, &mut buf[..mse_row_b]);

    let mut qjl_codes = vec![0u8; dim];
    encode_qjl_signs(s, r.view(), &mut qjl_codes);
    pack_row(&qjl_codes, dim, 1, &mut buf[mse_row_b..mse_row_b + qjl_row_b]);

    buf[mse_row_b + qjl_row_b..].copy_from_slice(&gamma.to_le_bytes());
    (norm, buf)
}

fn topk_argmax_indices(scores: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    order.truncate(k);
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qjl::apply_gaussian_matvec;
    use ndarray::{Array1, ArrayView2, Axis};
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
    fn simd_wide_matches_scalar_scores() {
        let dim = 24;
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
    fn prod_simd_wide_matches_scalar_scores() {
        let dim = 24;
        let bits = 4u8;
        let mut rng = StdRng::seed_from_u64(505);
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
            TurboQuantIndex::new_prod_with_lloyd_iterations(dim, bits, 77, 64);
        idx.add_vectors(db.view());

        let q = db.index_axis(Axis(0), 0).to_owned();
        let qrow = q.as_slice().unwrap();
        let (qnorm, q_rot) = rotated_unit_parts(qrow, dim, idx.rotation());

        let lut = crate::simd::flatten_query_lut(&q_rot, idx.centroids(), dim);
        let mse_b = packed_row_bytes(dim, idx.mse_bits_per_coordinate());
        let qjl_b = packed_row_bytes(dim, 1);
        let s = idx.s_matrix().unwrap();
        let unit_q = Array1::from_shape_fn(dim, |i| qrow[i] / qnorm.max(f32::MIN_POSITIVE));
        let s_q = apply_gaussian_matvec(s, unit_q.view());
        let s_q_sl = s_q.as_slice().unwrap();
        let qjl_scale = crate::qjl::QJL_INVERSE_NUMERATOR / dim as f32;

        let s_scalar = crate::simd::scores_for_query_prod(
            idx.packed_codes(),
            idx.norms(),
            dim,
            idx.mse_bits_per_coordinate(),
            mse_b,
            qjl_b,
            idx.row_packed_bytes(),
            &lut,
            idx.num_quant_levels(),
            s_q_sl,
            qjl_scale,
            qnorm,
            false,
            Scorer::Scalar,
        );
        let s_wide = crate::simd::scores_for_query_prod(
            idx.packed_codes(),
            idx.norms(),
            dim,
            idx.mse_bits_per_coordinate(),
            mse_b,
            qjl_b,
            idx.row_packed_bytes(),
            &lut,
            idx.num_quant_levels(),
            s_q_sl,
            qjl_scale,
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
    fn prod_disk_roundtrip() {
        let dim = 20;
        let mut idx =
            TurboQuantIndex::new_prod_with_lloyd_iterations(dim, 3, 4242, 50);
        let mut rng = StdRng::seed_from_u64(2);
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
        path.push("turboquant_prod_roundtrip_test.tq");
        idx.write_disk(&path).unwrap();
        let loaded = TurboQuantIndex::read_disk(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert!(loaded.is_prod_quantizer());
        assert_eq!(idx.len(), loaded.len());
        assert_eq!(idx.packed_codes(), loaded.packed_codes());
        assert_eq!(idx.mse_bits_per_coordinate(), loaded.mse_bits_per_coordinate());
        assert_eq!(idx.bits(), loaded.bits());
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
