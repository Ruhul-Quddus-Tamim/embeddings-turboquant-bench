//! Quantized Johnson–Lindenstrauss (QJL) map and Gaussian sketch matrix `S`
//! for TurboQuant inner-product mode (`Q_prod`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal as NormalDist};

/// `sqrt(π/2)` — scales the QJL inverse map (Definition 1 in TurboQuant paper).
pub const QJL_INVERSE_NUMERATOR: f32 = 1.2533141_f32; // (FRAC_PI_2.sqrt())

/// i.i.d. `N(0,1)` matrix `S ∈ R^{d×d}` for QJL, seeded deterministically from `seed`.
pub fn random_gaussian_matrix(dim: usize, seed: u64) -> Array2<f32> {
    assert!(dim > 0);
    let mut rng = StdRng::seed_from_u64(deterministic_s_seed(dim as u64, seed));
    let normal = NormalDist::new(0.0f64, 1.0).unwrap();
    Array2::from_shape_fn((dim, dim), |_| normal.sample(&mut rng) as f32)
}

/// Distinct seed mix so `S` is independent of the orthogonal `Q` seed stream.
#[inline]
fn deterministic_s_seed(dim: u64, index_seed: u64) -> u64 {
    let mut h = index_seed ^ 0xD503629E720E78F1 ^ dim.wrapping_mul(0x9E3779B97F4A7C15);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h
}

#[inline]
pub fn apply_gaussian_matvec(s: ArrayView2<f32>, x: ArrayView1<f32>) -> Array1<f32> {
    s.dot(&x.insert_axis(Axis(1))).index_axis_move(Axis(1), 0)
}

/// `sign(S · r)` as 0/1 packed symbols: `0 → -1`, `1 → +1` (zero maps to `+1`).
pub fn encode_qjl_signs(s: ArrayView2<f32>, r: ArrayView1<f32>, codes_out: &mut [u8]) {
    let dim = r.len();
    assert_eq!(s.nrows(), dim);
    assert_eq!(s.ncols(), dim);
    assert_eq!(codes_out.len(), dim);
    let proj = apply_gaussian_matvec(s, r);
    for (i, &v) in proj.iter().enumerate() {
        codes_out[i] = if v >= 0.0 { 1 } else { 0 };
    }
}

#[inline]
pub fn sign_from_code(bit: u8) -> f32 {
    if bit == 0 {
        -1.0
    } else {
        1.0
    }
}
