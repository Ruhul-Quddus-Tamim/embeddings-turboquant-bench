//! Fixed random orthogonal map Q ∈ R^{d×d} shared by index (data-oblivious).
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use nalgebra::{DMatrix, QR};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal as NormalDist};

/// Generate Q from QR decomposition of a dense Gaussian matrix.
pub fn random_orthogonal_matrix(dim: usize, seed: u64) -> Array2<f32> {
    assert!(dim > 0);
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = NormalDist::new(0.0f64, 1.0).unwrap();
    let mut data = vec![0.0f32; dim * dim];
    for x in data.iter_mut() {
        *x = normal.sample(&mut rng) as f32;
    }
    let m = DMatrix::from_column_slice(dim, dim, &data);
    let qr = QR::new(m);
    let q = qr.q();
    Array2::from_shape_fn((dim, dim), |(r, c)| q[(r, c)])
}

#[inline]
pub fn apply_rotation(q: ArrayView2<f32>, x: ArrayView1<f32>) -> Array1<f32> {
    q.dot(&x.insert_axis(Axis(1)))
        .index_axis_move(Axis(1), 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q_orthogonal_approx() {
        let d = 32;
        let q = random_orthogonal_matrix(d, 42);
        let mut identity = Array2::<f32>::zeros((d, d));
        for i in 0..d {
            identity[(i, i)] = 1.0;
        }
        let qtq = q.t().dot(&q);
        let diff: f32 = (&qtq - &identity).mapv(|e| e * e).sum();
        assert!(diff < 1e-3, "{diff}");
    }

    #[test]
    fn unit_vector_stays_normalized() {
        let d = 64;
        let q = random_orthogonal_matrix(d, 7);
        let mut x = Array1::zeros(d);
        x[0] = 1.0;
        let y = apply_rotation(q.view(), x.view());
        let n = y.fold(0f32, |a, &b| a + b * b).sqrt();
        assert!((n - 1.0).abs() < 1e-4);
    }
}
