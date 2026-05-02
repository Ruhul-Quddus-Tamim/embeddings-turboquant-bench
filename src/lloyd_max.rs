//! Lloyd–Max optimal scalar quantizer for N(0, σ²), used as high-d approximation
//! for rotated unit-sphere coordinates (marginal ~ N(0, 1/d)).
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

/// Centroids and cell boundaries for k = 2^bits levels on N(0, σ²).
#[derive(Clone, Debug)]
pub struct GaussianQuantizer {
    pub sigma: f32,
    pub boundaries: Vec<f32>,
    pub centroids: Vec<f32>,
    pub bits: u8,
}

fn normal_mean_truncated(a: f64, b: f64) -> f64 {
    let n = Normal::new(0.0, 1.0).unwrap();
    let phi_a = n.pdf(a);
    let phi_b = n.pdf(b);
    let cdf_b = n.cdf(b);
    let cdf_a = n.cdf(a);
    let den = cdf_b - cdf_a;
    if den < 1e-15 {
        return 0.5 * (a + b);
    }
    (phi_a - phi_b) / den
}

/// Build optimal MSE scalar quantizer for N(0, σ²) with `bits` per sample.
pub fn build_gaussian_lloyd_max(bits: u8, sigma: f32, iterations: usize) -> GaussianQuantizer {
    assert!(bits >= 1 && bits <= 8);
    let k = 1usize << bits;
    let n = Normal::new(0.0, 1.0).unwrap();

    // Initial boundaries: equal probability mass for N(0,1)
    let mut t: Vec<f64> = Vec::with_capacity(k - 1);
    for i in 1..k {
        let p = i as f64 / k as f64;
        t.push(n.inverse_cdf(p));
    }

    let mut c: Vec<f64> = vec![0.0; k];

    for _ in 0..iterations {
        // Centroids for each cell
        c[0] = normal_mean_truncated(f64::NEG_INFINITY, t[0]);
        for j in 1..k - 1 {
            c[j] = normal_mean_truncated(t[j - 1], t[j]);
        }
        c[k - 1] = normal_mean_truncated(t[k - 2], f64::INFINITY);

        // Boundaries as midpoints
        for j in 0..k - 1 {
            t[j] = 0.5 * (c[j] + c[j + 1]);
        }
    }

    let sigma_f = sigma as f64;
    GaussianQuantizer {
        sigma,
        boundaries: t.iter().map(|&x| (x * sigma_f) as f32).collect(),
        centroids: c.iter().map(|&x| (x * sigma_f) as f32).collect(),
        bits,
    }
}

impl GaussianQuantizer {
    pub fn encode(&self, x: f32) -> u8 {
        let k = 1usize << self.bits;
        let mut code: usize = 0;
        for (i, &b) in self.boundaries.iter().enumerate() {
            if x > b {
                code = i + 1;
            } else {
                break;
            }
        }
        assert!(code < k);
        code as u8
    }

    pub fn reconstruct(&self, code: u8) -> f32 {
        self.centroids[code as usize]
    }

    #[inline]
    pub fn num_levels(&self) -> usize {
        1usize << self.bits
    }

    /// Restore quantization tables persisted in [`crate::disk::write`] snapshots.
    pub fn from_quant_tables(
        bits: u8,
        sigma: f32,
        boundaries: Vec<f32>,
        centroids: Vec<f32>,
    ) -> Self {
        let k = 1usize << bits;
        assert_eq!(
            boundaries.len(),
            k.saturating_sub(1),
            "expected {} boundaries for {} bits",
            k.saturating_sub(1),
            bits
        );
        assert_eq!(centroids.len(), k);
        GaussianQuantizer {
            sigma,
            boundaries,
            centroids,
            bits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lloyd_centroids_ordered() {
        let q = build_gaussian_lloyd_max(2, (1.0_f32 / 256.0_f32).sqrt(), 64);
        for i in 1..q.centroids.len() {
            assert!(q.centroids[i - 1] < q.centroids[i], "{:?}", q.centroids);
        }
    }
}
