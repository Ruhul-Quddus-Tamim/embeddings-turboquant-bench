//! Per-query LUT scoring and optional `f32x4` accumulation over dimensions.

use rayon::prelude::*;

use crate::packing::unpack_code;
use wide::f32x4;

#[inline]
pub fn score_one_row_decoded_scalar(
    codes_row: &[u8],
    dim: usize,
    lut: &[f32],
    num_levels: usize,
    db_norm: f32,
    query_norm: f32,
    scale_by_norms: bool,
) -> f32 {
    debug_assert_eq!(codes_row.len(), dim);
    let mut ip_rot = 0f32;
    for d in 0..dim {
        let code = codes_row[d] as usize;
        debug_assert!(code < num_levels);
        ip_rot += lut[d * num_levels + code];
    }
    combine_ip(ip_rot, db_norm, query_norm, scale_by_norms)
}

#[inline]
pub fn score_one_row_decoded_wide4(
    codes_row: &[u8],
    dim: usize,
    lut: &[f32],
    num_levels: usize,
    db_norm: f32,
    query_norm: f32,
    scale_by_norms: bool,
) -> f32 {
    debug_assert_eq!(codes_row.len(), dim);
    let mut d = 0usize;
    let mut acc = f32x4::ZERO;
    while d + 4 <= dim {
        let c0 = codes_row[d] as usize;
        let c1 = codes_row[d + 1] as usize;
        let c2 = codes_row[d + 2] as usize;
        let c3 = codes_row[d + 3] as usize;
        acc += f32x4::from([
            lut[d * num_levels + c0],
            lut[(d + 1) * num_levels + c1],
            lut[(d + 2) * num_levels + c2],
            lut[(d + 3) * num_levels + c3],
        ]);
        d += 4;
    }
    let arr: [f32; 4] = acc.into();
    let mut ip_rot = arr[0] + arr[1] + arr[2] + arr[3];
    while d < dim {
        let c = codes_row[d] as usize;
        ip_rot += lut[d * num_levels + c];
        d += 1;
    }
    combine_ip(ip_rot, db_norm, query_norm, scale_by_norms)
}

fn rayon_db_scan_parallel() -> bool {
    !matches!(std::env::var("RAYON_NUM_THREADS"), Ok(ref s) if s == "1")
}

#[allow(clippy::too_many_arguments)]
pub fn scores_for_query_decoded(
    decoded_flat: &[u8],
    norms_db: &[f32],
    dim: usize,
    n: usize,
    lut: &[f32],
    num_levels: usize,
    query_norm: f32,
    scale_by_norms: bool,
    scorer: Scorer,
) -> Vec<f32> {
    debug_assert_eq!(decoded_flat.len(), n * dim);
    let row_fn = |i| {
        let row = &decoded_flat[i * dim..(i + 1) * dim];
        match scorer {
            Scorer::Scalar => score_one_row_decoded_scalar(
                row,
                dim,
                lut,
                num_levels,
                norms_db[i],
                query_norm,
                scale_by_norms,
            ),
            Scorer::Wide => score_one_row_decoded_wide4(
                row,
                dim,
                lut,
                num_levels,
                norms_db[i],
                query_norm,
                scale_by_norms,
            ),
        }
    };
    if rayon_db_scan_parallel() {
        (0..n).into_par_iter().map(row_fn).collect()
    } else {
        (0..n).map(row_fn).collect()
    }
}

/// **`lut[d * num_levels + k] == q_rot[d] * centroid[k]`**.
pub fn flatten_query_lut(q_rot: &[f32], centroids: &[f32], dim: usize) -> Vec<f32> {
    let num_levels = centroids.len();
    assert_eq!(q_rot.len(), dim);
    let mut lut = vec![0f32; dim * num_levels];
    for d in 0..dim {
        let qb = q_rot[d];
        let base = d * num_levels;
        for (k, c) in centroids.iter().enumerate() {
            lut[base + k] = qb * *c;
        }
    }
    lut
}

#[inline(always)]
pub fn combine_ip(ip_rot: f32, db_norm: f32, query_norm: f32, scale_by_norms: bool) -> f32 {
    if scale_by_norms {
        db_norm * query_norm * ip_rot
    } else {
        ip_rot
    }
}

#[inline]
pub fn score_one_row_scalar(
    packed_row: &[u8],
    dim: usize,
    bits: u8,
    lut: &[f32],
    num_levels: usize,
    db_norm: f32,
    query_norm: f32,
    scale_by_norms: bool,
) -> f32 {
    let mut ip_rot = 0f32;
    for d in 0..dim {
        let code = unpack_code(packed_row, dim, bits, d) as usize;
        debug_assert!(code < num_levels);
        ip_rot += lut[d * num_levels + code];
    }
    combine_ip(ip_rot, db_norm, query_norm, scale_by_norms)
}

#[inline]
pub fn score_one_row_wide4(
    packed_row: &[u8],
    dim: usize,
    bits: u8,
    lut: &[f32],
    num_levels: usize,
    db_norm: f32,
    query_norm: f32,
    scale_by_norms: bool,
) -> f32 {
    let mut d = 0usize;
    let mut acc = f32x4::ZERO;
    while d + 4 <= dim {
        let c0 = unpack_code(packed_row, dim, bits, d) as usize;
        let c1 = unpack_code(packed_row, dim, bits, d + 1) as usize;
        let c2 = unpack_code(packed_row, dim, bits, d + 2) as usize;
        let c3 = unpack_code(packed_row, dim, bits, d + 3) as usize;
        acc += f32x4::from([
            lut[d * num_levels + c0],
            lut[(d + 1) * num_levels + c1],
            lut[(d + 2) * num_levels + c2],
            lut[(d + 3) * num_levels + c3],
        ]);
        d += 4;
    }
    let arr: [f32; 4] = acc.into();
    let mut ip_rot = arr[0] + arr[1] + arr[2] + arr[3];
    while d < dim {
        let c = unpack_code(packed_row, dim, bits, d) as usize;
        ip_rot += lut[d * num_levels + c];
        d += 1;
    }
    combine_ip(ip_rot, db_norm, query_norm, scale_by_norms)
}

#[derive(Clone, Copy, Debug)]
pub enum Scorer {
    Scalar,
    Wide,
}

#[allow(clippy::too_many_arguments)]
pub fn scores_for_query(
    packed: &[u8],
    norms_db: &[f32],
    dim: usize,
    bits: u8,
    n: usize,
    row_bytes: usize,
    lut: &[f32],
    num_levels: usize,
    query_norm: f32,
    scale_by_norms: bool,
    scorer: Scorer,
) -> Vec<f32> {
    debug_assert!(packed.len() >= n * row_bytes);
    let row_fn = |i| {
        let off = i * row_bytes;
        let row = &packed[off..off + row_bytes];
        match scorer {
            Scorer::Scalar => score_one_row_scalar(
                row,
                dim,
                bits,
                lut,
                num_levels,
                norms_db[i],
                query_norm,
                scale_by_norms,
            ),
            Scorer::Wide => score_one_row_wide4(
                row,
                dim,
                bits,
                lut,
                num_levels,
                norms_db[i],
                query_norm,
                scale_by_norms,
            ),
        }
    };
    if rayon_db_scan_parallel() {
        (0..n).into_par_iter().map(row_fn).collect()
    } else {
        (0..n).map(row_fn).collect()
    }
}
