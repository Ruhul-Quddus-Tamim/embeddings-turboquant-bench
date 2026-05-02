//! Per-query LUT scoring and optional `f32x4` accumulation over dimensions.

use rayon::prelude::*;

use crate::packing::unpack_code;
use wide::f32x4;

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

/// Dot `w · qjl` with `qjl` stored as 1 bit per dim (`0 → -1`, `1 → +1`).
#[inline]
pub fn qjl_sign_dot_packed(w: &[f32], qjl_packed: &[u8], dim: usize) -> f32 {
    debug_assert_eq!(w.len(), dim);
    let mut acc = 0f32;
    let mut d = 0usize;
    while d + 4 <= dim {
        acc += w[d] * qjl_bit_sign(unpack_code(qjl_packed, dim, 1, d));
        acc += w[d + 1] * qjl_bit_sign(unpack_code(qjl_packed, dim, 1, d + 1));
        acc += w[d + 2] * qjl_bit_sign(unpack_code(qjl_packed, dim, 1, d + 2));
        acc += w[d + 3] * qjl_bit_sign(unpack_code(qjl_packed, dim, 1, d + 3));
        d += 4;
    }
    while d < dim {
        acc += w[d] * qjl_bit_sign(unpack_code(qjl_packed, dim, 1, d));
        d += 1;
    }
    acc
}

#[inline(always)]
fn qjl_bit_sign(bit: u8) -> f32 {
    if bit == 0 {
        -1.0
    } else {
        1.0
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

#[inline]
pub fn score_one_row_prod_scalar(
    row: &[u8],
    dim: usize,
    mse_bits: u8,
    mse_row_bytes: usize,
    qjl_row_bytes: usize,
    lut: &[f32],
    num_levels: usize,
    s_query: &[f32],
    qjl_scale: f32,
    db_norm: f32,
    query_norm: f32,
    scale_by_norms: bool,
) -> f32 {
    let mse_packed = &row[..mse_row_bytes];
    let qjl_packed = &row[mse_row_bytes..mse_row_bytes + qjl_row_bytes];
    let gamma = f32::from_le_bytes(
        row[mse_row_bytes + qjl_row_bytes..mse_row_bytes + qjl_row_bytes + 4]
            .try_into()
            .unwrap(),
    );
    let ip_mse = score_one_row_scalar(
        mse_packed,
        dim,
        mse_bits,
        lut,
        num_levels,
        1.0,
        1.0,
        false,
    );
    let ip_unit = ip_mse + qjl_scale * gamma * qjl_sign_dot_packed(s_query, qjl_packed, dim);
    combine_ip(ip_unit, db_norm, query_norm, scale_by_norms)
}

#[inline]
pub fn score_one_row_prod_wide4(
    row: &[u8],
    dim: usize,
    mse_bits: u8,
    mse_row_bytes: usize,
    qjl_row_bytes: usize,
    lut: &[f32],
    num_levels: usize,
    s_query: &[f32],
    qjl_scale: f32,
    db_norm: f32,
    query_norm: f32,
    scale_by_norms: bool,
) -> f32 {
    let mse_packed = &row[..mse_row_bytes];
    let qjl_packed = &row[mse_row_bytes..mse_row_bytes + qjl_row_bytes];
    let gamma = f32::from_le_bytes(
        row[mse_row_bytes + qjl_row_bytes..mse_row_bytes + qjl_row_bytes + 4]
            .try_into()
            .unwrap(),
    );
    let ip_mse = score_one_row_wide4(
        mse_packed,
        dim,
        mse_bits,
        lut,
        num_levels,
        1.0,
        1.0,
        false,
    );
    let ip_unit = ip_mse + qjl_scale * gamma * qjl_sign_dot_packed(s_query, qjl_packed, dim);
    combine_ip(ip_unit, db_norm, query_norm, scale_by_norms)
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
    (0..n)
        .into_par_iter()
        .map(|i| {
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
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn scores_for_query_prod(
    packed: &[u8],
    norms_db: &[f32],
    dim: usize,
    mse_bits: u8,
    mse_row_bytes: usize,
    qjl_row_bytes: usize,
    row_bytes: usize,
    lut: &[f32],
    num_levels: usize,
    s_query: &[f32],
    qjl_scale: f32,
    query_norm: f32,
    scale_by_norms: bool,
    scorer: Scorer,
) -> Vec<f32> {
    debug_assert!(packed.len() >= norms_db.len() * row_bytes);
    (0..norms_db.len())
        .into_par_iter()
        .map(|i| {
            let off = i * row_bytes;
            let row = &packed[off..off + row_bytes];
            match scorer {
                Scorer::Scalar => score_one_row_prod_scalar(
                    row,
                    dim,
                    mse_bits,
                    mse_row_bytes,
                    qjl_row_bytes,
                    lut,
                    num_levels,
                    s_query,
                    qjl_scale,
                    norms_db[i],
                    query_norm,
                    scale_by_norms,
                ),
                Scorer::Wide => score_one_row_prod_wide4(
                    row,
                    dim,
                    mse_bits,
                    mse_row_bytes,
                    qjl_row_bytes,
                    lut,
                    num_levels,
                    s_query,
                    qjl_scale,
                    norms_db[i],
                    query_norm,
                    scale_by_norms,
                ),
            }
        })
        .collect()
}
