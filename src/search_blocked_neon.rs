//! Blocked layout + NEON nibble LUT scoring for **2- and 4-bit** MSE codes.
//!
//! Decoded row-major symbols are packed **`codes_per_byte = 8 / bits`** per byte (blocked ARM lane layout),
//! then scored with paired nibble LUTs and a [`BLOCK`]‑wide (32 SIMD lanes) kernel over database rows.

pub(crate) const BLOCK: usize = 32;
const FLUSH_EVERY: usize = 256;

/// Owned nibble LUT + scale/bias for NEON scoring (convenience / tests; hot path uses [`NeonLutScratch`]).
#[allow(dead_code)]
pub(crate) struct QueryNeonLut {
    pub uint8_luts: Vec<u8>,
    pub scale: f32,
    pub bias: f32,
}

/// Reusable scratch for [`NeonLutScratch::build`] / neon search (avoids 3 Vec allocs per query).
#[derive(Debug)]
pub(crate) struct NeonLutScratch {
    uint8_luts: Vec<u8>,
    float_vals: Vec<f32>,
    mins: Vec<f32>,
}

impl NeonLutScratch {
    pub fn new() -> Self {
        Self {
            uint8_luts: Vec::new(),
            float_vals: Vec::new(),
            mins: Vec::new(),
        }
    }

    #[inline]
    pub fn uint8_luts(&self) -> &[u8] {
        &self.uint8_luts
    }

    /// Writes LUT bytes into `self`; returns `(scale, bias)` for kernels.
    pub fn build(&mut self, q_rot_row: &[f32], centroids: &[f32], bits: usize, dim: usize) -> (f32, f32) {
        let codes_per_byte = 8 / bits;
        let codes_per_nibble = codes_per_byte / 2;
        let n_byte_groups = dim / codes_per_byte;
        let code_mask = (1u16 << bits) - 1;
        let n_subs = n_byte_groups * 2;

        self.uint8_luts.resize(n_byte_groups * 32, 0);
        self.float_vals.resize(n_byte_groups * 32, 0.0);
        self.mins.resize(n_subs, 0.0);

        let uint8_luts = &mut self.uint8_luts;
        let float_vals = &mut self.float_vals;
        let mins = &mut self.mins;

        let mut max_span = 0.0f32;
        let mut bias = 0.0f32;

        for g in 0..n_byte_groups {
            let dim_start = g * codes_per_byte;

            let mut lo_min = f32::MAX;
            let mut lo_max = f32::MIN;
            for nibble_val in 0u16..16 {
                let mut s = 0.0f32;
                for c in 0..codes_per_nibble {
                    let shift = (codes_per_nibble - 1 - c) * bits;
                    let code = (nibble_val >> shift) & code_mask;
                    s += q_rot_row[dim_start + c] * centroids[code as usize];
                }
                float_vals[g * 32 + nibble_val as usize] = s;
                lo_min = lo_min.min(s);
                lo_max = lo_max.max(s);
            }

            let mut hi_min = f32::MAX;
            let mut hi_max = f32::MIN;
            for nibble_val in 0u16..16 {
                let mut s = 0.0f32;
                for c in 0..codes_per_nibble {
                    let shift = (codes_per_nibble - 1 - c) * bits;
                    let code = (nibble_val >> shift) & code_mask;
                    s += q_rot_row[dim_start + codes_per_nibble + c] * centroids[code as usize];
                }
                float_vals[g * 32 + 16 + nibble_val as usize] = s;
                hi_min = hi_min.min(s);
                hi_max = hi_max.max(s);
            }

            mins[g * 2] = lo_min;
            mins[g * 2 + 1] = hi_min;
            bias += lo_min + hi_min;

            let lo_span = lo_max - lo_min;
            let hi_span = hi_max - hi_min;
            max_span = max_span.max(lo_span).max(hi_span);
        }

        #[cfg(target_arch = "x86_64")]
        let max_lut = (65535.0 / (n_byte_groups as f64 * 2.0)).floor().min(127.0) as f32;
        #[cfg(not(target_arch = "x86_64"))]
        let max_lut = 127.0f32;

        let scale = if max_span > 1e-10 { max_span / max_lut } else { 1.0 };
        let inv_scale = 1.0 / scale;

        for g in 0..n_byte_groups {
            let lo_min = mins[g * 2];
            let hi_min = mins[g * 2 + 1];
            for i in 0..16 {
                let j_lo = g * 32 + i;
                let j_hi = g * 32 + 16 + i;
                uint8_luts[j_lo] =
                    ((float_vals[j_lo] - lo_min) * inv_scale).round().clamp(0.0, max_lut) as u8;
                uint8_luts[j_hi] =
                    ((float_vals[j_hi] - hi_min) * inv_scale).round().clamp(0.0, max_lut) as u8;
            }
        }

        (scale, bias)
    }
}

/// Reusable per–4-query top‑`k` heap rows (avoids `4×` Vec allocs per batch).
#[derive(Debug)]
pub(crate) struct Neon4xTopkBuffers {
    pub heap_s: [Vec<f32>; 4],
    pub heap_i: [Vec<u32>; 4],
}

impl Neon4xTopkBuffers {
    pub fn new() -> Self {
        Self {
            heap_s: std::array::from_fn(|_| Vec::new()),
            heap_i: std::array::from_fn(|_| Vec::new()),
        }
    }

    pub fn ensure_k(&mut self, k: usize) {
        for v in &mut self.heap_s {
            v.resize(k, 0.0);
        }
        for v in &mut self.heap_i {
            v.resize(k, 0);
        }
    }
}

/// Reusable single-query fused top‑`k` heap (avoids two Vec allocs per query in the tail path).
#[derive(Debug)]
pub(crate) struct Neon1xTopkBuffers {
    heap_s: Vec<f32>,
    heap_i: Vec<u32>,
}

impl Neon1xTopkBuffers {
    pub fn new() -> Self {
        Self {
            heap_s: Vec::new(),
            heap_i: Vec::new(),
        }
    }

    pub fn ensure_k(&mut self, k: usize) {
        self.heap_s.resize(k, 0.0);
        self.heap_i.resize(k, 0);
    }
}

/// Scratch reused across repeated [`crate::TurboQuantIndex::search_topk_indices`] calls on aarch64 NEON
/// (LUT + fused top‑`k` `Vec`s keep capacity instead of reallocating fresh every invocation).
#[cfg(target_arch = "aarch64")]
#[derive(Debug)]
pub(crate) struct Aarch64NeonSearchReuse {
    pub(crate) lut_scratch_batch: [NeonLutScratch; 4],
    pub(crate) lut_scratch_tail: NeonLutScratch,
    pub(crate) topk4: Neon4xTopkBuffers,
    pub(crate) topk1: Neon1xTopkBuffers,
}

#[cfg(target_arch = "aarch64")]
impl Aarch64NeonSearchReuse {
    pub(crate) fn new() -> Self {
        Self {
            lut_scratch_batch: std::array::from_fn(|_| NeonLutScratch::new()),
            lut_scratch_tail: NeonLutScratch::new(),
            topk4: Neon4xTopkBuffers::new(),
            topk1: Neon1xTopkBuffers::new(),
        }
    }
}

/// Per-sub-quantized nibble LUTs from a rotated query row ([`NeonLutScratch::build`]).
#[allow(dead_code)]
pub(crate) fn build_query_neon_lut(q_rot_row: &[f32], centroids: &[f32], bits: usize, dim: usize) -> QueryNeonLut {
    let mut scratch = NeonLutScratch::new();
    let (scale, bias) = scratch.build(q_rot_row, centroids, bits, dim);
    QueryNeonLut {
        uint8_luts: std::mem::take(&mut scratch.uint8_luts),
        scale,
        bias,
    }
}

/// **2- or 4-bit** MSE index with `dim` divisible by `8 / bits` and full decoded flat.
pub(crate) fn neon_mse_blocked_eligible(bits: u8, dim: usize, decoded_len: usize, n: usize) -> bool {
    if n == 0 || decoded_len != n * dim {
        return false;
    }
    let cpb = match bits {
        2 | 4 => (8 / bits) as usize,
        _ => return false,
    };
    dim % cpb == 0
}

/// Row-major decoded codes → blocked ARM-friendly bytes (`n_blocks * n_byte_groups * BLOCK`).
pub(crate) fn build_blocked_codes_from_decoded(decoded_flat: &[u8], n: usize, dim: usize, bits: u8) -> Vec<u8> {
    debug_assert_eq!(decoded_flat.len(), n * dim);
    assert!(
        bits == 2 || bits == 4,
        "build_blocked_codes_from_decoded: only bits 2 and 4 supported"
    );
    let cpb = (8 / bits) as usize;
    assert_eq!(dim % cpb, 0, "dim must be divisible by {}", cpb);
    let n_byte_groups = dim / cpb;
    let mask = (1u32 << bits) - 1;
    let n_blocks = (n + BLOCK - 1) / BLOCK;
    let blocked_size = n_blocks * n_byte_groups * BLOCK;
    let mut blocked = vec![0u8; blocked_size];
    for block_idx in 0..n_blocks {
        let base_vec = block_idx * BLOCK;
        for g in 0..n_byte_groups {
            let out_offset = (block_idx * n_byte_groups + g) * BLOCK;
            for lane in 0..BLOCK {
                let vi = base_vec + lane;
                if vi < n {
                    let row = vi * dim;
                    let dim_start = g * cpb;
                    let mut acc = 0u32;
                    for c in 0..cpb {
                        let code = decoded_flat[row + dim_start + c] as u32 & mask;
                        let sh = ((cpb - 1 - c) * bits as usize) as u32;
                        acc |= code << sh;
                    }
                    blocked[out_offset + lane] = acc as u8;
                }
            }
        }
    }
    blocked
}

#[cfg(target_arch = "aarch64")]
unsafe fn score_4bit_block_neon(
    blocked_codes: &[u8],
    uint8_luts: &[u8],
    block_offset: usize,
    n_byte_groups: usize,
    scale: f32,
    bias: f32,
    norms: &[f32],
    apply_db_norm: bool,
    base_vec: usize,
    n_vectors: usize,
    out: &mut [f32; BLOCK],
) {
    use std::arch::aarch64::*;

    let mask = vdupq_n_u8(0x0F);
    let v_scale = vdupq_n_f32(scale);
    let n_batches = (n_byte_groups + FLUSH_EVERY - 1) / FLUSH_EVERY;

    let mut fa = [vdupq_n_f32(bias); 8];

    let codes_base = blocked_codes.as_ptr().add(block_offset);
    let luts_base = uint8_luts.as_ptr();

    for batch in 0..n_batches {
        let g_start = batch * FLUSH_EVERY;
        let g_end = (g_start + FLUSH_EVERY).min(n_byte_groups);

        let mut accum = [vdupq_n_u16(0); 4];

        let mut g = g_start;
        while g + 3 < g_end {
            let lp0 = luts_base.add(g * 32);
            let lp1 = luts_base.add((g + 1) * 32);
            let lp2 = luts_base.add((g + 2) * 32);
            let lp3 = luts_base.add((g + 3) * 32);
            let cp0 = codes_base.add(g * BLOCK);
            let cp1 = codes_base.add((g + 1) * BLOCK);
            let cp2 = codes_base.add((g + 2) * BLOCK);
            let cp3 = codes_base.add((g + 3) * BLOCK);

            for (lp, cp) in [(lp0, cp0), (lp1, cp1), (lp2, cp2), (lp3, cp3)] {
                let lut_hi = vld1q_u8(lp);
                let lut_lo = vld1q_u8(lp.add(16));
                let c0 = vld1q_u8(cp);
                let c1 = vld1q_u8(cp.add(16));
                let s0 = vaddq_u8(vqtbl1q_u8(lut_lo, vandq_u8(c0, mask)), vqtbl1q_u8(lut_hi, vshrq_n_u8(c0, 4)));
                let s1 = vaddq_u8(vqtbl1q_u8(lut_lo, vandq_u8(c1, mask)), vqtbl1q_u8(lut_hi, vshrq_n_u8(c1, 4)));
                accum[0] = vaddw_u8(accum[0], vget_low_u8(s0));
                accum[1] = vaddw_u8(accum[1], vget_high_u8(s0));
                accum[2] = vaddw_u8(accum[2], vget_low_u8(s1));
                accum[3] = vaddw_u8(accum[3], vget_high_u8(s1));
            }
            g += 4;
        }

        while g < g_end {
            let lp = luts_base.add(g * 32);
            let lut_hi = vld1q_u8(lp);
            let lut_lo = vld1q_u8(lp.add(16));
            let cp = codes_base.add(g * BLOCK);
            let c0 = vld1q_u8(cp);
            let c1 = vld1q_u8(cp.add(16));
            let s0 = vaddq_u8(
                vqtbl1q_u8(lut_lo, vandq_u8(c0, mask)),
                vqtbl1q_u8(lut_hi, vshrq_n_u8(c0, 4)),
            );
            let s1 = vaddq_u8(
                vqtbl1q_u8(lut_lo, vandq_u8(c1, mask)),
                vqtbl1q_u8(lut_hi, vshrq_n_u8(c1, 4)),
            );
            accum[0] = vaddw_u8(accum[0], vget_low_u8(s0));
            accum[1] = vaddw_u8(accum[1], vget_high_u8(s0));
            accum[2] = vaddw_u8(accum[2], vget_low_u8(s1));
            accum[3] = vaddw_u8(accum[3], vget_high_u8(s1));
            g += 1;
        }

        for i in 0..4 {
            let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(accum[i])));
            let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(accum[i])));
            fa[i * 2] = vfmaq_f32(fa[i * 2], v_scale, lo);
            fa[i * 2 + 1] = vfmaq_f32(fa[i * 2 + 1], v_scale, hi);
        }
    }

    let end_lane = (base_vec + BLOCK).min(n_vectors) - base_vec;
    let out_ptr = out.as_mut_ptr();
    let norms_ptr = norms.as_ptr().add(base_vec);

    if end_lane == BLOCK {
        for i in 0..8 {
            let v = fa[i];
            let m = if apply_db_norm {
                vmulq_f32(v, vld1q_f32(norms_ptr.add(i * 4)))
            } else {
                v
            };
            vst1q_f32(out_ptr.add(i * 4), m);
        }
    } else {
        let mut float_accum = [0.0f32; BLOCK];
        for i in 0..8 {
            vst1q_f32(float_accum.as_mut_ptr().add(i * 4), fa[i]);
        }
        for lane in 0..end_lane {
            let mut x = float_accum[lane];
            if apply_db_norm {
                x *= *norms_ptr.add(lane);
            }
            *out_ptr.add(lane) = x;
        }
        for lane in end_lane..BLOCK {
            *out_ptr.add(lane) = f32::NEG_INFINITY;
        }
    }
}

/// Four queries per block: shared DB code loads, nibble splits, and fused NEON accumulation.
#[cfg(target_arch = "aarch64")]
unsafe fn score_4query_block_neon(
    blocked_codes: &[u8],
    luts: [&[u8]; 4],
    block_offset: usize,
    n_byte_groups: usize,
    scales: [f32; 4],
    biases: [f32; 4],
    norms: &[f32],
    apply_db_norm: bool,
    base_vec: usize,
    n_vectors: usize,
    block_out: &mut [[f32; BLOCK]; 4],
) {
    use std::arch::aarch64::*;

    let mask = vdupq_n_u8(0x0F);
    let n_batches = (n_byte_groups + FLUSH_EVERY - 1) / FLUSH_EVERY;

    let mut fa: [[float32x4_t; 8]; 4] = [
        [vdupq_n_f32(biases[0]); 8],
        [vdupq_n_f32(biases[1]); 8],
        [vdupq_n_f32(biases[2]); 8],
        [vdupq_n_f32(biases[3]); 8],
    ];

    let codes_base = blocked_codes.as_ptr().add(block_offset);

    for batch in 0..n_batches {
        let g_start = batch * FLUSH_EVERY;
        let g_end = (g_start + FLUSH_EVERY).min(n_byte_groups);

        let mut acc: [[uint16x8_t; 4]; 4] = [[vdupq_n_u16(0); 4]; 4];

        // Unroll byte-groups by 4 (same rhythm as [`score_4bit_block_neon`]): fewer loop
        // trips vs `for g`, and DB code loads stay once per group while sharing nibbles across queries.
        let mut g = g_start;
        while g + 3 < g_end {
            for gi_off in 0..4usize {
                let gi = g + gi_off;
                let cp = codes_base.add(gi * BLOCK);
                let c0 = vld1q_u8(cp);
                let c1 = vld1q_u8(cp.add(16));
                let lo0 = vandq_u8(c0, mask);
                let lo1 = vandq_u8(c1, mask);
                let hi0 = vshrq_n_u8(c0, 4);
                let hi1 = vshrq_n_u8(c1, 4);

                for q in 0..4 {
                    let lp = luts[q].as_ptr().add(gi * 32);
                    let lut_hi = vld1q_u8(lp);
                    let lut_lo = vld1q_u8(lp.add(16));
                    let s0 = vaddq_u8(vqtbl1q_u8(lut_lo, lo0), vqtbl1q_u8(lut_hi, hi0));
                    let s1 = vaddq_u8(vqtbl1q_u8(lut_lo, lo1), vqtbl1q_u8(lut_hi, hi1));
                    acc[q][0] = vaddw_u8(acc[q][0], vget_low_u8(s0));
                    acc[q][1] = vaddw_u8(acc[q][1], vget_high_u8(s0));
                    acc[q][2] = vaddw_u8(acc[q][2], vget_low_u8(s1));
                    acc[q][3] = vaddw_u8(acc[q][3], vget_high_u8(s1));
                }
            }
            g += 4;
        }

        while g < g_end {
            let cp = codes_base.add(g * BLOCK);
            let c0 = vld1q_u8(cp);
            let c1 = vld1q_u8(cp.add(16));
            let lo0 = vandq_u8(c0, mask);
            let lo1 = vandq_u8(c1, mask);
            let hi0 = vshrq_n_u8(c0, 4);
            let hi1 = vshrq_n_u8(c1, 4);

            for q in 0..4 {
                let lp = luts[q].as_ptr().add(g * 32);
                let lut_hi = vld1q_u8(lp);
                let lut_lo = vld1q_u8(lp.add(16));
                let s0 = vaddq_u8(vqtbl1q_u8(lut_lo, lo0), vqtbl1q_u8(lut_hi, hi0));
                let s1 = vaddq_u8(vqtbl1q_u8(lut_lo, lo1), vqtbl1q_u8(lut_hi, hi1));
                acc[q][0] = vaddw_u8(acc[q][0], vget_low_u8(s0));
                acc[q][1] = vaddw_u8(acc[q][1], vget_high_u8(s0));
                acc[q][2] = vaddw_u8(acc[q][2], vget_low_u8(s1));
                acc[q][3] = vaddw_u8(acc[q][3], vget_high_u8(s1));
            }
            g += 1;
        }

        for q in 0..4 {
            let v_scale = vdupq_n_f32(scales[q]);
            for i in 0..4 {
                let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(acc[q][i])));
                let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(acc[q][i])));
                fa[q][i * 2] = vfmaq_f32(fa[q][i * 2], v_scale, lo);
                fa[q][i * 2 + 1] = vfmaq_f32(fa[q][i * 2 + 1], v_scale, hi);
            }
        }
    }

    let end_lane = (base_vec + BLOCK).min(n_vectors) - base_vec;
    let norms_ptr = norms.as_ptr().add(base_vec);

    for q in 0..4 {
        let rp = block_out[q].as_mut_ptr();
        if end_lane == BLOCK {
            for i in 0..8 {
                let v = fa[q][i];
                let m = if apply_db_norm {
                    vmulq_f32(v, vld1q_f32(norms_ptr.add(i * 4)))
                } else {
                    v
                };
                vst1q_f32(rp.add(i * 4), m);
            }
        } else {
            let mut buf = [0.0f32; BLOCK];
            for i in 0..8 {
                vst1q_f32(buf.as_mut_ptr().add(i * 4), fa[q][i]);
            }
            for lane in 0..end_lane {
                let mut x = buf[lane];
                if apply_db_norm {
                    x *= *norms_ptr.add(lane);
                }
                *rp.add(lane) = x;
            }
            for lane in end_lane..BLOCK {
                *rp.add(lane) = f32::NEG_INFINITY;
            }
        }
    }
}

/// Four contiguous query score rows (`out_flat.len() >= 4 * n_vectors`).
#[cfg(target_arch = "aarch64")]
pub(crate) fn scores_4x_4bit_blocked_neon(
    blocked: &[u8],
    lut_u8: [&[u8]; 4],
    scales: [f32; 4],
    biases: [f32; 4],
    n_byte_groups: usize,
    norms: &[f32],
    n_vectors: usize,
    apply_db_norm: bool,
    out_flat: &mut [f32],
) {
    assert!(out_flat.len() >= 4 * n_vectors);
    assert_eq!(norms.len(), n_vectors);
    out_flat[..4 * n_vectors].fill(f32::NEG_INFINITY);

    let n_blocks = (n_vectors + BLOCK - 1) / BLOCK;
    let mut scratch = [[0f32; BLOCK]; 4];
    for block_idx in 0..n_blocks {
        let base_vec = block_idx * BLOCK;
        let block_offset = block_idx * n_byte_groups * BLOCK;
        unsafe {
            score_4query_block_neon(
                blocked,
                lut_u8,
                block_offset,
                n_byte_groups,
                scales,
                biases,
                norms,
                apply_db_norm,
                base_vec,
                n_vectors,
                &mut scratch,
            );
        }
        let end_lane = (base_vec + BLOCK).min(n_vectors) - base_vec;
        for q in 0..4 {
            for lane in 0..end_lane {
                out_flat[q * n_vectors + base_vec + lane] = scratch[q][lane];
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn min_heap_push_topk(
    heap_s: &mut [f32],
    heap_i: &mut [u32],
    sz: &mut usize,
    hmin: &mut f32,
    hmi: &mut usize,
    k: usize,
    score: f32,
    idx: usize,
) {
    if *sz < k {
        heap_s[*sz] = score;
        heap_i[*sz] = idx as u32;
        *sz += 1;
        if *sz == k {
            *hmin = heap_s[0];
            *hmi = 0;
            for h in 1..k {
                if heap_s[h] < *hmin {
                    *hmin = heap_s[h];
                    *hmi = h;
                }
            }
        }
    } else if score > *hmin {
        heap_s[*hmi] = score;
        heap_i[*hmi] = idx as u32;
        *hmin = heap_s[0];
        *hmi = 0;
        for h in 1..k {
            if heap_s[h] < *hmin {
                *hmin = heap_s[h];
                *hmi = h;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn min_heap_finalize_to_indices(heap_s: &[f32], heap_i: &[u32], sz: usize) -> Vec<usize> {
    let mut pairs: Vec<(f32, usize)> = heap_s[..sz]
        .iter()
        .zip(heap_i[..sz].iter())
        .map(|(&s, &i)| (s, i as usize))
        .collect();
    pairs.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    pairs.into_iter().map(|(_, i)| i).collect()
}

/// Fused DB scan + top‑`k` for four queries (no full `4 × n` score buffer).
/// **`scale_by_query_norm`**: when **`false`** (cosine ranking), **`qnorm_scale`** is skipped—same numerical result as multiplying by **`1.0`**, minus per‑candidate `fmul` cost on the fused path.
#[cfg(target_arch = "aarch64")]
pub(crate) fn search_4x_4bit_blocked_neon_topk(
    blocked: &[u8],
    lut_u8: [&[u8]; 4],
    scales: [f32; 4],
    biases: [f32; 4],
    n_byte_groups: usize,
    norms: &[f32],
    n_vectors: usize,
    apply_db_norm: bool,
    scale_by_query_norm: bool,
    qnorm_scale: [f32; 4],
    k: usize,
    bufs: &mut Neon4xTopkBuffers,
) -> [Vec<usize>; 4] {
    assert!(k > 0 && k < n_vectors);
    assert_eq!(norms.len(), n_vectors);

    bufs.ensure_k(k);

    let heap_s = &mut bufs.heap_s;
    let heap_i = &mut bufs.heap_i;
    let mut sz = [0usize; 4];
    let mut hmin = [f32::NEG_INFINITY; 4];
    let mut hmi = [0usize; 4];

    let mut scratch = [[0f32; BLOCK]; 4];
    let n_blocks = (n_vectors + BLOCK - 1) / BLOCK;
    for block_idx in 0..n_blocks {
        let base_vec = block_idx * BLOCK;
        let block_offset = block_idx * n_byte_groups * BLOCK;
        unsafe {
            score_4query_block_neon(
                blocked,
                lut_u8,
                block_offset,
                n_byte_groups,
                scales,
                biases,
                norms,
                apply_db_norm,
                base_vec,
                n_vectors,
                &mut scratch,
            );
        }
        let end_lane = (base_vec + BLOCK).min(n_vectors) - base_vec;
        if scale_by_query_norm {
            for q in 0..4 {
                let mul = qnorm_scale[q];
                for lane in 0..end_lane {
                    let score = scratch[q][lane] * mul;
                    min_heap_push_topk(
                        &mut heap_s[q],
                        &mut heap_i[q],
                        &mut sz[q],
                        &mut hmin[q],
                        &mut hmi[q],
                        k,
                        score,
                        base_vec + lane,
                    );
                }
            }
        } else {
            for q in 0..4 {
                for lane in 0..end_lane {
                    let score = scratch[q][lane];
                    min_heap_push_topk(
                        &mut heap_s[q],
                        &mut heap_i[q],
                        &mut sz[q],
                        &mut hmin[q],
                        &mut hmi[q],
                        k,
                        score,
                        base_vec + lane,
                    );
                }
            }
        }
    }
    [
        min_heap_finalize_to_indices(&heap_s[0], &heap_i[0], sz[0]),
        min_heap_finalize_to_indices(&heap_s[1], &heap_i[1], sz[1]),
        min_heap_finalize_to_indices(&heap_s[2], &heap_i[2], sz[2]),
        min_heap_finalize_to_indices(&heap_s[3], &heap_i[3], sz[3]),
    ]
}

/// Fused scan + top‑`k`; see [`search_4x_4bit_blocked_neon_topk`] for **`scale_by_query_norm`**.
#[cfg(target_arch = "aarch64")]
pub(crate) fn search_1x_4bit_blocked_neon_topk(
    blocked: &[u8],
    lut_u8: &[u8],
    scale: f32,
    bias: f32,
    n_byte_groups: usize,
    norms: &[f32],
    n_vectors: usize,
    apply_db_norm: bool,
    scale_by_query_norm: bool,
    qnorm_scale: f32,
    k: usize,
    bufs: &mut Neon1xTopkBuffers,
) -> Vec<usize> {
    assert!(k > 0 && k < n_vectors);
    assert_eq!(norms.len(), n_vectors);

    bufs.ensure_k(k);
    let heap_s = &mut bufs.heap_s;
    let heap_i = &mut bufs.heap_i;
    let mut sz = 0usize;
    let mut hmin = f32::NEG_INFINITY;
    let mut hmi = 0usize;
    let mut block_out = [0f32; BLOCK];
    let n_blocks = (n_vectors + BLOCK - 1) / BLOCK;
    for block_idx in 0..n_blocks {
        let base_vec = block_idx * BLOCK;
        let block_offset = block_idx * n_byte_groups * BLOCK;
        unsafe {
            score_4bit_block_neon(
                blocked,
                lut_u8,
                block_offset,
                n_byte_groups,
                scale,
                bias,
                norms,
                apply_db_norm,
                base_vec,
                n_vectors,
                &mut block_out,
            );
        }
        let end_lane = (base_vec + BLOCK).min(n_vectors) - base_vec;
        if scale_by_query_norm {
            for lane in 0..end_lane {
                let score = block_out[lane] * qnorm_scale;
                min_heap_push_topk(
                    heap_s,
                    heap_i,
                    &mut sz,
                    &mut hmin,
                    &mut hmi,
                    k,
                    score,
                    base_vec + lane,
                );
            }
        } else {
            for lane in 0..end_lane {
                let score = block_out[lane];
                min_heap_push_topk(
                    heap_s,
                    heap_i,
                    &mut sz,
                    &mut hmin,
                    &mut hmi,
                    k,
                    score,
                    base_vec + lane,
                );
            }
        }
    }
    min_heap_finalize_to_indices(heap_s, heap_i, sz)
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn search_4x_4bit_blocked_neon_topk(
    _blocked: &[u8],
    _lut_u8: [&[u8]; 4],
    _scales: [f32; 4],
    _biases: [f32; 4],
    _n_byte_groups: usize,
    _norms: &[f32],
    _n_vectors: usize,
    _apply_db_norm: bool,
    _scale_by_query_norm: bool,
    _qnorm_scale: [f32; 4],
    _k: usize,
    _bufs: &mut Neon4xTopkBuffers,
) -> [Vec<usize>; 4] {
    [Vec::new(), Vec::new(), Vec::new(), Vec::new()]
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn search_1x_4bit_blocked_neon_topk(
    _blocked: &[u8],
    _lut_u8: &[u8],
    _scale: f32,
    _bias: f32,
    _n_byte_groups: usize,
    _norms: &[f32],
    _n_vectors: usize,
    _apply_db_norm: bool,
    _scale_by_query_norm: bool,
    _qnorm_scale: f32,
    _k: usize,
    _bufs: &mut Neon1xTopkBuffers,
) -> Vec<usize> {
    Vec::new()
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn scores_4x_4bit_blocked_neon(
    _blocked: &[u8],
    _lut_u8: [&[u8]; 4],
    _scales: [f32; 4],
    _biases: [f32; 4],
    _n_byte_groups: usize,
    _norms: &[f32],
    n_vectors: usize,
    _apply_db_norm: bool,
    out_flat: &mut [f32],
) {
    if out_flat.len() >= 4 * n_vectors {
        out_flat[..4 * n_vectors].fill(f32::NEG_INFINITY);
    }
}

/// Fill `out_scores[..n_vectors]` using blocked 4-bit NEON scoring when available.
#[cfg(target_arch = "aarch64")]
pub(crate) fn scores_4bit_blocked_neon(
    blocked: &[u8],
    lut_u8: &[u8],
    scale: f32,
    bias: f32,
    n_byte_groups: usize,
    norms: &[f32],
    n_vectors: usize,
    apply_db_norm: bool,
    out_scores: &mut [f32],
) {
    assert_eq!(out_scores.len(), n_vectors);
    assert_eq!(norms.len(), n_vectors);
    out_scores.fill(f32::NEG_INFINITY);

    let n_blocks = (n_vectors + BLOCK - 1) / BLOCK;
    for block_idx in 0..n_blocks {
        let base_vec = block_idx * BLOCK;
        let block_offset = block_idx * n_byte_groups * BLOCK;
        let end = (base_vec + BLOCK).min(n_vectors);
        let mut block_out = [0.0f32; BLOCK];
        unsafe {
            score_4bit_block_neon(
                blocked,
                lut_u8,
                block_offset,
                n_byte_groups,
                scale,
                bias,
                norms,
                apply_db_norm,
                base_vec,
                n_vectors,
                &mut block_out,
            );
        }
        for lane in 0..(end - base_vec) {
            out_scores[base_vec + lane] = block_out[lane];
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn scores_4bit_blocked_neon(
    _blocked: &[u8],
    _lut_u8: &[u8],
    _scale: f32,
    _bias: f32,
    _n_byte_groups: usize,
    _norms: &[f32],
    _n_vectors: usize,
    _apply_db_norm: bool,
    out_scores: &mut [f32],
) {
    out_scores.fill(f32::NEG_INFINITY);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_layout_nibble_order_matches_pair_of_dims() {
        let n = 3usize;
        let dim = 4usize;
        let mut dec = vec![0u8; n * dim];
        dec[0 * dim + 0] = 0xA;
        dec[0 * dim + 1] = 0xB;
        dec[0 * dim + 2] = 0xC;
        dec[0 * dim + 3] = 0xD;
        let blocked = build_blocked_codes_from_decoded(&dec, n, dim, 4);
        let ng = dim / 2;
        let b0_g0_lane0 = blocked[0 * ng * BLOCK + 0 * BLOCK + 0];
        assert_eq!(b0_g0_lane0, (0xA << 4) | 0xB);
        let b0_g1_lane0 = blocked[0 * ng * BLOCK + 1 * BLOCK + 0];
        assert_eq!(b0_g1_lane0, (0xC << 4) | 0xD);
    }

    #[test]
    fn blocked_layout_2bit_quartet_matches_shift_order() {
        let n = 1usize;
        let dim = 8usize;
        let mut dec = vec![0u8; n * dim];
        dec[0] = 0;
        dec[1] = 1;
        dec[2] = 2;
        dec[3] = 3;
        let blocked = build_blocked_codes_from_decoded(&dec, n, dim, 2);
        let ng = dim / 4;
        let expected = (0u32 << 6) | (1 << 4) | (2 << 2) | 3;
        assert_eq!(blocked[0 * ng * BLOCK + 0 * BLOCK + 0], expected as u8);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn scores_4x_matches_four_singles_2bit() {
        let dim = 16usize;
        let n = 32usize;
        let nbg = dim / 4;
        let bits_lut = 2usize;
        let mut decoded = vec![0u8; n * dim];
        for i in 0..decoded.len() {
            decoded[i] = (i % 4) as u8;
        }
        let centroids: Vec<f32> = (0..4).map(|i| i as f32 * 0.05 - 0.1).collect();
        let blocked = build_blocked_codes_from_decoded(&decoded, n, dim, 2);
        let norms: Vec<f32> = vec![1.0; n];
        let q0: Vec<f32> = (0..dim).map(|j| 0.1 * j as f32 - 0.7).collect();
        let q1: Vec<f32> = (0..dim).map(|j| -0.05 * j as f32 + 0.3).collect();
        let q2: Vec<f32> = (0..dim).map(|j| 0.02 * (j as f32).sin()).collect();
        let q3: Vec<f32> = (0..dim).map(|j| 0.01 * (j * j) as f32 - 0.4).collect();
        let l0 = build_query_neon_lut(&q0, &centroids, bits_lut, dim);
        let l1 = build_query_neon_lut(&q1, &centroids, bits_lut, dim);
        let l2 = build_query_neon_lut(&q2, &centroids, bits_lut, dim);
        let l3 = build_query_neon_lut(&q3, &centroids, bits_lut, dim);
        let lut_u8: [&[u8]; 4] = [
            &l0.uint8_luts[..],
            &l1.uint8_luts[..],
            &l2.uint8_luts[..],
            &l3.uint8_luts[..],
        ];
        let scales = [l0.scale, l1.scale, l2.scale, l3.scale];
        let biases = [l0.bias, l1.bias, l2.bias, l3.bias];
        let mut four = vec![f32::NEG_INFINITY; 4 * n];
        scores_4x_4bit_blocked_neon(&blocked, lut_u8, scales, biases, nbg, &norms, n, false, &mut four);
        for (lut, off) in [(&l0, 0usize), (&l1, 1), (&l2, 2), (&l3, 3)] {
            let mut one = vec![f32::NEG_INFINITY; n];
            scores_4bit_blocked_neon(
                &blocked,
                &lut.uint8_luts,
                lut.scale,
                lut.bias,
                nbg,
                &norms,
                n,
                false,
                &mut one,
            );
            for i in 0..n {
                let a = four[off * n + i];
                let b = one[i];
                assert!((a - b).abs() < 1e-4, "off={off} i={i} a={a} b={b}");
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fused_topk_4x_matches_materialized_small() {
        let dim = 16usize;
        let n = 200usize;
        let k = 10usize;
        let nbg = dim / 2;
        let bits = 4usize;
        let mut decoded = vec![0u8; n * dim];
        for i in 0..decoded.len() {
            decoded[i] = ((i * 7) % 16) as u8;
        }
        let centroids: Vec<f32> = (0..16).map(|i| i as f32 * 0.02 - 0.15).collect();
        let blocked = build_blocked_codes_from_decoded(&decoded, n, dim, 4);
        let norms: Vec<f32> = (0..n).map(|i| 0.9 + (i as f32) * 0.0003).collect();
        let q0: Vec<f32> = (0..dim).map(|j| 0.1 * j as f32 - 0.7).collect();
        let q1: Vec<f32> = (0..dim).map(|j| -0.05 * j as f32 + 0.3).collect();
        let q2: Vec<f32> = (0..dim).map(|j| 0.02 * (j as f32).sin()).collect();
        let q3: Vec<f32> = (0..dim).map(|j| 0.01 * (j * j) as f32 - 0.4).collect();
        let l0 = build_query_neon_lut(&q0, &centroids, bits, dim);
        let l1 = build_query_neon_lut(&q1, &centroids, bits, dim);
        let l2 = build_query_neon_lut(&q2, &centroids, bits, dim);
        let l3 = build_query_neon_lut(&q3, &centroids, bits, dim);
        let lut_u8: [&[u8]; 4] = [
            &l0.uint8_luts[..],
            &l1.uint8_luts[..],
            &l2.uint8_luts[..],
            &l3.uint8_luts[..],
        ];
        let scales = [l0.scale, l1.scale, l2.scale, l3.scale];
        let biases = [l0.bias, l1.bias, l2.bias, l3.bias];
        let qscale = [1.1f32, 0.95, 1.03, 1.0];
        let mut topk4 = Neon4xTopkBuffers::new();
        let fused = search_4x_4bit_blocked_neon_topk(
            &blocked,
            lut_u8,
            scales,
            biases,
            nbg,
            &norms,
            n,
            true,
            true,
            qscale,
            k,
            &mut topk4,
        );
        let mut scores4 = vec![f32::NEG_INFINITY; 4 * n];
        scores_4x_4bit_blocked_neon(&blocked, lut_u8, scales, biases, nbg, &norms, n, true, &mut scores4);
        for q in 0..4 {
            let row = &mut scores4[q * n..(q + 1) * n];
            for s in row.iter_mut() {
                *s *= qscale[q];
            }
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_unstable_by(|&a, &b| row[b].total_cmp(&row[a]));
            let exp: Vec<usize> = order.into_iter().take(k).collect();
            assert_eq!(fused[q], exp, "query {q}");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn scores_4x_matches_four_singles() {
        let dim = 16usize;
        let n = 64usize;
        let nbg = dim / 2;
        let bits = 4usize;
        let mut decoded = vec![0u8; n * dim];
        for i in 0..decoded.len() {
            decoded[i] = (i % 16) as u8;
        }
        let centroids: Vec<f32> = (0..16).map(|i| i as f32 * 0.02 - 0.15).collect();
        let blocked = build_blocked_codes_from_decoded(&decoded, n, dim, 4);
        let norms: Vec<f32> = vec![1.0; n];
        let q0: Vec<f32> = (0..dim).map(|j| 0.1 * j as f32 - 0.7).collect();
        let q1: Vec<f32> = (0..dim).map(|j| -0.05 * j as f32 + 0.3).collect();
        let q2: Vec<f32> = (0..dim).map(|j| 0.02 * (j as f32).sin()).collect();
        let q3: Vec<f32> = (0..dim).map(|j| 0.01 * (j * j) as f32 - 0.4).collect();
        let l0 = build_query_neon_lut(&q0, &centroids, bits, dim);
        let l1 = build_query_neon_lut(&q1, &centroids, bits, dim);
        let l2 = build_query_neon_lut(&q2, &centroids, bits, dim);
        let l3 = build_query_neon_lut(&q3, &centroids, bits, dim);
        let lut_u8: [&[u8]; 4] = [
            &l0.uint8_luts[..],
            &l1.uint8_luts[..],
            &l2.uint8_luts[..],
            &l3.uint8_luts[..],
        ];
        let scales = [l0.scale, l1.scale, l2.scale, l3.scale];
        let biases = [l0.bias, l1.bias, l2.bias, l3.bias];
        let mut four = vec![f32::NEG_INFINITY; 4 * n];
        scores_4x_4bit_blocked_neon(&blocked, lut_u8, scales, biases, nbg, &norms, n, false, &mut four);
        for (lut, off) in [(&l0, 0usize), (&l1, 1), (&l2, 2), (&l3, 3)] {
            let mut one = vec![f32::NEG_INFINITY; n];
            scores_4bit_blocked_neon(
                &blocked,
                &lut.uint8_luts,
                lut.scale,
                lut.bias,
                nbg,
                &norms,
                n,
                false,
                &mut one,
            );
            for i in 0..n {
                let a = four[off * n + i];
                let b = one[i];
                assert!((a - b).abs() < 1e-4, "off={off} i={i} a={a} b={b}");
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_scores_rank_similar_to_f32_lut_small() {
        use crate::simd::{flatten_query_lut, scores_for_query_decoded, Scorer};

        let dim = 8usize;
        let n = 48usize;
        let bits = 4u8;
        let nl = 16usize;
        let mut decoded = vec![0u8; n * dim];
        for i in 0..decoded.len() {
            decoded[i] = (i % nl) as u8;
        }
        let centroids: Vec<f32> = (0..nl).map(|i| i as f32 * 0.03 - 0.2).collect();
        let q_rot: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 - 0.35).collect();
        let lut = flatten_query_lut(&q_rot, &centroids, dim);
        let norms: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let wide = scores_for_query_decoded(
            &decoded,
            &norms,
            dim,
            n,
            &lut,
            nl,
            1.0,
            true,
            Scorer::Wide,
        );
        let blocked = build_blocked_codes_from_decoded(&decoded, n, dim, 4);
        let neon_lut = build_query_neon_lut(&q_rot, &centroids, bits as usize, dim);
        let mut neon = vec![0f32; n];
        scores_4bit_blocked_neon(
            &blocked,
            &neon_lut.uint8_luts,
            neon_lut.scale,
            neon_lut.bias,
            dim / 2,
            &norms,
            n,
            true,
            &mut neon,
        );
        let mut order_w: Vec<usize> = (0..n).collect();
        order_w.sort_unstable_by(|&a, &b| wide[b].total_cmp(&wide[a]));
        let mut order_n: Vec<usize> = (0..n).collect();
        order_n.sort_unstable_by(|&a, &b| neon[b].total_cmp(&neon[a]));
        let top = 8usize;
        let mut hit = 0usize;
        for &i in &order_n[..top] {
            if order_w[..top].contains(&i) {
                hit += 1;
            }
        }
        assert!(hit >= top / 2, "neon top-{top} overlap {hit} with f32 path");
    }
}
