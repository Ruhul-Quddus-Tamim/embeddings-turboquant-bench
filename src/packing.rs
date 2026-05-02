//! Bit-packed quantized codes (`b` bits per dimension, least-significant-first within stream).

#[inline]
pub fn packed_row_bytes(dim: usize, bits: u8) -> usize {
    (dim * bits as usize + 7) / 8
}

/// Pack `codes`, one `bits`-wide symbol per dimension (`dim` symbols).
pub fn pack_row(codes: &[u8], dim: usize, bits: u8, out: &mut [u8]) {
    assert_eq!(codes.len(), dim);
    let pb = packed_row_bytes(dim, bits);
    debug_assert!(out.len() >= pb);
    out[..pb].fill(0);

    let mut bit_pos = 0usize;
    for d in 0..dim {
        let mask = (1u32 << bits) - 1;
        let mut v = (codes[d] as u32) & mask;
        for _ in 0..bits {
            let byte = bit_pos / 8;
            let shift = bit_pos % 8;
            out[byte] |= ((v & 1) as u8) << shift;
            v >>= 1;
            bit_pos += 1;
        }
    }
}

#[inline]
pub fn unpack_code(bytes: &[u8], dim: usize, bits: u8, dim_idx: usize) -> u8 {
    debug_assert!(dim_idx < dim);
    let bit_start = dim_idx * bits as usize;
    let mut v = 0u16;
    for b in 0..bits as usize {
        let pos = bit_start + b;
        let byte = pos / 8;
        let shift = pos % 8;
        let bit = ((bytes[byte] >> shift) & 1) as u16;
        v |= bit << b;
    }
    v as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_2_and_4bit() {
        for bits in [2u8, 4u8] {
            let dim = 11;
            let codes: Vec<u8> = (0..dim as u8).map(|i| i & ((1 << bits) - 1)).collect();
            let plen = packed_row_bytes(dim, bits);
            let mut buf = vec![0u8; plen];
            pack_row(&codes, dim, bits, &mut buf);
            for d in 0..dim {
                assert_eq!(
                    unpack_code(&buf, dim, bits, d),
                    codes[d],
                    "bits={bits} d={d}"
                );
            }
        }
    }
}
