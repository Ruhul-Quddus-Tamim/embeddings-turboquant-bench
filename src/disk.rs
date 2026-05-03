//! Versioned snapshot I/O (`TQ01` magic, **v1** / **v2** MSE with rotation `Q` on disk).
use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

use ndarray::Array2;

use crate::{
    index::TurboQuantIndex,
    lloyd_max::GaussianQuantizer,
    packing::packed_row_bytes,
};

/// File magic + format version (`v1` / `v2`).
pub const MAGIC: &[u8; 4] = b"TQ01";
const VERSION_V1: u32 = 1;
const VERSION_V2: u32 = 2;
const VERSION_V3: u32 = 3;

const SCHEME_MSE: u8 = 0;
/// Historical: product+QJL snapshots are no longer loaded; see [`read_v2`].
const SCHEME_PROD: u8 = 1;

fn wr_u32<W: Write>(w: &mut W, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn wr_u64<W: Write>(w: &mut W, v: u64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn rd_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn rd_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn rd_f32<R: Read>(r: &mut R) -> std::io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

pub fn write(path: &Path, idx: &TurboQuantIndex) -> std::io::Result<()> {
    write_v2(path, idx)
}

fn write_v2(path: &Path, idx: &TurboQuantIndex) -> std::io::Result<()> {
    let mut w = File::create(path)?;
    w.write_all(MAGIC)?;
    wr_u32(&mut w, VERSION_V2)?;
    wr_u32(&mut w, idx.dim() as u32)?;
    w.write_all(&[idx.bits()])?;
    w.write_all(&[SCHEME_MSE])?;
    wr_u64(&mut w, idx.seed())?;
    wr_u32(&mut w, idx.lloyd_iterations() as u32)?;
    wr_u64(&mut w, idx.len() as u64)?;

    for &x in idx.rotation().iter() {
        w.write_all(&x.to_le_bytes())?;
    }

    let b = idx.scalar_quant_tables().boundaries.len() as u32;
    wr_u32(&mut w, b)?;
    for &x in idx.scalar_quant_tables().boundaries.iter() {
        w.write_all(&x.to_le_bytes())?;
    }
    let c = idx.scalar_quant_tables().centroids.len() as u32;
    wr_u32(&mut w, c)?;
    for &x in idx.scalar_quant_tables().centroids.iter() {
        w.write_all(&x.to_le_bytes())?;
    }

    for &x in idx.norms() {
        w.write_all(&x.to_le_bytes())?;
    }
    w.write_all(idx.packed_codes())?;

    Ok(())
}

pub fn read(path: &Path) -> std::io::Result<TurboQuantIndex> {
    let mut r = File::open(path)?;
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid TurboQuant snapshot magic",
        ));
    }
    let ver = rd_u32(&mut r)?;

    match ver {
        VERSION_V1 => read_v1(&mut r),
        VERSION_V2 => read_v2(&mut r),
        VERSION_V3 => read_v3_unsupported(),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "unsupported snapshot version",
        )),
    }
}

fn read_v3_unsupported() -> std::io::Result<TurboQuantIndex> {
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "snapshot v3 (historical third-party bit-plane layout) is no longer supported; use v1/v2 snapshots or rebuild the index with this crate",
    ))
}

fn read_v1<R: Read>(r: &mut R) -> std::io::Result<TurboQuantIndex> {
    let dim = rd_u32(r)? as usize;
    let mut bits_arr = [0u8; 1];
    r.read_exact(&mut bits_arr)?;
    let bits = bits_arr[0];
    let seed = rd_u64(r)?;
    let lloyd_iterations = rd_u32(r)? as usize;
    let nvec = rd_u64(r)? as usize;

    let rotation_q = read_matrix_f32(r, dim)?;

    let quantizer = read_quantizer(r, dim, bits)?;
    let norms = read_norms(r, nvec)?;
    let row_b = packed_row_bytes(dim, bits);
    let packed = read_packed(r, nvec, row_b)?;

    Ok(TurboQuantIndex::from_loaded(
        dim,
        bits,
        seed,
        lloyd_iterations,
        rotation_q,
        quantizer,
        norms,
        packed,
    ))
}

fn read_v2<R: Read>(r: &mut R) -> std::io::Result<TurboQuantIndex> {
    let dim = rd_u32(r)? as usize;
    let mut bits_arr = [0u8; 1];
    r.read_exact(&mut bits_arr)?;
    let bits = bits_arr[0];
    let mut scheme_arr = [0u8; 1];
    r.read_exact(&mut scheme_arr)?;
    let scheme = scheme_arr[0];
    if scheme == SCHEME_PROD {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "snapshot format Q_prod (product + QJL) is no longer supported; use an MSE-only snapshot",
        ));
    }
    if scheme != SCHEME_MSE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid quantizer scheme byte",
        ));
    }
    let seed = rd_u64(r)?;
    let lloyd_iterations = rd_u32(r)? as usize;
    let nvec = rd_u64(r)? as usize;

    let rotation_q = read_matrix_f32(r, dim)?;

    let quantizer = read_quantizer(r, dim, bits)?;
    let norms = read_norms(r, nvec)?;
    let row_b = packed_row_bytes(dim, bits);
    let packed = read_packed(r, nvec, row_b)?;

    Ok(TurboQuantIndex::from_loaded(
        dim,
        bits,
        seed,
        lloyd_iterations,
        rotation_q,
        quantizer,
        norms,
        packed,
    ))
}

fn read_matrix_f32<R: Read>(r: &mut R, dim: usize) -> std::io::Result<Array2<f32>> {
    let n = dim * dim;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(rd_f32(r)?);
    }
    Array2::from_shape_vec((dim, dim), v).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
    })
}

fn read_quantizer<R: Read>(
    r: &mut R,
    dim: usize,
    bits: u8,
) -> std::io::Result<GaussianQuantizer> {
    let nb = rd_u32(r)? as usize;
    let mut boundaries = Vec::with_capacity(nb);
    for _ in 0..nb {
        boundaries.push(rd_f32(r)?);
    }
    let nc = rd_u32(r)? as usize;
    let mut centroids = Vec::with_capacity(nc);
    for _ in 0..nc {
        centroids.push(rd_f32(r)?);
    }
    let sigma = (1.0f32 / dim as f32).sqrt();
    Ok(GaussianQuantizer::from_quant_tables(
        bits,
        sigma,
        boundaries,
        centroids,
    ))
}

fn read_norms<R: Read>(r: &mut R, nvec: usize) -> std::io::Result<Vec<f32>> {
    let mut norms = Vec::with_capacity(nvec);
    for _ in 0..nvec {
        norms.push(rd_f32(r)?);
    }
    Ok(norms)
}

fn read_packed<R: Read>(r: &mut R, nvec: usize, row_b: usize) -> std::io::Result<Vec<u8>> {
    let total = nvec
        .checked_mul(row_b)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "packed size overflow"))?;
    let mut packed = vec![0u8; total];
    r.read_exact(&mut packed)?;
    Ok(packed)
}
