//! Versioned snapshot I/O (`TQ01` magic, format `v1` / `v2`).
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

/// File magic + format version (`v1` / `v2` little-endian payloads).
pub const MAGIC: &[u8; 4] = b"TQ01";
const VERSION_V1: u32 = 1;
const VERSION_V2: u32 = 2;

const SCHEME_MSE: u8 = 0;
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
    let mut w = File::create(path)?;
    w.write_all(MAGIC)?;
    wr_u32(&mut w, VERSION_V2)?;
    wr_u32(&mut w, idx.dim() as u32)?;
    w.write_all(&[idx.bits()])?;
    let scheme = if idx.is_prod_quantizer() {
        SCHEME_PROD
    } else {
        SCHEME_MSE
    };
    w.write_all(&[scheme])?;
    wr_u64(&mut w, idx.seed())?;
    wr_u32(&mut w, idx.lloyd_iterations() as u32)?;
    wr_u64(&mut w, idx.len() as u64)?;

    for &x in idx.rotation().iter() {
        w.write_all(&x.to_le_bytes())?;
    }

    if let Some(sv) = idx.s_matrix() {
        for &x in sv.iter() {
            w.write_all(&x.to_le_bytes())?;
        }
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
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "unsupported snapshot version",
        )),
    }
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
        None,
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
    let seed = rd_u64(r)?;
    let lloyd_iterations = rd_u32(r)? as usize;
    let nvec = rd_u64(r)? as usize;

    let rotation_q = read_matrix_f32(r, dim)?;

    let s_matrix = if scheme == SCHEME_PROD {
        if !(2..=8).contains(&bits) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Q_prod snapshot requires bits in 2..=8",
            ));
        }
        Some(read_matrix_f32(r, dim)?)
    } else if scheme != SCHEME_MSE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid quantizer scheme byte",
        ));
    } else {
        None
    };

    let quant_bits = if s_matrix.is_some() { bits - 1 } else { bits };
    let quantizer = read_quantizer(r, dim, quant_bits)?;
    let norms = read_norms(r, nvec)?;
    let row_b = if s_matrix.is_some() {
        packed_row_bytes(dim, quant_bits) + packed_row_bytes(dim, 1) + 4
    } else {
        packed_row_bytes(dim, bits)
    };
    let packed = read_packed(r, nvec, row_b)?;

    Ok(TurboQuantIndex::from_loaded(
        dim,
        bits,
        seed,
        lloyd_iterations,
        rotation_q,
        s_matrix,
        quantizer,
        norms,
        packed,
    ))
}

fn read_matrix_f32<R: Read>(r: &mut R, dim: usize) -> std::io::Result<Array2<f32>> {
    let mut q_flat = Vec::with_capacity(dim * dim);
    for _ in 0..(dim * dim) {
        q_flat.push(rd_f32(r)?);
    }
    Array2::from_shape_vec((dim, dim), q_flat).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, format!("matrix shape {e:?}"))
    })
}

fn read_quantizer<R: Read>(r: &mut R, dim: usize, bits: u8) -> std::io::Result<GaussianQuantizer> {
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
        bits, sigma, boundaries, centroids,
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
    let packed_len = nvec * row_b;
    let mut packed = vec![0u8; packed_len];
    r.read_exact(&mut packed)?;
    Ok(packed)
}
