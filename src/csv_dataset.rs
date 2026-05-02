//! Shared CSV embedding loading and train/query split for benchmarks.
//! Same contract as `food_reviews_bench` / `recall` (Fine Food–style `embedding` column).

use std::path::Path;

use ndarray::{Array2, ArrayView2, Axis};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Row width for the bundled Fine Food reviews embedding CSV.
pub const FINE_FOOD_EMBEDDING_DIM: usize = 1536;

/// Load all rows from `path`; each row’s **`embedding`** cell is a JSON array of `dim` floats.
pub fn load_embedding_csv(path: &Path, dim: usize) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();
    let emb_col = headers
        .iter()
        .position(|h| h == "embedding")
        .ok_or("CSV missing \"embedding\" column")?;

    let mut flat: Vec<f32> = Vec::new();

    for rec in rdr.records() {
        let rec = rec?;
        let cell = rec.get(emb_col).ok_or("row missing embedding cell")?;
        let v: Vec<f32> = serde_json::from_str(cell)?;
        if v.len() != dim {
            return Err(format!("embedding dim {} != expected {}", v.len(), dim).into());
        }
        flat.extend(v);
    }

    let n = flat.len() / dim;
    if n == 0 {
        return Err("no embedding rows loaded".into());
    }

    Ok(Array2::from_shape_vec((n, dim), flat)?)
}

pub fn l2_normalize_rows(a: &mut Array2<f32>) {
    for mut row in a.axis_iter_mut(Axis(0)) {
        let s = row.iter().fold(0f32, |acc, &x| acc + x * x).sqrt();
        let denom = s.max(f32::MIN_POSITIVE);
        row.iter_mut().for_each(|x| *x /= denom);
    }
}

pub fn gather_rows(embeddings: ArrayView2<f32>, indices: &[usize]) -> Array2<f32> {
    let dim = embeddings.ncols();
    let mut out = Array2::<f32>::zeros((indices.len(), dim));
    for (dst, &src_row) in indices.iter().enumerate() {
        let src = embeddings.index_axis(Axis(0), src_row);
        out.row_mut(dst).assign(&src);
    }
    out
}

/// Deterministic shuffle split: first block = train (DB), second = queries.
pub fn split_train_query(
    n: usize,
    train_fraction: f64,
    split_seed: u64,
) -> (Vec<usize>, Vec<usize>) {
    assert!(train_fraction > 0.0 && train_fraction < 1.0);
    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(&mut StdRng::seed_from_u64(split_seed));

    let mut n_db = ((n as f64) * train_fraction).floor() as usize;
    if n_db >= n {
        n_db = n.saturating_sub(1);
    }
    if n_db < 1 {
        n_db = 1;
    }

    let (db_slice, q_slice) = perm.split_at(n_db);
    (db_slice.to_vec(), q_slice.to_vec())
}
