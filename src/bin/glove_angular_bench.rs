//! **GloVe 200 angular** (and similar ANN layouts): row-major **`f32`** `.npy` matrices, L2-normalized.
//!
//! - **Recall / batch timings**: same stderr report as `recall` (TurboQuant vs exact cosine).
//! - **Reference-style latency** (Python benchmarks): **median ms/query** after warmup, **`tq_ms_per_query`**, and **`faiss_ms_per_query`** via FAISS PQ FastScan (`benchmarks/faiss_npy_time.py`, needs **`python3.10`** on `PATH` + `faiss-cpu`).
//!
//! Pass **`--no-faiss`** to skip FAISS (TQ + recall only).
//!
//! ```text
//! cargo run --release -p turboquant_index --bin glove_angular_bench -- \\
//!   data/glove_train.npy data/glove_test.npy [k=64] [index_seed=99] [lloyd_iters=80] [bits=4]
//! ```
//!
//! Default **`k=64`** matches [`benchmarks/benchmark_speed_glove200.py`]. Use **~100k×dim train / 1k×dim test**
//! (GloVe slice) for headline comparisons; with a **small** `n_db`, FAISS IndexPQFastScan often wins on wall-clock anyway.

use std::env;
use std::path::{Path, PathBuf};

use ndarray::{Array2, Axis};
use ndarray_npy::read_npy;

use turboquant_index::recall_report::report_recall_for_index;
use turboquant_index::latency::{
    eprint_if_turboquant_slower_than_faiss, median_tq_ms_per_query, try_faiss_median_ms,
};
use turboquant_index::TurboQuantIndex;

fn arg_usize(args: &[String], i: usize, default: usize) -> usize {
    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn arg_u64(args: &[String], i: usize, default: u64) -> u64 {
    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn arg_u8(args: &[String], i: usize, default: u8) -> u8 {
    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn l2_normalize_rows(a: &mut Array2<f32>) {
    for mut row in a.axis_iter_mut(Axis(0)) {
        let s = row.iter().fold(0f32, |acc, &x| acc + x * x).sqrt();
        let denom = s.max(f32::MIN_POSITIVE);
        row.iter_mut().for_each(|x| *x /= denom);
    }
}

/// Resolve `p` if relative: try cwd first, then walk up to ancestor dirs (finds `repo/data/…` when cwd is a subdir).
fn resolve_existing_file(p: &Path) -> Option<PathBuf> {
    if p.is_absolute() {
        return p.is_file().then(|| p.to_path_buf());
    }
    let mut dir = env::current_dir().ok()?;
    for _ in 0..12 {
        let cand = dir.join(p);
        if cand.is_file() {
            return Some(cand);
        }
        if !dir.pop() {
            break;
        }
    }
    None
}

fn load_npy_matrix(path: &Path) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let arr: Array2<f32> = read_npy(path)?;
    Ok(arr)
}

fn missing_npy_message(label: &str, p: &Path) -> String {
    let cwd = env::current_dir()
        .map(|d| d.display().to_string())
        .unwrap_or_else(|_| "(unknown)".to_string());
    format!(
        "{label} .npy not found: {}\n\
         Current directory: {cwd}\n\
         Paths are relative to the process cwd unless absolute. The file must exist before running.\n\
         Example (from a HDF5 with `train` / `test`): save float32 2D arrays as .npy, e.g. first 100k train rows and 1k test rows.",
        p.display()
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env::set_var("RAYON_NUM_THREADS", "1");

    let mut raw: Vec<String> = env::args().skip(1).collect();
    let skip_faiss = raw.iter().any(|s| s == "--no-faiss");
    raw.retain(|s| s != "--no-faiss");
    let positional: Vec<String> = raw.into_iter().filter(|s| !s.starts_with('-')).collect();

    if positional.len() < 2 {
        eprintln!(
            "Usage: glove_angular_bench [--no-faiss] <train.npy> <test.npy> [k=64] [index_seed=99] [lloyd_iters=80] [bits=4]\n\
             train/test: 2D float32 row-major. Default k=64 matches benchmarks/benchmark_speed_glove200.py.\n\
             Needs python3.10 + faiss-cpu for faiss_ms_per_query (see benchmarks/faiss_npy_time.py)."
        );
        return Ok(());
    }

    let train_arg = Path::new(&positional[0]);
    let test_arg = Path::new(&positional[1]);
    let Some(train_path) = resolve_existing_file(train_arg) else {
        return Err(missing_npy_message("train", train_arg).into());
    };
    let Some(test_path) = resolve_existing_file(test_arg) else {
        return Err(missing_npy_message("test", test_arg).into());
    };

    let k = arg_usize(&positional, 2, 64);
    let index_seed = arg_u64(&positional, 3, 99);
    let lloyd_iters = arg_usize(&positional, 4, 80);
    let bits = arg_u8(&positional, 5, 4);

    if k < 1 {
        return Err("k must be >= 1".into());
    }
    if bits != 2 && bits != 4 {
        return Err("bits must be 2 or 4 (FAISS comparison uses matched PQ budget)".into());
    }

    let mut db = load_npy_matrix(&train_path)?;
    let mut queries = load_npy_matrix(&test_path)?;
    if db.ncols() != queries.ncols() {
        return Err(format!(
            "train dim {} != test dim {}",
            db.ncols(),
            queries.ncols()
        )
        .into());
    }
    let dim = db.ncols();
    let n_db = db.nrows();
    let n_query = queries.nrows();
    if n_db < 1 || n_query < 1 {
        return Err("train and test must have at least one row".into());
    }
    if k > n_db {
        return Err(format!("k={k} > n_db={n_db}").into());
    }

    const REFERENCE_N_DB: usize = 100_000;
    if n_db < REFERENCE_N_DB {
        eprintln!(
            "note: benchmark_speed_glove200.py uses train[:{REFERENCE_N_DB}] and test[:1000] by default.\n\
             With n_db={n_db}, FAISS IndexPQFastScan is often **faster** than TurboQuant on wall-clock; \
             use a ~{REFERENCE_N_DB}-row train slice for the same regime as the Python speed reference."
        );
    }

    l2_normalize_rows(&mut db);
    l2_normalize_rows(&mut queries);

    let label = format!(
        "{} + {} ({}×{} train, {}×{} test, angular)",
        train_path.display(),
        test_path.display(),
        n_db,
        dim,
        n_query,
        dim,
    );

    let timed_runs = 5usize;
    let mut idx = TurboQuantIndex::with_lloyd_iterations(dim, bits, index_seed, lloyd_iters);
    idx.add_vectors(db.view());

    let tq_ms = median_tq_ms_per_query(&idx, queries.view(), k, n_query, timed_runs);
    eprintln!(
        "speed_median_ms_per_query (warmup + {timed_runs} timed runs, RAYON_NUM_THREADS=1): tq_ms_per_query={tq_ms:.4}",
    );
    if !skip_faiss {
        match try_faiss_median_ms(&train_path, &test_path, k, bits, timed_runs) {
            Ok(f) => {
                eprintln!(
                    "faiss_ms_per_query={:.4}  backend={}  (same protocol as benchmarks/benchmark_speed_glove200.py)",
                    f.faiss_ms_per_query,
                    f.faiss_pq_backend,
                );
                eprint_if_turboquant_slower_than_faiss(tq_ms, f.faiss_ms_per_query);
            }
            Err(e) => {
                eprintln!("faiss_ms_per_query=(skipped)  reason: {e}");
            }
        }
    } else {
        eprintln!("faiss_ms_per_query=(skipped)  --no-faiss");
    }

    report_recall_for_index(&idx, db.view(), queries.view(), k, &label);

    Ok(())
}
