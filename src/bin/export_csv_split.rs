//! Emit JSON with **global** CSV row indices for DB vs queries — same split as `food_reviews_bench`
//! (`csv_dataset::split_train_query`). Use with `benchmarks/benchmark_speed_food.py --split-json`.

use std::env;
use std::path::PathBuf;

use turboquant_index::csv_dataset::{self, FINE_FOOD_EMBEDDING_DIM};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let csv_path = PathBuf::from(
        args.get(1)
            .map(String::as_str)
            .unwrap_or("fine_food_reviews_with_embeddings_1k.csv"),
    );
    let train_fraction = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.85_f64);
    let split_seed = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(42_u64);

    let emb = csv_dataset::load_embedding_csv(&csv_path, FINE_FOOD_EMBEDDING_DIM)?;
    let n = emb.nrows();
    let (db_ix, q_ix) = csv_dataset::split_train_query(n, train_fraction, split_seed);

    let out = serde_json::json!({
        "dataset_path": csv_path.display().to_string(),
        "dim": FINE_FOOD_EMBEDDING_DIM,
        "n_total": n,
        "train_fraction": train_fraction,
        "split_seed": split_seed,
        "db_row_indices_global": db_ix,
        "query_row_indices_global": q_ix,
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
