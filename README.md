# Vector embeddings with TurboQuant indexing

Rust library that implements a **data-oblivious** compressed vector index: random orthogonal rotation, **Lloyd–Max scalar quantization** on Gaussian-marginal coordinates (high‑d approximation), bit-packed storage, and **LUT-based** approximate nearest-neighbor search.

**Benchmark meaning:** the binaries measure **approximate TurboQuant cosine top‑*k*** against **exact cosine top‑*k*** on **the identical embedding rows** rerendered as a **dense fp32 database** (Food dataset, train/query split). That is a **search / representation** comparison, not two different corpora. The library implements **MSE / `Q_mse`** ([`TurboQuantIndex::new`](turboquant_index/src/index.rs)) and **`Q_prod`** ([`TurboQuantIndex::new_prod`](turboquant_index/src/index.rs)) — `(b−1)`‑bit Lloyd–Max plus QJL on the residual per [TurboQuant](https://arxiv.org/abs/2504.19874) Algorithm 2.

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (stable), `cargo` on your `PATH`.

## Build and test

From the repository root:

```bash
cargo build --release -p turboquant_index
cargo test -p turboquant_index
```

## Using the library (Rust)

```rust
use ndarray::array;
use turboquant_index::TurboQuantIndex;
use turboquant_index::index::{SearchConfig, SearchObjective};
use turboquant_index::simd::Scorer;

let mut index = TurboQuantIndex::new(1536, 4, 42);
index.add_vectors(array![[0f32; 1536]].view()); // row-major f32, one row per vector

let hits = index.search_topk_indices(
    array![[0f32; 1536]].view(),
    10,
    SearchConfig {
        objective: SearchObjective::Cosine,
        scorer: Scorer::Wide,
    },
);
```

- **Quantizers:** [`TurboQuantIndex::new`](turboquant_index/src/index.rs) — `Q_mse` at **`bits`** per coordinate; [`TurboQuantIndex::new_prod`](turboquant_index/src/index.rs) — `Q_prod` at **`bits`** per coordinate (`bits ≥ 2`: MSE uses `bits − 1`, plus one QJL bit per coordinate and a stored residual norm). Snapshots v2 persist `Q_prod`’s extra Gaussian matrix **`S`**.
- **`SearchObjective::Cosine`**: scores approximate cosine similarity; **`InnerProduct`** scales by stored norms.
- **Disk**: `index.write_disk("snap.tq")` / `TurboQuantIndex::read_disk("snap.tq")`.

## Command-line tools

| Binary | Purpose |
|--------|--------|
| [`recall`](turboquant_index/src/bin/recall.rs) | Same Fine Food CSV + split as `food_reviews_bench`: **dense fp32 brute** vs **TurboQuant wide** batch times, **Recall@k**, SIMD sanity. Single **`bits`**; full JSON (**bytes**, scalar/wide timings, extrapolation): `food_reviews_bench`. |
| [`food_reviews_bench`](turboquant_index/src/bin/food_reviews_bench.rs) | Loads **CSV with JSON embeddings** (e.g. Fine Food 1k), reports **bytes**, **recall**, **timings**, and **dense brute vs TurboQuant** search batch times. |

### Recall

```bash
./benchmarks/run_recall.sh
# same defaults as food bench: csv k train_fraction split_seed index_seed lloyd_iterations bits
cargo run --release -p turboquant_index --bin recall -- \
  fine_food_reviews_with_embeddings_1k.csv 10 0.85 42 99 80 4
```

### Fine Food embeddings benchmark (CSV)

Place (or symlink) your CSV as `fine_food_reviews_with_embeddings_1k.csv` in the repo root, or pass a path as the first argument.

```bash
./benchmarks/run_food_bench.sh
# same as:
cargo run --release -p turboquant_index --bin food_reviews_bench -- \
  fine_food_reviews_with_embeddings_1k.csv
```

Output is printed to the terminal and saved under [`benchmarks/results/food_1k.json`](benchmarks/results/food_1k.json) (via `tee`).

**Optional positional arguments** (after the CSV path):

`k`, `train_fraction`, `split_seed`, `index_seed`, `lloyd_iterations`

Example: `k=10`, default train 85% / split seed 42 / index seed 99 / 80 Lloyd iterations:

```bash
cargo run --release -p turboquant_index --bin food_reviews_bench -- \
  fine_food_reviews_with_embeddings_1k.csv 10 0.85 42 99 80
```

The CSV must include an **`embedding`** column: a JSON array of **1536** floats per row (as in the OpenAI-style Fine Food reviews dataset).

---

## How to read the Fine Food benchmark JSON

The report is one JSON object. Think of it as answering **three questions**: *How much smaller?* *How accurate?* *How fast (on this machine)?*

### 1. Memory / storage (smaller footprints)

| Field | Plain language |
|--------|----------------|
| **`bytes_dense_db_vectors_only`** | Size of the **database vectors** if you stored them raw as **`f32`** (4 bytes × 1536 × number of DB rows). This is your “before” corpus size for vectors. |
| **`bytes_quantized_corpus_payload`** | “After” **per-vector compressed payload**: packed quantized codes **plus** one **`f32` norm** per vector. This is where TurboQuant saves space on the corpus. |
| **`compression_ratio_dense_over_quantized_corpus`** | **How many times larger** dense storage is versus this quantized payload (**higher = more shrink**). Example: ~**16×** at 2-bit, ~**8×** at 4-bit. |
| **`bytes.index_structure_total`** | **Everything** the running index holds: quantized payload **plus** the **rotation matrix Q** (~9 MB at *d*=1536) **plus** tiny quantization tables. For **small** *n*, **Q** can dominate, so **`compression_ratio_dense_db_vs_index_structure`** may be **&lt; 1** (total index larger than dense vectors alone). That does **not** contradict corpus compression—it means the **fixed** cost of **Q** is large when you only have ~850 vectors. |
| **`extrapolated_*_n10m`** | **Hypothetical** sizes at **10 million** DB vectors (good for “at scale” intuition). |

**Takeaway for non-experts:** the headline is usually **“we store the embedding table much smaller”** (`compression_ratio_dense_over_quantized_corpus`); the full index also includes a one-time **Q** that matters most when the collection is small.

### 2. Search quality (still finding the right neighbors?)

| Field | Plain language |
|--------|----------------|
| **`recall_at_k.mean` / `std_sample` / `min`** | Compares **approximate** top‑*k* (TurboQuant) to **exact** top‑*k* **cosine** on the **same** dense **f32** database. **Mean** ≈ average fraction of the true top‑*k* that the approximate list still contains. **~0.86** at 2-bit and **~0.96** at 4-bit (typical on this split) means “usually most of the top‑10 are still there”; 4-bit is closer to exact. |

### 3. Timings (M5 pro / this run only)

| Field | Plain language |
|--------|----------------|
| **`reference_search.brute_dense_cosine_topk_batch_ms`** | Time to run **exact** top‑*k* for **all** queries: full scan of the dense **f32** DB per query. |
| **`timings_ms.search_wide_batch`** / **`search_scalar_batch`** | Time for **all** queries using the **compressed** index (approximate). |
| **`speedup_wide_vs_brute_dense_batch`** | `brute_ms / turboquant_ms`. **&gt; 1** → TurboQuant batch was **faster** on that run; **&lt; 1** → dense brute was **faster**. On a **small** DB (~850 vectors), dense dot-products are very cheap, so values **below 1** are **common**—the benchmark is still useful to **compare** methods honestly. |

**Takeaway:** on this 1k-row setup, the **clearest win** to communicate is **smaller storage** (and good **recall** at 4-bit). **Speed** crosses over in favor of optimized compressed scan when **n** and **d** are large enough and the implementation is tuned (as in production-style FastScan paths); use larger *n* or repeat on your target hardware if “faster search” is the story you need.

---

## Repository layout

- [`turboquant_index/`](turboquant_index/) — library + binaries.
- [`benchmarks/run_food_bench.sh`](benchmarks/run_food_bench.sh) — Food CSV benchmark → [`benchmarks/results/food_1k.json`](benchmarks/results/food_1k.json).
- [`benchmarks/run_recall.sh`](benchmarks/run_recall.sh) — `recall` (dense vs TurboQuant on the same CSV split).

## Citation

- Zandieh et al., *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate* — [arXiv:2504.19874](https://arxiv.org/abs/2504.19874).
