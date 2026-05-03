# Vector embeddings with TurboQuant

Rust library that implements a **compressed** vector index: random orthogonal rotation, **Lloyd–Max scalar quantization** on Gaussian‑marginal coordinates, bit‑packed codes, and **LUT‑based** approximate top‑**k** search.

## TurboQuant Findings

Here is what matters for **vector search** in ordinary terms: **how small** the embeddings can get, **how fast** queries run versus brute force, whether **answers stay good**, and what you need to **deploy**.

| Topic | What TurboQuant buys you |
| --- | --- |
| **Vectors use less space** | The **corpus is not stored as raw float32 rows forever**. Packing many bits per scalar into compact codes (**plus one norm per vector**) cuts **disk and RAM bandwidth** versus a dense embedding table - that is direct savings on replication, backups, and memory‑mapped serving. One fixed overhead is the **rotation matrix** (scaled by embedding **dimension squared**, not row count): with **few** vectors, total index bytes can exceed “vectors only”; with **many** rows, corpus compression dominates. **`compression_ratio_*`** in the Fine Food JSON makes that tangible. |
| **Search stays “good enough”** | Search is **approximate**: benchmarks report **Recall@k**—how much overlap there is with **exact cosine** top‑**k** on the **same** original vectors. That answers “if I compress, do I still get almost the right neighbors?” **More bits per coordinate → higher recall and larger payloads**; fewer bits → smaller but riskier. |
| **Fast implementation** | Where reported, **wide SIMD vs scalar** paths are cross‑checked (`max_abs_diff` on stderr), so aggressive vectorization is not a silent fork from the reference scoring you compare against exact search. |
| **Honest speed comparisons** | Timings compare **TurboQuant** against **full linear scan** on identical data and, optionally, **FAISS product quantization** with a **matched bit budget**—so speed stories are not hand‑waved against unrelated systems. |
| **Operational fit** | **Save / load** indexes from disk, optional **FAISS reference timing** via [`benchmarks/faiss_npy_time.py`](benchmarks/faiss_npy_time.py), and [`latency`](turboquant_index/src/latency.rs) helpers for **median ms/query**—useful for regression testing **latency and behavior** on real hardware. |
| **The levers you turn** | **Bits per coordinate** is the main **speed / size / recall** knob. **Training iterations and seeds** affect build time and fine centroid details. **Cosine vs inner product** matches how your embedding model was trained or normalized. |


### Where each benefit shows up: [`benchmarks/results/food_1k.json`](benchmarks/results/food_1k.json)

Each row below points to **the same artifact** (`food_reviews_bench` output). Figures are taken from **this repo’s checked‑in snapshot** (**850 vectors × 1536 dim**, **`k`** = split / query counts in **`split`**).

| Benefit | Field(s) to open | What the numbers mean (this file) |
| --- | --- | --- |
| **Smaller embedding table than raw floats** | **`bytes_dense_db_vectors_only`**; **`configs[].bytes_quantized_corpus_payload`**, **`compression_ratio_dense_over_quantized_corpus`**, **`configs[].bytes.index_structure_total`** | **≈ 5.2 MB** (**5 224 800** bytes) if every database vector were stored as **float32** (dense only). **Quantized payload:** **≈ 0.33 MB** at **2 bits/coordinate**, **≈ 0.66 MB** at **4 bits/coordinate**. **Versus dense:** dense is **≈ 16×** larger than the packed payload at 2 bit and **≈ 8×** larger at 4 bit. **Total index footprint** includes rotation **Q** (**~9 MB**), so **dense vectors only** can still be **smaller than the whole index** (ratio **under 1** on this snapshot)—the **vector table** shrinks, but **total running index** is still **dominated by Q** on this small corpus. |
| **Faster batch search vs exact cosine brute force** | **`reference_search.brute_dense_cosine_topk_batch_ms`**, each **`configs[].timings_ms`**, **`speedup_wide_vs_brute_dense_batch`** | Exact linear scan (**all queries**): **`brute_dense_cosine_topk_batch_ms` ≈ 132 ms**. TurboQuant **wide path** (**`timings_ms.search_wide_batch`**): **≈ 33 ms** (**bits=2**) and **≈ 39 ms** (**bits=4**). **Speedup columns** (**`speedup_wide_vs_brute_dense_batch`**): **≈ 4.0×** and **≈ 3.4×** (**`brute_ms / turboquant_wide_ms`** &gt; 1 ⇒ compressed search wins this run). **`search_scalar_batch`** shows the slower non‑wide scorer for contrast. *(Timings drift by machine—rerun bench to refresh.)* |
| **Quality of approximate neighbors** | Each **`configs[].recall_at_k`** | **`mean`** answers “what fraction of the true top‑**k** cosine neighbors stayed in TurboQuant’s list?” — here **≈ 0.86** @10 for **bits=2**, **≈ 0.96** for **bits=4** (**`configs[0].recall_at_k.mean`**, **`configs[1].…`**). **`min`** shows the worst query in the sample run. More **bits** ⇒ higher **`mean`** and larger **`bytes_quantized_corpus_payload`**. |
| **How big things get at scale** | **`extrapolated_*_n10m`** inside each **`configs[]`** | Not measured on real **10M rows** - those fields are **rough projections from formulas**, so treat them as **ballpark intuition only**, not benchmarks. **`disk_snapshot_bytes_tempfile`** is an actual measured **`.tq`**‑style save size from one run on disk (still depends on toolchain and snapshot format). |

**How to navigate the file:** top level describes **dataset and split**. **`configs`** is length **two**: element **`bits: 2`**, element **`bits: 4`** - same run, **two quantization budgets**. **`reference_search`** and **`split`** are shared across both entries.

*(**`max_abs_diff`** wide vs scalar and optional **`tq_ms_per_query` / `faiss_ms_per_query`** appear on **`stderr`** from the bench; they are often **not duplicated** inside this JSON—only what you see serialized here.)*

---

## Experiments

### Fine Food

```bash
cargo run --release -p turboquant_index --bin food_reviews_bench -- \
  fine_food_reviews_with_embeddings_1k.csv
```

```text
- Note: Fine Food split uses n_db=850; FAISS vs TQ ordering here is for matched protocol, not the large-corpus.

- speed_median_ms_per_query: tq_ms_per_query=0.0870  (bits=4)

- faiss_ms_per_query=0.0760  backend=IndexPQFastScan(m=1536,nbits=4)

- dataset=fine_food_reviews_with_embeddings_1k.csv (850×1536 train, 150×1536 query, angular, bits=4)

- dim=1536 bits=4 n_db=850 n_queries=150 k=10 index_seed=99 lloyd_iters=80

- search_batch_ms: brute_dense_cosine(top-k all queries)=195.4323 turboquant_wide(top-k all queries)=13.3716

- speedup_wide_vs_brute_dense_batch=14.6155 (>1 ⇒ TurboQuant wide batch faster on this machine(M3 Pro); often <1 for small n_db)

- mean Recall@10 (overlap with exact cosine brute force): 0.9440

- max_abs_diff(scalar SIMD vs wide SIMD one query scores): 0.000001

=== latency (bits=4): tq_ms_per_query=0.0870  faiss_ms_per_query=0.0760 ===
```

### GloVe

```bash
cargo run --release -p turboquant_index --bin glove_angular_bench -- \
  data/glove_train.npy data/glove_test.npy
```

```text
- speed_median_ms_per_query: tq_ms_per_query=0.5006

- faiss_ms_per_query=0.5030  backend=IndexPQFastScan(m=200,nbits=4)
```

---

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (stable), `cargo` on your `PATH`.
- Optional **FAISS reference timing (`faiss_ms_per_query`):** **`python3.10`** and **`faiss-cpu`** (see [`benchmarks/faiss_npy_time.py`](benchmarks/faiss_npy_time.py) / requirements under [`benchmarks/`](benchmarks/)).

## Build and test

From the repository root:

```bash
cargo build --release -p turboquant_index
cargo test -p turboquant_index
```

## Using the library (Rust)

Queries passed to **`search_topk_indices`** must be **contiguous row‑major**.

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

- **Quantizers:** [`TurboQuantIndex::new`](turboquant_index/src/index.rs) — `Q_mse` at **`bits`** per coordinate; [`TurboQuantIndex::new_prod`](turboquant_index/src/index.rs) — `Q_prod`; snapshots v2 persist `Q_prod`’s **`S`** matrix.
- **`SearchObjective::Cosine`** vs **`InnerProduct`** (scaling by stored norms).
- **Disk:** `index.write_disk("snap.tq")` / `TurboQuantIndex::read_disk("snap.tq")`.
- **`turboquant_index::latency`:** median **`tq`** ms/query and optional **`faiss_ms_per_query`** subprocess (same bits / protocol intent as benches).

## Command-line tools

| Binary | Purpose |
| --- | --- |
| [`recall`](turboquant_index/src/bin/recall.rs) | Fine Food‑style CSV: dense brute vs TurboQuant, Recall@**k**. |
| [`food_reviews_bench`](turboquant_index/src/bin/food_reviews_bench.rs) | CSV **embedding** column: **bytes**, recall (**bits 2 & 4**), timings, extrapolation JSON. |
| [`glove_angular_bench`](turboquant_index/src/bin/glove_angular_bench.rs) | GloVe‑style **`f32` `.npy`**: **`tq`** / optional FAISS ms/query **+** Recall block. |

### Recall

```bash
./benchmarks/run_recall.sh
# explicit:
cargo run --release -p turboquant_index --bin recall -- \
  fine_food_reviews_with_embeddings_1k.csv 10 0.85 42 99 80 4
```

### Fine Food embeddings benchmark (CSV)

Place (or symlink) your CSV as `fine_food_reviews_with_embeddings_1k.csv` in the repo root, or pass a path.

```bash
./benchmarks/run_food_bench.sh
# same as:
cargo run --release -p turboquant_index --bin food_reviews_bench -- \
  fine_food_reviews_with_embeddings_1k.csv
```

Output to [`benchmarks/results/food_1k.json`](benchmarks/results/food_1k.json).

**Optional positional arguments** after the CSV path: **`k`, `train_fraction`, `split_seed`, `index_seed`, `lloyd_iterations`.**

The CSV must include an **`embedding`** column: a JSON array of **`1536`** floats per row (**OpenAI‑style Fine Food** layout).

Pass **`--no-faiss`** to skip the FAISS subprocess (**`food_reviews_bench`**).

---

## How to read the Fine Food benchmark JSON

The JSON answers **three questions:** *How much smaller?* *How faithful?* *How fast vs dense brute vs FAISS (optional)?*

### 1. Memory / storage

| Field | Meaning |
| --- | --- |
| **`bytes_dense_db_vectors_only`** | Raw **`f32`** corpus table size (**4 × dim × `n_db`**). |
| **`bytes_quantized_corpus_payload`** | Packed codes **`+`** one **`f32` norm**/vector. |
| **`compression_ratio_dense_over_quantized_corpus`** | How many × larger dense is versus this quantized payload. |
| **`bytes.index_structure_total`** | Corpus payload **`+`** rotation **`Q`** **`+`** centroids/tables — for **tiny `n`**, **`Q`** can dominate so **dense‑vs‑index** ratio &lt; 1 does **not** contradict corpus shrink. |
| **`extrapolated_*_n10m`** | Back‑of‑envelope at **`n = 10M`**. |

### 2. Search quality (`recall_at_k`)

Compared to **exact cosine** top‑**k** on the **same dense** rows: **`mean`** is average overlap of approximate vs ground‑truth **`k`**; higher **bits** ⇒ higher **`mean`** (see entries for **`bits: 2`** vs **`bits: 4`** in **`configs`**).

### 3. Timings (`timings_ms`, `speedup_*`, optional `reference_latency_ms_per_query`)

| Field | Plain language |
| --- | --- |
| **`latency_ms_per_query`** | **`tq_ms_per_query`** vs optional **`faiss_ms_per_query`** (median batches; **Warmup + 5 runs** mirror [`latency`](turboquant_index/src/latency.rs)). |
| **`search.brute_dense_cosine_*`** | Exact brute top‑**k** for **all** queries. |
| **`timings_ms.search_wide_batch`** / **`search_scalar_batch`** | Compressed index, all queries — **`speedup_*`** vs brute. |

**Batch speedup ≠ median ms/query parity with FAISS:** batch timing is TurboQuant search wall-clock for **all queries** divided by **`n_q`**; **`ms/query`** also reflects **TQ query-side preprocessing** versus **PQ search-only timing** (see caveat [above](#turboquant-findings)).

---

## Repository layout

- [`turboquant_index/src/latency.rs`](turboquant_index/src/latency.rs) — median **`tq`** + optional **`faiss`** subprocess timings.
- [`benchmarks/run_food_bench.sh`](benchmarks/run_food_bench.sh) → [`benchmarks/results/food_1k.json`](benchmarks/results/food_1k.json).
- [`benchmarks/run_recall.sh`](benchmarks/run_recall.sh) — **`recall`**.

## Citation

- Zandieh et al., *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate* — [arXiv:2504.19874](https://arxiv.org/abs/2504.19874).
