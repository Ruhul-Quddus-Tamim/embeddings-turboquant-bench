#!/usr/bin/env python3
"""
Speed benchmark: **Fine Food** embeddings CSV vs **FAISS PQ FastScan** (`turbovec.TurboQuantIndex`).

This does **not** build or benchmark the Rust `turboquant_index` crate from Python — it uses **`turbovec`’s**
PyO3 bindings (the **`turbovec-python`** layout at this repo’s workspace root).

- **TurboQuant side**: `TurboQuantIndex` from the `turbovec` Python package (`maturin develop` there).
- **FAISS**: `IndexPQFastScan` with matched total bits per vector:
  - **4-bit** → `IndexPQFastScan(dim, m=dim, nbits=4)`
  - **2-bit** → `IndexPQFastScan(dim, m=dim//2, nbits=4)`
- **Timing**: warmup, then **5** timed batch searches, report **median** ms/query.

With `--split-json` from `export_csv_split`, the train/query split matches `food_reviews_bench`.

Setup: venv, `pip install maturin -r benchmarks/requirements-speed-food.txt`, then `maturin develop --release`
in **`turbovec-python`** from the workspace root.

Split JSON (optional):

    cargo run --release -p turboquant_index --bin export_csv_split -- \\
      fine_food_reviews_with_embeddings_1k.csv 0.85 42 > benchmarks/split.json

Run:

    python benchmarks/benchmark_speed_food.py --split-json benchmarks/split.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


def load_embedding_csv(path: Path, dim: int) -> Any:
    import numpy as np

    flat: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "embedding" not in (r.fieldnames or []):
            raise SystemExit('CSV must have an "embedding" column')
        for rec in r:
            cell = rec.get("embedding")
            if not cell:
                continue
            v = json.loads(cell)
            if len(v) != dim:
                raise SystemExit(f"embedding dim {len(v)} != {dim}")
            flat.extend(float(x) for x in v)
    if not flat:
        raise SystemExit("no rows loaded")
    return np.asarray(flat, dtype=np.float32).reshape(-1, dim)


def load_food_split(
    csv_path: Path,
    dim: int,
    split_json: Path | None,
    train_fraction: float,
    split_seed: int,
) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np

    from benchmark_speed_common import l2_normalize_rows

    emb = load_embedding_csv(csv_path, dim)
    n = emb.shape[0]
    if split_json and split_json.is_file():
        data = json.loads(split_json.read_text())
        db_ix = data["db_row_indices_global"]
        q_ix = data["query_row_indices_global"]
        meta = {
            "train_fraction": data.get("train_fraction", train_fraction),
            "split_seed": data.get("split_seed", split_seed),
            "n_total": int(data["n_total"]),
            "source": "export_csv_split",
        }
    else:
        rng = np.random.default_rng(split_seed)
        perm = rng.permutation(n).tolist()
        n_db = max(1, min(int(math.floor(n * train_fraction)), n - 1))
        db_ix = perm[:n_db]
        q_ix = perm[n_db:]
        meta = {
            "train_fraction": train_fraction,
            "split_seed": split_seed,
            "n_total": n,
            "source": "numpy_random_permutation",
        }
    xb = emb[np.asarray(db_ix, dtype=np.int64)]
    xq = emb[np.asarray(q_ix, dtype=np.int64)]
    l2_normalize_rows(xb)
    l2_normalize_rows(xq)
    meta["n_db"] = int(xb.shape[0])
    meta["n_query"] = int(xq.shape[0])
    return xb, xq, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Food CSV: TurboQuant vs FAISS FastScan latency")
    ap.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=Path("fine_food_reviews_with_embeddings_1k.csv"),
        help="CSV with embedding column",
    )
    ap.add_argument("--split-json", type=Path, help="From export_csv_split")
    ap.add_argument("--train-fraction", type=float, default=0.85)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--bit-width", type=int, choices=(2, 4), default=4)
    ap.add_argument("--threading", choices=("st", "mt"), default="st")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--runs", type=int, default=5, help="Timed repetitions (median = middle index)")
    ap.add_argument(
        "--out",
        type=Path,
        help="JSON output (default: benchmarks/results/speed_food_<bits>bit_<arch>_<threading>.json)",
    )
    args = ap.parse_args()

    bench_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(bench_dir))

    dim = 1536
    if args.threading == "st":
        os.environ["RAYON_NUM_THREADS"] = "1"

    from benchmark_speed_common import (
        arch_label,
        faiss_pq_fastscan_matched_bit_budget,
        import_faiss_and_turboquant_index,
    )

    faiss, TurboQuantIndex = import_faiss_and_turboquant_index()

    root = Path(__file__).resolve().parents[1]
    csv_path = args.csv if args.csv.is_absolute() else root / args.csv
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    split_json = args.split_json
    if split_json and not split_json.is_file():
        raise SystemExit(f"split-json not found: {split_json}")

    xb, xq, split_meta = load_food_split(
        csv_path,
        dim,
        split_json,
        args.train_fraction,
        args.split_seed,
    )
    n_db, n_q = xb.shape[0], xq.shape[0]
    if n_q < 1:
        raise SystemExit("no query rows after split; lower --train-fraction or use more CSV rows")
    k = min(args.k, n_db)
    if k < 1:
        raise SystemExit("k must be >= 1")

    if args.threading == "st":
        faiss.omp_set_num_threads(1)
    else:
        try:
            faiss.omp_set_num_threads(0)
        except Exception:
            pass

    median_i = min(args.runs // 2, args.runs - 1)

    tq = TurboQuantIndex(dim=dim, bit_width=args.bit_width)
    tq.add(xb)
    tq.search(xq[:1], k=k)
    tq_times: list[float] = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        tq.search(xq, k=k)
        tq_times.append((time.perf_counter() - t0) / n_q * 1000.0)
    tq_ms = float(sorted(tq_times)[median_i])

    pq, faiss_label = faiss_pq_fastscan_matched_bit_budget(dim, args.bit_width)
    pq.train(xb)
    pq.add(xb)
    pq.search(xq[: min(1, n_q)], k)
    faiss_times: list[float] = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        pq.search(xq, k)
        faiss_times.append((time.perf_counter() - t0) / n_q * 1000.0)
    faiss_ms = float(sorted(faiss_times)[median_i])

    result: dict[str, Any] = {
        "dataset_path": str(csv_path.resolve()),
        "dim": dim,
        "bit_width": args.bit_width,
        "arch": arch_label(),
        "threading": args.threading,
        "k": k,
        "split": split_meta,
        "tq_ms_per_query": round(tq_ms, 3),
        "faiss_ms_per_query": round(faiss_ms, 3),
        "faiss_pq_backend": faiss_label,
        "median_run_index": median_i,
        "n_timed_runs": args.runs,
        "note": "Median ms/query; FAISS PQ layout matched to TurboQuant bit budget.",
    }

    out = args.out
    if out is None:
        res_dir = Path(__file__).resolve().parent / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        out = res_dir / f"speed_food_{args.bit_width}bit_{arch_label()}_{args.threading}.json"
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nWrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
