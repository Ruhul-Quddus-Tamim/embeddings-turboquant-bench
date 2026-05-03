#!/usr/bin/env python3
"""
Speed benchmark: **GloVe 200 angular** HDF5 vs **FAISS PQ FastScan** (`turbovec.TurboQuantIndex`).

Uses the optional **`turbovec`** Python bindings (`TurboQuantIndex`), not `turboquant_index` itself (that crate has no PyO3 path from Python).

Default: `train[:100_000]` as database, `test[:1_000]` as queries (ANN-benchmarks layout).
Vectors are L2-normalized after load.

Same protocol as `benchmark_speed_food.py` (warmup, 5 runs, median ms/query, matched PQ budget).

Dependencies:

    pip install -r benchmarks/requirements-speed-glove.txt

Then build the TurboQuant Python extension (`maturin develop --release` in the **`turbovec-python`** directory at the workspace root — see comments in requirements-speed-food.txt).

Run:

    python benchmarks/benchmark_speed_glove200.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def load_glove_arrays(
    h5_path: Path,
    n_db: int,
    n_query: int,
) -> tuple[Any, Any, dict[str, Any]]:
    import h5py
    import numpy as np

    from benchmark_speed_common import l2_normalize_rows

    with h5py.File(h5_path, "r") as f:
        train = f["train"]
        test = f["test"]
        nt, dim = int(train.shape[0]), int(train.shape[1])
        nte, dtest = int(test.shape[0]), int(test.shape[1])
        if dim != dtest:
            raise SystemExit(f"train dim {dim} != test dim {dtest}")
        if n_db > nt:
            raise SystemExit(f"--n-db {n_db} > train rows {nt}")
        if n_query > nte:
            raise SystemExit(f"--n-query {n_query} > test rows {nte}")
        xb = np.asarray(train[:n_db], dtype=np.float32)
        xq = np.asarray(test[:n_query], dtype=np.float32)

    l2_normalize_rows(xb)
    l2_normalize_rows(xq)

    meta: dict[str, Any] = {
        "dataset": "glove-200-angular",
        "hdf5_train_rows": nt,
        "hdf5_test_rows": nte,
        "dim": dim,
        "n_db": n_db,
        "n_query": n_query,
        "slice": "train[:n_db], test[:n_query]",
    }
    return xb, xq, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="GloVe HDF5: TurboQuant vs FAISS FastScan")
    ap.add_argument(
        "hdf5",
        nargs="?",
        type=Path,
        default=Path("data/py-turboquant/glove-200-angular.hdf5"),
        help="glove-200-angular.hdf5 path",
    )
    ap.add_argument("--n-db", type=int, default=100_000, help="DB vectors (from train)")
    ap.add_argument("--n-query", type=int, default=1_000, help="queries (from test)")
    ap.add_argument("--bit-width", type=int, choices=(2, 4), default=4)
    ap.add_argument("--threading", choices=("st", "mt"), default="st")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", type=Path, help="JSON output path")
    args = ap.parse_args()

    bench_dir = Path(__file__).resolve().parent
    root = bench_dir.parent
    sys.path.insert(0, str(bench_dir))

    if args.threading == "st":
        os.environ["RAYON_NUM_THREADS"] = "1"

    from benchmark_speed_common import (
        arch_label,
        faiss_pq_fastscan_matched_bit_budget,
        import_faiss_and_turboquant_index,
    )

    faiss, TurboQuantIndex = import_faiss_and_turboquant_index()

    h5 = args.hdf5 if args.hdf5.is_absolute() else root / args.hdf5
    if not h5.is_file():
        raise SystemExit(
            f"HDF5 not found: {h5}\n"
            "Download with e.g.:\n"
            "  mkdir -p data/py-turboquant && curl -L -o data/py-turboquant/glove-200-angular.hdf5 "
            "http://ann-benchmarks.com/glove-200-angular.hdf5"
        )

    xb, xq, split_meta = load_glove_arrays(h5, args.n_db, args.n_query)
    dim = split_meta["dim"]
    n_db, n_q = int(xb.shape[0]), int(xq.shape[0])
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
    pq.search(xq[:1], k)
    faiss_times: list[float] = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        pq.search(xq, k)
        faiss_times.append((time.perf_counter() - t0) / n_q * 1000.0)
    faiss_ms = float(sorted(faiss_times)[median_i])

    result: dict[str, Any] = {
        "dataset_path": str(h5.resolve()),
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
        "note": "GloVe: train[:n_db], test[:n_query], L2-normalized; median ms/query; matched PQ budget.",
    }

    out = args.out
    if out is None:
        res_dir = bench_dir / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        out = res_dir / f"speed_glove200_{args.bit_width}bit_{arch_label()}_{args.threading}.json"
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nWrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
