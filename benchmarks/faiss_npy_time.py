#!/usr/bin/env python3.10
"""FAISS PQ FastScan timing on two float32 `.npy` matrices (reference protocol).

Matches `benchmark_speed_common.faiss_pq_fastscan_matched_bit_budget`: same bits/vector as TQ `bit_width`.

Stdout: one JSON object, e.g. {"faiss_ms_per_query": 1.23, "faiss_pq_backend": "..."}

Usage:
  python3.10 benchmarks/faiss_npy_time.py train.npy test.npy <k> <bit_width> <runs> [st|mt]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


def l2_normalize_rows(x: Any) -> None:
    import numpy as np

    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(nrm, np.float32(1e-12), out=nrm)
    x[:] = (x / nrm).astype(np.float32)


def faiss_pq_fastscan_matched_bit_budget(dim: int, tq_bit_width: int) -> tuple[Any, str]:
    import faiss

    if tq_bit_width == 4:
        m, nbits = dim, 4
    elif tq_bit_width == 2:
        m, nbits = dim // 2, 4
    else:
        raise SystemExit("bit_width must be 2 or 4")

    if dim % m != 0:
        raise SystemExit(f"dim={dim} not divisible by m={m}")

    try:
        idx = faiss.IndexPQFastScan(dim, m, nbits)  # type: ignore[misc]
        return idx, f"IndexPQFastScan(m={m},nbits={nbits})"
    except Exception:
        pass
    desc = f"PQ{m}x{nbits}fs"
    try:
        idx = faiss.index_factory(dim, desc, faiss.METRIC_L2)
        return idx, f"index_factory({desc},L2)"
    except Exception as e:
        raise SystemExit(
            f"FastScan PQ not available (need faiss with PQFastScan). Last error: {e!r}"
        ) from e


def main() -> None:
    if len(sys.argv) < 6:
        raise SystemExit(
            "usage: faiss_npy_time.py train.npy test.npy k bit_width runs [st|mt]"
        )

    import numpy as np

    try:
        import faiss
    except ModuleNotFoundError as e:
        raise SystemExit(f"faiss import failed: {e}") from e

    train_p = Path(sys.argv[1])
    test_p = Path(sys.argv[2])
    k = int(sys.argv[3])
    bit_width = int(sys.argv[4])
    runs = int(sys.argv[5])
    threading = sys.argv[6] if len(sys.argv) > 6 else "st"

    if bit_width not in (2, 4):
        raise SystemExit("bit_width must be 2 or 4")
    if runs < 1:
        raise SystemExit("runs must be >= 1")

    xb = np.load(str(train_p)).astype(np.float32, copy=False)
    xq = np.load(str(test_p)).astype(np.float32, copy=False)
    if xb.ndim != 2 or xq.ndim != 2:
        raise SystemExit("train and test must be 2D arrays")
    dim = int(xb.shape[1])
    if int(xq.shape[1]) != dim:
        raise SystemExit("train/test dim mismatch")
    n_db = int(xb.shape[0])
    n_q = int(xq.shape[0])
    k = min(k, n_db)
    if k < 1 or n_q < 1:
        raise SystemExit("invalid k or empty queries")

    l2_normalize_rows(xb)
    l2_normalize_rows(xq)

    if threading == "st":
        faiss.omp_set_num_threads(1)
    else:
        try:
            faiss.omp_set_num_threads(0)
        except Exception:
            pass

    pq, faiss_label = faiss_pq_fastscan_matched_bit_budget(dim, bit_width)
    pq.train(xb)
    pq.add(xb)
    pq.search(xq[:1], k)

    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        pq.search(xq, k)
        samples.append((time.perf_counter() - t0) / n_q * 1000.0)

    samples.sort()
    median_i = min(runs // 2, runs - 1)
    out = {
        "faiss_ms_per_query": round(float(samples[median_i]), 3),
        "faiss_pq_backend": faiss_label,
        "median_run_index": median_i,
        "n_timed_runs": runs,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
