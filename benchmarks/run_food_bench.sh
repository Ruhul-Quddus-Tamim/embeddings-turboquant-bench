#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CSV="${1:-fine_food_reviews_with_embeddings_1k.csv}"
shift || true
mkdir -p benchmarks/results
cargo run --release -p turboquant_index --bin food_reviews_bench -- "$CSV" "$@" | tee benchmarks/results/food_1k.json
