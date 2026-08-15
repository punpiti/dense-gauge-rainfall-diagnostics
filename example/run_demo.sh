#!/usr/bin/env bash
set -euo pipefail

example_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_dir="$(cd "$example_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
generated="$example_dir/generated"
demo_output="$example_dir/demo_output"

echo "[demo] Creating the two required synthetic input files"
"$python_bin" "$example_dir/make_synthetic_inputs.py"

PYTHON_BIN="$python_bin" bash "$example_dir/run_pipeline.sh" \
  --rainfall "$generated/rainfall.parquet" \
  --stations "$generated/stations.csv" \
  --output-dir "$demo_output" \
  --event-id SYNTHETIC_EVENT
