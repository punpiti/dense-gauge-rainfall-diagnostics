#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash example/run_pipeline.sh \
    --rainfall PATH/TO/rainfall.parquet \
    --stations PATH/TO/stations.csv \
    --output-dir PATH/TO/output \
    [--event-id EVENT_ID] [--boundary PATH/TO/boundary.geojson]

Required scientific inputs:
  --rainfall   One post-QA/QC event-window rainfall table (Parquet).
  --stations   Station metadata with id, nameShort, lat, and lon (CSV).

The optional boundary changes only the analysis domain. When omitted, the
released gridding code derives a rectangular domain from station extent.
EOF
}

example_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_dir="$(cd "$example_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

rainfall=""
stations=""
output_dir=""
event_id="EVENT"
boundary=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rainfall) rainfall="$2"; shift 2 ;;
    --stations) stations="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --event-id) event_id="$2"; shift 2 ;;
    --boundary) boundary="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$rainfall" || -z "$stations" || -z "$output_dir" ]]; then
  echo "ERROR: --rainfall, --stations, and --output-dir are required." >&2
  usage >&2
  exit 2
fi
if [[ ! -f "$rainfall" ]]; then
  echo "ERROR: rainfall input not found: $rainfall" >&2
  exit 2
fi
if [[ ! -f "$stations" ]]; then
  echo "ERROR: station metadata not found: $stations" >&2
  exit 2
fi

mkdir -p "$output_dir"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$output_dir/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"

grid_command=(
  "$python_bin" "$package_dir/code/build_event_grid_xarray.py"
  --cache "$rainfall"
  --stations "$stations"
  --grid-km 1.0 --pad-km 2.0
  --smooth --sigma-km 2.5 --cutoff-sigma 3.0
  --no-anchor-stations --only-influenced --min-rain 0.1
  --out "$output_dir/event_grid.nc" --overwrite
)
if [[ -n "$boundary" ]]; then
  grid_command+=(--boundary "$boundary")
fi

echo "Inputs"
echo "  rainfall observations : $rainfall"
echo "  station metadata      : $stations"
echo "  event identifier      : $event_id"
echo "  output directory      : $output_dir"
echo

echo "[1/6] Reconstructing the gridded rainfall field"
"${grid_command[@]}"

echo "[2/6] Computing optical flow, advection-only reference, and residual"
"$python_bin" "$package_dir/code/compute_optical_flow_xarray.py" \
  --in "$output_dir/event_grid.nc" \
  --out "$output_dir/flow_residual.nc" \
  --method farneback --rain-threshold 0.1 --dt-min 15 --overwrite

echo "[3/6] Extracting frame- and event-level residual metrics"
"$python_bin" "$package_dir/code/classify_convective_from_residual.py" \
  --flow "$output_dir/flow_residual.nc" \
  --grid "$output_dir/event_grid.nc" \
  --event-id "$event_id" \
  --out-summary "$output_dir/intermediate_storm_metrics.csv" \
  --out-frames "$output_dir/frame_metrics.csv"

echo "[4/6] Mapping extracted metrics to the final-rule column names"
"$python_bin" "$example_dir/prepare_final_rule_input.py" \
  --input "$output_dir/intermediate_storm_metrics.csv" \
  --output "$output_dir/storm_metrics_for_final_rule.csv"

echo "[5/6] Applying the manuscript's final operational rule"
"$python_bin" "$package_dir/code/apply_convective_thresholds.py" \
  --input "$output_dir/storm_metrics_for_final_rule.csv" \
  --outdir "$output_dir/final_label" \
  --area-threshold 500 --peak-sum-threshold 1000 \
  --event-sum-threshold 4000 --frames-threshold 10 --mode strict

echo "[6/6] Rendering the visual summary"
"$python_bin" "$example_dir/make_demo_visual.py" \
  --grid "$output_dir/event_grid.nc" \
  --flow "$output_dir/flow_residual.nc" \
  --result "$output_dir/final_label/all_storm_results_labeled.csv" \
  --output "$output_dir/summary.png"

echo
echo "Final event result"
"$python_bin" -c "import pandas as pd; print(pd.read_csv(r'$output_dir/final_label/all_storm_results_labeled.csv')[['storm_id','class_peak_pos_area_cells','class_peak_pos_sum','class_event_pos_sum','class_n_strong_frames','convective_rule_label']].to_string(index=False))"
echo
echo "Main outputs"
echo "  final label and metrics : $output_dir/final_label/all_storm_results_labeled.csv"
echo "  visual summary          : $output_dir/summary.png"
echo "  gridded rainfall        : $output_dir/event_grid.nc"
echo "  flow and residual fields: $output_dir/flow_residual.nc"
echo "  frame-level metrics     : $output_dir/frame_metrics.csv"
