#!/usr/bin/env bash
set -euo pipefail

example_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_dir="$(cd "$example_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
generated="$example_dir/generated"
demo_output="$example_dir/demo_output"

mkdir -p "$generated" "$demo_output"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$demo_output/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"

echo "[1/7] Generating synthetic station rainfall in the documented input format"
"$python_bin" "$example_dir/make_synthetic_inputs.py"

echo "[2/7] Reconstructing the gridded rainfall field"
"$python_bin" "$package_dir/code/build_event_grid_xarray.py" \
  --cache "$generated/event_window.parquet" \
  --stations "$generated/stations.csv" \
  --boundary "$generated/no_boundary_file.geojson" \
  --grid-km 1.0 --pad-km 2.0 \
  --smooth --sigma-km 2.5 --cutoff-sigma 3.0 \
  --no-anchor-stations --only-influenced --min-rain 0.1 \
  --out "$demo_output/event_grid.nc" --overwrite

echo "[3/7] Computing optical flow, advection-only reference, and residual"
"$python_bin" "$package_dir/code/compute_optical_flow_xarray.py" \
  --in "$demo_output/event_grid.nc" \
  --out "$demo_output/flow_residual.nc" \
  --method farneback --rain-threshold 0.1 --dt-min 15 --overwrite

echo "[4/7] Extracting frame- and event-level residual metrics"
"$python_bin" "$package_dir/code/classify_convective_from_residual.py" \
  --flow "$demo_output/flow_residual.nc" \
  --grid "$demo_output/event_grid.nc" \
  --event-id SYNTHETIC_EVENT \
  --out-summary "$demo_output/intermediate_storm_metrics.csv" \
  --out-frames "$demo_output/frame_metrics.csv"

echo "[5/7] Preparing the final-rule input columns"
"$python_bin" "$example_dir/prepare_final_rule_input.py" \
  --input "$demo_output/intermediate_storm_metrics.csv" \
  --output "$demo_output/storm_metrics_for_final_rule.csv"

echo "[6/7] Applying the manuscript's final operational rule"
"$python_bin" "$package_dir/code/apply_convective_thresholds.py" \
  --input "$demo_output/storm_metrics_for_final_rule.csv" \
  --outdir "$demo_output/final_label" \
  --area-threshold 500 --peak-sum-threshold 1000 \
  --event-sum-threshold 4000 --frames-threshold 10 --mode strict

echo "[7/7] Rendering the visual summary"
"$python_bin" "$example_dir/make_demo_visual.py" \
  --grid "$demo_output/event_grid.nc" \
  --flow "$demo_output/flow_residual.nc" \
  --result "$demo_output/final_label/all_storm_results_labeled.csv" \
  --output "$demo_output/demo_summary.png"

echo
echo "Demo result:"
"$python_bin" -c "import pandas as pd; print(pd.read_csv(r'$demo_output/final_label/all_storm_results_labeled.csv')[['storm_id','class_peak_pos_area_cells','class_peak_pos_sum','class_event_pos_sum','class_n_strong_frames','convective_rule_label']].to_string(index=False))"
echo "Visual summary: $demo_output/demo_summary.png"
