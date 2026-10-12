# Input schemas

For the current pair-max primary and controls, use
[`primary/README.md`](primary/README.md#required-scientific-inputs).
The schemas below describe the historical fixed15 single-event interface.

## Event-window station table

`build_event_grid_xarray.py --cache` reads a Parquet table with at least:

This table represents measured station rainfall after the documented QA/QC and
selection of an event window. It does not describe the BMA provider's original
raw database/export schema.

| Column | Meaning |
|---|---|
| `time` | observation timestamp |
| `stationid` | station identifier; numeric or an `S`-prefixed integer |
| `rainfall` | rainfall accumulation for the 15 min interval |
| `rainfall_flag` | optional quality flag; used only with `--only-flag-N` |

The file is called a cache in the historical command-line interface, but it is
simply the event-window station table consumed by the public processing stage.
No cache files from the Bangkok analysis are distributed.
`example/measured_rainfall_format.csv` shows the same columns in a
human-readable template, while `example/make_synthetic_inputs.py` writes the
Parquet form used by the demonstration.

## Station metadata

`--stations` reads CSV columns `id`, `nameShort`, `lat`, and `lon`. Coordinates
are WGS84 decimal degrees. The processing script projects these coordinates to
UTM zone 47N (EPSG:32647).
See `example/station_metadata_format.csv` for a minimal template.

## Optional boundary

`--boundary` accepts a WGS84 GeoJSON Polygon or MultiPolygon. If the file is
absent, the script derives a rectangular domain from station extent and adds
the requested padding. It is therefore not a required scientific input to
`example/run_pipeline.sh`.

## Storm-metric table

`apply_convective_thresholds.py` and the plotting/threshold-analysis scripts
read CSV or Parquet with these paper variables:

- `class_peak_pos_area_cells`
- `class_peak_pos_sum`
- `class_event_pos_sum`
- `class_n_strong_frames`

The optional `status` column is filtered to `done` by default. In the demo,
`example/prepare_final_rule_input.py` maps the metrics calculated from the
synthetic rainfall event to these final-rule column names without changing
their values.
