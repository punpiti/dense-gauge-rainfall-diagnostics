# Pair-max primary and computational controls — v1.1.0

Calculation code for the revised Atmospheric Research manuscript, published
in the same GitHub repository as the historical fixed15 v1.0.0 implementation.
There is no separate code supplement. AI-assisted code development and
checks are disclosed in the manuscript; successful execution is not independent
meteorological validation or author scientific approval.

## What is supplied

- `run_primary.py`: pair-max primary indicators, fixed15/no-advection comparisons,
  and optional complete finite-M no-fill reconstruction at fixed windows.
- `analysis/flow_scaling_check.py`: identical numerical pair scaling, mask,
  Farneback displacement, approximate inverse warp and translation controls.
- `analysis/density_motion_check.py`: source-origin nominal/available support,
  local nearest-neighbor moments, within-frame intensity matching and storm bootstrap.
- `analysis/audit_no_fill_event_linkage.py`: separate station/event relinking control.
- `processing/`: retained reconstruction, flow helpers and storm clustering.
- `demo.py`: unrestricted in-memory Gaussian test, not Bangkok observations.
- Source/output code hashes document packaging adaptations. No restricted
  rainfall, cached fields, station data, third-party papers, internal review
  files or archives are included.

## Run without observations

Use Python 3.13 and the execution packages listed in requirements.txt. OpenCV
runtime used for the paper was 4.12.0. Do not infer bit-identical results across
unverified dependency versions.

```bash
python demo.py
python analysis/flow_scaling_check.py --stage 0 --output synthetic_control
```

The demo checks shared scaling, masking, exact residual subtraction and the
mandatory-area Boolean condition. Synthetic control output is software-test
evidence, not a replacement for restricted rainfall observations.

## Required scientific inputs

Exact Bangkok reproduction requires BMA permission and the retained inputs:

1. Product directory with `all_storm_results.csv` (status, storm_id, cache_path;
   157 rows with status=done) and `grids/STORM_*.zarr/rainfall`. The reader
   accepts the retained Zarr v3 float32, little-endian bytes/zstd, full-spatial
   one-time-step chunks and fails on different formats.
2. For field reconstruction, event caches in parquet with time, stationid,
   rainfall and original rainfall_flag. Relative cache_path resolves against
   `--data-root`. Finite M values include zeros and lead-in slots.
3. `stations.csv`: id, nameShort, lat, lon, using the coordinate version that
   generated the archived grids. Do not substitute later coordinate updates.
4. Bangkok boundary GeoJSON in WGS84, with Polygon/MultiPolygon features.
5. Event-linkage control additionally requires canonical station/year parquet
   observations and retained station events, initial storms and final storm table.

These inputs are not bundled, and authors cannot grant access to BMA rainfall.
The fixed-cache control does not re-run event linkage or alter interval boundaries.

```bash
python run_primary.py --products /path/to/products --output primary_results --workers 8

python run_primary.py --products /path/to/products --data-root /path/to/data \
  --stations /path/to/retained_station_coordinates.csv \
  --boundary /path/to/bangkok_boundary.geojson \
  --no-fill --workers 8 --output primary_nofill_results

python analysis/flow_scaling_check.py --products /path/to/products \
  --stage all --workers 8 --output scaling_results

python analysis/density_motion_check.py --products /path/to/products \
  --stations /path/to/retained_station_coordinates.csv \
  --boundary /path/to/bangkok_boundary.geojson \
  --workers 8 --output density_results

python analysis/audit_no_fill_event_linkage.py \
  --parquet-dataset /path/to/canonical_station_year_dataset \
  --retained-events /path/to/station_events.parquet \
  --retained-storms /path/to/storm_episodes.parquet \
  --final-storms /path/to/final_157_storms.csv \
  --cluster-script processing/cluster_station_events_to_storms.py \
  --out-dir linkage_results --workers 8
```

## Numerical decisions

Pair-max shares one upper bound, max(0.1, max(first), max(second)), across both
uint8 images. Non-finite inputs become zero; the union >=0.1-mm signal mask is
applied. Farneback settings and the source-defined forward-flow subtraction
match the paper. Backward sampling is an approximate inverse map for spatially
varying flow, not a solver for deformation or conservative transport. Original
rainfall amplitudes, not uint8 values, are warped and differenced.

The four indicators use later finite rainy cells >=0.1 mm, residuals >=0.2 mm,
and strong residuals >=0.5 mm. The unchanged rule is area>=500 AND
(peak sum>=1000 OR analysis-window integrated sum>=4000 OR strong frames>=10). Thresholds are not
re-fitted. Baseline reconstruction must match original grids including NaN
support before any no-fill result is interpreted. Supply retained coordinates
directly; this package does not automatically repair a coordinate mismatch.

Every consecutive pair in the complete retained cache is used. The requested
lead-in is 120 minutes: 155 storms retain it in full, STORM_000499 retains
105 minutes, and STORM_000216 none. All caches end at linked storm end.
Analysis-window duration is linked storm duration plus the retained lead-in.
All four indicators use this full
window, not a crop to the linked event. The legacy column name `event_pos_sum`
is retained for compatibility and represents the analysis-window integrated
positive-residual sum E; its meaning and numerical values have not changed.

The CLI can use --limit for a smoke subset; that is not the complete study.
All primary results require the full 157 retained windows. Density comparisons
are recovery associations on reconstructed fields, not thinning or validation
against unobserved rainfall. Bootstrap intervals are exploratory and unadjusted.

## Reproducibility boundary

This release supplies executable primary/control calculations and retained construction
code, not BMA acquisition or automatic recovery of historical metadata. Figure
layout scripts and internal prose/QA logs are intentionally excluded. Saved
primary products, figures and review PDFs remain separate deliverables. No
new observational or physical experiment is introduced by the synthetic demo.
