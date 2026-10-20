# Complete-source catalogue and fixed-rule evaluation

This is a separate extension of the original fixed 157-window analysis. It
uses the full cleaned archive, links station events before screening, retains
unique conversion chunk names, and applies the original support screens and
Boolean rule. Inputs remain provider restricted. No restricted observations,
coordinates, extracted caches, or analysis-result rows are distributed here.
The original releases and legacy calculations remain available unchanged.

Use the environment in `../primary/requirements.txt`. Supply the historical
reconstruction coordinate version explicitly, not later updated locations.
`stations.csv` follows `primary/run_primary.py`'s schema. Run from the repository
root, substituting your authorized input paths:

```sh
python full_archive/build_catalogue.py --archive /data/cleaned.csv.bz2 \
  --output /work/full-evaluation --expected-rows 114844032
python full_archive/run_catalogue.py --output /work/full-evaluation \
  --source-parts /work/full-evaluation/restricted/source_parts \
  --stations /data/reconstruction-stations.csv --boundary /data/boundary.geojson \
  --workers 8
python full_archive/threshold_sensitivity.py \
  --input /work/full-evaluation/results.csv --output /work/thresholds
python full_archive/known_deformation.py --output /work/mapping-control
```

The expected row count is optional for other datasets. Station rainfall
occurrence is >0.1 mm, station-event gap <=120 min, and cross-station merge gap
<=30 min. Components are linked before requiring >=5 member events, >=5 member
stations and duration 30–360 min. Caches include 120 min before linked start.
The support screen requires >=20 station-time records with rainfall >=0.1 mm,
>=3 rainy stations, >=4 rainy timestamps and >=2 simultaneous rainy stations.
The default cache completeness assertion expects all 131 station series at
every timestamp, including missing values; `--expected-stations` can configure
other networks. The unlocated station participates in temporal linkage but
not reconstruction. Support qualification is not independent storm validation.

Calculation repeats sigma 1/1.5/2.5/3.5/5 km, four image scalings, no advection,
complete bilinear donor support and no anchoring. The primary rule remains
A>=500 AND (P>=1000 OR E>=4000 OR N>=10). Threshold perturbations use all 81
combinations of 0.8/1/1.2 multipliers without choosing a preferred rule. Full
fixed-window temporal-fill reconstruction is not repeated by this entrypoint;
that original-sample control remains in `primary/`.

Per-window JSON checkpoints support resumption. Use a new output directory
whenever inputs, configuration or code change: checkpoint resumption assumes
the same inputs and settings. The build entrypoint refuses existing chunk
files. Its `--catalogue-only` option reuses a previously completed extraction.
The numerical calculation uses cached station weights and float64 sums in
original station order. This optimization reproduced all 157 archived fields
bit-for-bit, including NaN support. The portable calculation reproduced all
controls on eight newly included windows and all 35 scale rows of the seven
support-qualified pilot cases. All 155 intact historical-window metrics match
the complete-source evaluation. These checks are implementation checks, not
independent meteorological validation.

For the requested original-sample anchoring ablation:

```sh
python primary/run_anchoring_control.py --products /data/original-products \
  --data-root /data --stations /data/reconstruction-stations.csv \
  --boundary /data/boundary.geojson --output /work/anchoring --workers 4
```

This checks every baseline field exactly before removing station-cell
anchoring. The known-deformation experiment uses analytically prescribed affine
motion and exact inverse coordinates; it does not estimate atmospheric motion
or establish the real-data error of the approximate inverse warp.

## Fixed support and full-cohort summaries (v1.4.0)

The bandwidth experiment now separates the original 3-sigma truncation policy
from a fixed 7.5-km radius. Run on the completed year-round evaluation:

```sh
python full_archive/run_fixed_footprint.py --input /work/full-evaluation \
  --output /work/fixed-support --stations /data/reconstruction-stations.csv \
  --boundary /data/boundary.geojson --workers 8
python full_archive/full_context.py --input /work/full-evaluation \
  --output /work/context --stations /data/reconstruction-stations.csv \
  --boundary /data/boundary.geojson \
  --historical-window-table /data/original-pair-max-windows.csv --workers 4
python full_archive/cohort_grouping.py --input /work/full-evaluation/results.csv \
  --output /work/grouping
python full_archive/summarize_catalogue.py --input /work/full-evaluation \
  --output /work/full-evaluation/summary.json
python full_archive/plot_controls.py \
  --rebuild-summary /work/full-evaluation/summary.json \
  --fixed-summary /work/fixed-support/summary.csv --output /work/control-plot
```

The fixed-support calculation requires the `results.csv`, `scale_results.csv`
and restricted caches from `run_catalogue.py`. It enforces identical cell-time
finite support across all five bandwidths and reproduces the 2.5-km primary
metrics. The new cutoff argument defaults to the previous 3-sigma policy, so
the primary calculations and rule are unchanged. Use a fresh checkpoint folder
for a different configuration. Fixed finite support does not equalize effective
gauge weights, observational information, or uncertainty.

`full_context.py` reports class statistics, hourly counts/percentages and knees
on the entire completed cohort. Intensity is the maximum gauge accumulation
inside linked start/end, excluding the lead-in; residual indicators retain the
original lead-inclusive windows. The historical table requires `storm_start`
and `convective_rule_label` (`convective` / `non_convective`), and is used only
for descriptive monthly-composition standardization. The decomposition is
arithmetic, not a causal or physical-classification analysis.

In the manuscript evaluation, fixed 7.5-km high-residual counts were
634/706/901/1027/1061, compared with 73/332/901/1397/1788 for 3-sigma truncation.
Portable source reproduces 80 control rows in eight windows and all 3,304
context rows, class statistics, distributions and monthly standardization.
An independent original frame-loop implementation matched 75 sampled fields;
all 157 original primary fields still reproduce bit-for-bit. Known translations,
local density and temporal-fill reconstruction retain the historical benchmark;
they have not been extended to every rebuilt window. No independent real-motion,
physical storm-type or radar validation is added by these checks.

The independent original frame-loop check is also available:

```sh
python full_archive/verify_fixed_footprint.py --input /work/full-evaluation \
  --output /work/independent-check --stations /data/reconstruction-stations.csv \
  --boundary /data/boundary.geojson
```

This checks 75 selected fields across five windows and five bandwidths; it does
not label that sample check as a complete field-by-field external validation.
