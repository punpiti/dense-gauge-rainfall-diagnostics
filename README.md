# Paper processing code

[![Demo guide](https://img.shields.io/badge/demo-GitHub%20Pages-006eb7)](https://punpiti.github.io/egusphere-2026-1314-processing-code/)
[![Release](https://img.shields.io/badge/release-v1.0.0-19734a)](https://github.com/punpiti/egusphere-2026-1314-processing-code/releases/tag/v1.0.0)
[![License: MIT](https://img.shields.io/badge/license-MIT-df930b.svg)](LICENSE)

For a one-page visual guide to the demonstration and its outputs, open
[`index.html`](index.html) in a browser. The same file can serve as the project
landing page when this package is published from a GitHub repository.

After publication, the complete command-line quick start is:

```bash
git clone https://github.com/punpiti/egusphere-2026-1314-processing-code.git
cd egusphere-2026-1314-processing-code
python -m pip install -r requirements.txt
bash example/run_demo.sh
```

This supplement contains the Python processing scripts used for the
paper-specific rainfall-field, optical-flow, residual-metric, and final
threshold-labeling stages of egusphere-2026-1314.

It is a focused source release, not a copy of the authors' internal workspace.
It intentionally excludes raw-data acquisition, event-cache contents, batch
orchestration, performance/parallelization utilities, machine-specific paths,
logs, intermediate arrays, and restricted Bangkok Metropolitan Administration
(BMA) observations.

## Processing stages

1. `build_event_grid_xarray.py`
   converts an event-window station table to the 1 km gridded rainfall field
   and implements the normalized Gaussian reconstruction used in the paper.
2. `compute_optical_flow_xarray.py`
   estimates Farneback optical flow, constructs the advected reference field,
   and calculates `residual = R_next - R_adv`.
3. `classify_convective_from_residual.py`
   calculates frame- and storm-level residual metrics. Its low-threshold
   `convective_label` is an intermediate processing/QC label; it is not the
   final 36/121 manuscript classification.
4. `apply_convective_thresholds.py`
   applies the final paper rule. In `strict` mode this is
   `area >= 500 AND (peak_sum >= 1000 OR event_sum >= 4000 OR frames >= 10)`.
5. `analyze_convective_thresholds.py`
   produces the empirical threshold summaries and exploratory plots.
6. `plot_convective_feature_space.py`
   produces the residual-feature-space plots.

The final 157-storm labeling was run in `strict` mode and produced 36
convective and 121 non-convective operational labels.

## Inputs

The source release does not contain BMA observations. Required schemas are
documented in `INPUT_SCHEMA.md`. Human-readable templates are supplied as
`example/measured_rainfall_format.csv` and
`example/station_metadata_format.csv`. The synthetic generator writes the same
rainfall table to Parquet, which is the format consumed by the gridding script.
This is the event-window processing-input format after the study's documented
QA/QC and event selection; it is not the provider's original raw-export format.

## Environment

Install the packages listed in `requirements.txt`. The authors' checked
environment used Python 3.13 and the package versions recorded in
`ENVIRONMENT_CHECKED.txt`; compatible newer or older releases may also work.

## End-to-end synthetic demonstration

From this directory, run:

```bash
bash example/run_demo.sh
```

To select a particular Python executable:

```bash
PYTHON_BIN=/path/to/python bash example/run_demo.sh
```

The demonstration creates a 15-frame, 36-station synthetic rainfall event in
the documented measured-rainfall format and then runs the released gridding,
optical-flow, residual, metric-extraction, and final-label stages. The final
result is printed on screen and saved under
`example/demo_output/final_label/`. The expected operational label is
`convective`; the checked metrics are recorded in
`example/DEMO_EXPECTED_RESULT.md`. The command also creates
`example/demo_output/demo_summary.png`, showing the final observed rainfall,
advection-only reference, signed residual, threshold ratios, and final label.
A checked preview is included as `example/DEMO_EXPECTED_VISUAL.png`.

`example/prepare_final_rule_input.py` is a transparent column-name adapter
between the metric-extraction output and the final threshold script; it does
not calculate or modify the metrics. The scientific processing is performed by
the released paper source files, not by a simplified reimplementation.

## Reproducibility boundary

This release permits inspection and execution of the method-specific
processing code. Exact regeneration of the Bangkok results additionally
requires the BMA station observations and the authors' event-window definitions,
which cannot be redistributed here. Consequently, the package supports code
inspection and method execution but does not claim data-complete replication
of the Bangkok analysis.
