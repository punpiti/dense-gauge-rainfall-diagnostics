# Dense-gauge rainfall diagnostics

[![Demo guide](https://img.shields.io/badge/demo-GitHub%20Pages-006eb7)](https://punpiti.github.io/dense-gauge-rainfall-diagnostics/)
[![Release](https://img.shields.io/badge/release-v1.2.0-19734a)](https://github.com/punpiti/dense-gauge-rainfall-diagnostics/releases/tag/v1.2.0)
[![License: MIT](https://img.shields.io/badge/license-MIT-df930b.svg)](LICENSE)

Transport-relative analysis of rainfall development from dense rain-gauge
networks, with reproducible processing scripts and a synthetic demonstration.

## Current manuscript: pair-max, v1.2.0

Use [`primary/run_primary.py`](primary/run_primary.py) for the revised
Atmospheric Research manuscript. It calculates pair-max indicators and labels,
fixed15/no-advection comparisons, and the fixed-window finite-fill control
(including filled zeros). The unchanged Boolean rule gives 36 high-residual
and 121 other windows in the retained 157-storm analysis. These are operational
diagnostic groups, not independently validated convective labels.

```bash
git clone --branch v1.2.0 https://github.com/punpiti/dense-gauge-rainfall-diagnostics.git
cd dense-gauge-rainfall-diagnostics
python -m pip install -r primary/requirements.txt
python primary/demo.py
python primary/run_primary.py --products /path/to/products --workers 8 --output /path/to/results
```

The demo uses unrestricted synthetic inputs only. Bangkok execution requires
permission to use BMA observations and retained products/metadata, which are
not distributed. See [`primary/README.md`](primary/README.md) for input schemas,
no-fill, scaling-recovery, local-support and event-linkage control commands.
All indicators use complete retained analysis windows, including available
pre-storm lead-ins; linked storm duration is separate.

This repository is the sole public code location; there is no separate code
supplement. The primary directory contains calculation code and essential usage
documentation only: no manuscript, restricted inputs, figure/layout scripts,
internal review records, or experimental prototypes. Releases v1.0.0 and v1.1.0 are retained unchanged as historical provenance.

## Reconstruction and archive-window controls

Release v1.2.0 adds `primary/run_reconstruction_controls.py` for complete
bilinear donor support and five reconstruction scales at fixed windows. The
primary 36/121 grouping and rule remain unchanged. Supported donors give
31 high-residual windows; scales of 1/1.5/2.5/3.5/5 km give 2/11/36/63/92.

`primary/audit_archive_windows.py` compares local contexts with the complete
cleaned archive, reconstructing station events and storm components with the
original gap rules. One retained storm start is cut and another window has
an incomplete lead-in. Contextual summaries exclude both (155 windows,
36/119 groups); the high-residual scale/donor counts remain unchanged.
No annual catalogue is rebuilt. Restricted inputs and extracted rows are not
included. See the primary README for schemas, commands and synthetic tests.

## Related research and provenance

The processing code was initially released with the following preprint. Its
persistent identifier records the earlier work; the software uses a journal-neutral name.

Piamsa-nga, P., Phisanbut, N., and Piamsa-nga, N. (2026), “A dense
rain-gauge observation framework for advection-consistency diagnosis of urban
convective rainfall,” *EGUsphere* [preprint].

- [Official article and public-discussion page](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-1314/)
- [https://doi.org/10.5194/egusphere-2026-1314](https://doi.org/10.5194/egusphere-2026-1314)

## Historical fixed15 interface

The following commands and visual guide use the original fixed15 workflow,
not the current pair-max primary analysis.

For a one-page visual guide to the historical demonstration and its outputs, open
[`index.html`](index.html) in a browser. The same file can serve as the project
landing page when this package is published from a GitHub repository.

After publication, the complete command-line quick start is:

```bash
git clone https://github.com/punpiti/dense-gauge-rainfall-diagnostics.git
cd dense-gauge-rainfall-diagnostics
python -m pip install -r requirements.txt
bash example/run_demo.sh
```

The historical interface contains the Python processing scripts used for the
rainfall-field, optical-flow, residual-metric, and operational threshold-labeling
stages. Release v1.0.0 preserves the original numerical implementation.

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

The runnable event pipeline requires exactly two scientific input files:

1. a post-QA/QC event-window rainfall table in Parquet; and
2. a station-metadata table in CSV.

The source release does not contain BMA observations. Required columns are
documented in `INPUT_SCHEMA.md`; human-readable templates are supplied as
`example/measured_rainfall_format.csv` and
`example/station_metadata_format.csv`. This is the event-window
processing-input format after the study's documented QA/QC and event selection;
it is not the provider's original raw-export format.

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
the two documented input files and then calls `example/run_pipeline.sh`, which
runs the released gridding, optical-flow, residual, metric-extraction, and
final-label stages. The main final result is printed on screen and saved as
`example/demo_output/final_label/all_storm_results_labeled.csv`. The expected operational label is
`convective`; the checked metrics are recorded in
`example/DEMO_EXPECTED_RESULT.md`. The command also creates
`example/demo_output/summary.png`, showing the final observed rainfall,
advection-only reference, signed residual, threshold ratios, and final label.
A checked preview is included as `example/DEMO_EXPECTED_VISUAL.png`.

## Hook in one measured event

Use the same pipeline wrapper with your own two input files:

```bash
bash example/run_pipeline.sh \
  --rainfall /path/to/rainfall.parquet \
  --stations /path/to/stations.csv \
  --output-dir results/my_event \
  --event-id MY_EVENT
```

An optional `--boundary boundary.geojson` may be supplied; otherwise the domain
is derived from station extent. `run_pipeline.sh` only passes paths and the
manuscript parameters to the released paper scripts. It does not reimplement
the gridding, optical flow, residual metrics, or final rule.

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
