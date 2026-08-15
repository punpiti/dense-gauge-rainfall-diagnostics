# Checked demo result

The one-command demonstration was executed in the environment recorded in
`../ENVIRONMENT_CHECKED.txt`. The main machine-readable output is
`demo_output/final_label/all_storm_results_labeled.csv`; it produced:

| Field | Checked value |
|---|---:|
| `class_peak_pos_area_cells` | 931 |
| `class_peak_pos_sum` | 1312.666016 |
| `class_event_pos_sum` | 14512.84552 |
| `class_n_strong_frames` | 14 |
| `convective_rule_label` | `convective` |

The exact floating-point sums may vary slightly with numerical-library or
OpenCV versions, but the supplied synthetic event is designed to pass the
documented operational rule. These are demonstration values, not results from
BMA observations and not evidence about classification performance.

The human-readable visual output is `demo_output/summary.png`. Intermediate
NetCDF and CSV files are retained under `demo_output/` so each processing stage
can be inspected.
