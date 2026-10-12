#!/usr/bin/env python3
"""All-storm no-advection ablation for the retained Bangkok storm sample.

The analysis reads the archived Zarr v3 arrays directly because opening many
small Zarr chunks through xarray on the mounted source drive is unnecessarily
slow.  The decoder is deliberately narrow: it accepts the float32, time-major,
Zstandard-compressed arrays used by the retained products and fails on any
unexpected metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numcodecs import Zstd


PRODUCTS = Path("inputs/products")
OUTPUT = Path(__file__).resolve().parent / "no_advection_ablation"
MANUSCRIPT_FIGURE = (
    Path(__file__).resolve().parents[2]
    / "manuscript"
    / "figures"
    / "fig_no_advection_ablation.png"
)

METRICS = {
    "peak_pos_area_cells": "Peak positive-residual area (cells)",
    "peak_pos_sum": "Peak positive-residual sum",
    "event_pos_sum": "Analysis-window integrated\npositive residual",
    "n_strong_frames": "Strong-residual frames",
}


def read_zarr_v3_array(array_dir: Path) -> np.ndarray:
    """Read a retained Zarr v3 array with one full spatial field per chunk."""
    metadata = json.loads((array_dir / "zarr.json").read_text())
    shape = tuple(metadata["shape"])
    chunk_shape = tuple(metadata["chunk_grid"]["configuration"]["chunk_shape"])
    codecs = metadata["codecs"]
    if metadata["data_type"] != "float32":
        raise ValueError(f"Unexpected dtype in {array_dir}: {metadata['data_type']}")
    if chunk_shape != (1, shape[1], shape[2]):
        raise ValueError(f"Unexpected chunks in {array_dir}: {chunk_shape}")
    if [c["name"] for c in codecs] != ["bytes", "zstd"]:
        raise ValueError(f"Unexpected codecs in {array_dir}: {codecs}")
    if codecs[0]["configuration"].get("endian") != "little":
        raise ValueError(f"Unexpected endian setting in {array_dir}")

    result = np.full(shape, np.nan, dtype="<f4")
    decoder = Zstd()
    frame_size = int(np.prod(chunk_shape))
    for k in range(shape[0]):
        chunk_path = array_dir / "c" / str(k) / "0" / "0"
        if not chunk_path.exists():
            continue
        raw = decoder.decode(chunk_path.read_bytes())
        frame = np.frombuffer(raw, dtype="<f4", count=frame_size)
        result[k] = frame.reshape(chunk_shape)[0]
    return result


def summarize(residual: np.ndarray, observed: np.ndarray) -> dict[str, float | int]:
    """Apply the retained frame masks and storm-level summaries unchanged."""
    if residual.shape != observed.shape:
        raise ValueError(f"Shape mismatch: residual={residual.shape}, observed={observed.shape}")
    pos_areas: list[int] = []
    pos_sums: list[float] = []
    strong_frames: list[bool] = []
    for res, robs in zip(residual, observed):
        rainy = np.isfinite(robs) & (robs >= 0.1)
        positive = rainy & np.isfinite(res) & (res >= 0.2)
        strong = rainy & np.isfinite(res) & (res >= 0.5)
        pos_areas.append(int(np.sum(positive)))
        pos_sums.append(float(np.nansum(np.where(positive, res, 0.0))))
        strong_sum = float(np.nansum(np.where(strong, res, 0.0)))
        strong_frames.append(bool(np.sum(rainy) >= 3 and np.sum(strong) >= 3 and strong_sum > 0.0))
    return {
        "peak_pos_area_cells": max(pos_areas, default=0),
        "peak_pos_sum": max(pos_sums, default=0.0),
        "event_pos_sum": float(np.sum(pos_sums)),
        "n_strong_frames": int(np.sum(strong_frames)),
    }


def operational_label(values: dict[str, float | int]) -> str:
    qualifies = values["peak_pos_area_cells"] >= 500 and (
        values["peak_pos_sum"] >= 1000
        or values["event_pos_sum"] >= 4000
        or values["n_strong_frames"] >= 10
    )
    return "high_residual" if qualifies else "other"


def analyze() -> pd.DataFrame:
    source = pd.read_csv(PRODUCTS / "all_storm_results.csv")
    source = source.loc[source["status"].eq("done")].copy()
    if len(source) != 157:
        raise ValueError(f"Expected 157 completed storms, found {len(source)}")

    rows: list[dict[str, object]] = []
    for row in source.itertuples(index=False):
        storm_id = row.storm_id
        rainfall = read_zarr_v3_array(PRODUCTS / "grids" / f"{storm_id}.zarr" / "rainfall")
        archived_residual = read_zarr_v3_array(
            PRODUCTS / "flows" / f"{storm_id}.zarr" / "residual"
        )
        observed = rainfall[1:]
        rainfall_for_difference = np.where(np.isfinite(rainfall), rainfall, 0.0).astype(np.float32)
        no_advection_residual = rainfall_for_difference[1:] - rainfall_for_difference[:-1]

        optical = summarize(archived_residual, observed)
        no_advection = summarize(no_advection_residual, observed)
        result: dict[str, object] = {
            "storm_id": storm_id,
            "storm_start": row.storm_start,
            "optical_label": operational_label(optical),
            "no_advection_label": operational_label(no_advection),
        }
        for metric in METRICS:
            result[f"optical_{metric}"] = optical[metric]
            result[f"no_advection_{metric}"] = no_advection[metric]
            result[f"difference_{metric}"] = no_advection[metric] - optical[metric]
            result[f"source_{metric}"] = getattr(row, f"class_{metric}")
        rows.append(result)
    return pd.DataFrame(rows)


def validate_reconstruction(results: pd.DataFrame) -> pd.DataFrame:
    checks = []
    for metric in METRICS:
        delta = results[f"optical_{metric}"] - results[f"source_{metric}"]
        checks.append(
            {
                "metric": metric,
                "maximum_absolute_difference": float(np.max(np.abs(delta))),
                "all_close": bool(np.allclose(delta, 0.0, rtol=1e-6, atol=1e-5)),
            }
        )
    validation = pd.DataFrame(checks)
    if not validation["all_close"].all():
        raise RuntimeError(f"Archived metric reconstruction failed:\n{validation}")
    return validation


def build_summary(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    for metric, label in METRICS.items():
        optical = results[f"optical_{metric}"].to_numpy(dtype=float)
        baseline = results[f"no_advection_{metric}"].to_numpy(dtype=float)
        metric_rows.append(
            {
                "metric": metric,
                "display_name": label,
                "optical_median": float(np.median(optical)),
                "no_advection_median": float(np.median(baseline)),
                "median_paired_difference_noadv_minus_optical": float(np.median(baseline - optical)),
                "median_relative_difference_percent": float(
                    np.median(np.divide(baseline - optical, optical, out=np.full_like(optical, np.nan), where=optical != 0))
                    * 100
                ),
                "spearman_correlation": float(pd.Series(optical).corr(pd.Series(baseline), method="spearman")),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    contingency = pd.crosstab(
        results["optical_label"], results["no_advection_label"], margins=True
    ).rename_axis(index="optical_flow", columns="no_advection")
    return metrics, contingency


def make_figure(results: pd.DataFrame, output_path: Path) -> None:
    colors = {"other": "#1f77b4", "high_residual": "#d95f02"}
    labels = {"other": "Other class", "high_residual": "High-residual class"}
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 9.0), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, METRICS.items()):
        x = results[f"optical_{metric}"].to_numpy(dtype=float)
        y = results[f"no_advection_{metric}"].to_numpy(dtype=float)
        for cls in ["other", "high_residual"]:
            mask = results["optical_label"].eq(cls).to_numpy()
            ax.scatter(x[mask], y[mask], s=24, alpha=0.75, color=colors[cls], label=labels[cls])
        high = max(float(np.max(x)), float(np.max(y)))
        ax.plot([0, high], [0, high], color="#555555", linewidth=1.2, linestyle="--")
        ax.set_xlabel("Optical-flow residual")
        ax.set_ylabel("No-advection difference")
        ax.set_title(title)
        ax.grid(alpha=0.18)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside upper center", ncol=2, frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = analyze()
    validation = validate_reconstruction(results)
    metrics, contingency = build_summary(results)
    results.to_csv(OUTPUT / "storm_level_results.csv", index=False)
    validation.to_csv(OUTPUT / "archived_metric_validation.csv", index=False)
    metrics.to_csv(OUTPUT / "metric_summary.csv", index=False)
    contingency.to_csv(OUTPUT / "label_contingency.csv")
    make_figure(results, OUTPUT / "paired_metric_comparison.png")
    make_figure(results, MANUSCRIPT_FIGURE)

    changed = int((results["optical_label"] != results["no_advection_label"]).sum())
    print(validation.to_string(index=False))
    print("\nMetric summary")
    print(metrics.to_string(index=False))
    print("\nLabel contingency")
    print(contingency.to_string())
    print(f"\nChanged labels: {changed} of {len(results)}")


if __name__ == "__main__":
    main()
