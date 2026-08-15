#!/usr/bin/env python3
"""Render a compact visual summary of the end-to-end synthetic demo."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    grid = xr.open_dataset(args.grid)
    flow = xr.open_dataset(args.flow)
    result = pd.read_csv(args.result).iloc[0]

    observed = grid["rainfall"].values[-1]
    advected = flow["R_adv"].values[-1]
    residual = flow["residual"].values[-1]

    rain_vmax = float(np.nanmax([np.nanmax(observed), np.nanmax(advected)]))
    residual_limit = float(np.nanmax(np.abs(residual)))

    fig = plt.figure(figsize=(14, 8.4), facecolor="white", constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.08, 0.82])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]

    rain_cmap = "Blues"
    signed_cmap = "PuOr_r"

    im0 = axes[0].imshow(observed, origin="lower", cmap=rain_cmap, vmin=0, vmax=rain_vmax)
    axes[0].set_title("Observed synthetic rainfall, final frame")
    im1 = axes[1].imshow(advected, origin="lower", cmap=rain_cmap, vmin=0, vmax=rain_vmax)
    axes[1].set_title("Advection-only reference, final frame")
    im2 = axes[2].imshow(
        residual,
        origin="lower",
        cmap=signed_cmap,
        vmin=-residual_limit,
        vmax=residual_limit,
    )
    axes[2].set_title("Residual: observed minus advected")

    for axis in axes:
        axis.set_xlabel("Grid x index")
        axis.set_ylabel("Grid y index")

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="mm per 15 min")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="mm per 15 min")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="mm per 15 min")

    score_ax = fig.add_subplot(gs[1, :])
    metric_names = ["Peak area", "Peak sum", "Event sum", "Strong frames"]
    metric_values = np.array(
        [
            result["class_peak_pos_area_cells"],
            result["class_peak_pos_sum"],
            result["class_event_pos_sum"],
            result["class_n_strong_frames"],
        ],
        dtype=float,
    )
    thresholds = np.array([500.0, 1000.0, 4000.0, 10.0])
    ratios = metric_values / thresholds
    y = np.arange(len(metric_names))
    colors = ["#D98C10" if value >= 1.0 else "#59788E" for value in ratios]
    score_ax.barh(y, ratios, color=colors, edgecolor="#263238", linewidth=0.8)
    score_ax.axvline(1.0, color="#263238", linestyle="--", linewidth=1.2, label="Threshold")
    score_ax.set_yticks(y, metric_names)
    score_ax.invert_yaxis()
    score_ax.set_xlabel("Metric / operational threshold")
    score_ax.set_title("Final operational rule")
    score_ax.grid(axis="x", color="#D9DEE2", linewidth=0.7)
    score_ax.set_axisbelow(True)

    exact_labels = [
        f"{metric_values[0]:.0f} cells / 500",
        f"{metric_values[1]:.1f} / 1000",
        f"{metric_values[2]:.1f} / 4000",
        f"{metric_values[3]:.0f} frames / 10",
    ]
    for index, (ratio, label) in enumerate(zip(ratios, exact_labels)):
        score_ax.text(ratio + 0.04, index, label, va="center", fontsize=9, color="#263238")

    score_ax.set_xlim(0, max(4.7, float(np.max(ratios)) + 1.0))
    final_label = str(result["convective_rule_label"]).upper().replace("_", "-")
    score_ax.text(
        0.985,
        0.96,
        f"Final label: {final_label}",
        transform=score_ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="right",
        va="top",
        color="#8A5700" if final_label == "CONVECTIVE" else "#36566D",
    )
    score_ax.text(
        0.985,
        0.84,
        "Synthetic demonstration only — not BMA observations or a performance estimate.",
        transform=score_ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        color="#4F5B62",
    )

    fig.suptitle(
        "End-to-end demonstration of the released paper-processing code",
        fontsize=16,
        fontweight="bold",
        y=1.03,
    )
    fig.text(
        0.5,
        0.985,
        "15 synthetic 15-min frames from 36 stations; shared rainfall scale and signed residual scale",
        ha="center",
        fontsize=10,
        color="#4F5B62",
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Demo visual written to {output}")


if __name__ == "__main__":
    main()
