#!/usr/bin/env python3
"""Adapt released metric-extraction output to the final-rule column names."""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = pd.read_csv(args.input)
    rename = {
        "peak_pos_area_cells": "class_peak_pos_area_cells",
        "peak_pos_sum": "class_peak_pos_sum",
        "event_pos_sum": "class_event_pos_sum",
        "n_strong_frames": "class_n_strong_frames",
        "convective_label": "intermediate_qc_label",
    }
    missing = [column for column in rename if column not in source.columns]
    if missing:
        raise KeyError(f"Missing metric-extraction columns: {missing}")

    output = source.rename(columns=rename)
    output.insert(0, "status", "done")
    output["storm_id"] = output["event_id"].fillna("SYNTHETIC_EVENT")

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(target, index=False)
    print(f"Final-rule input written to {target}")


if __name__ == "__main__":
    main()

