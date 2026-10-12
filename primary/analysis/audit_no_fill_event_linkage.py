#!/usr/bin/env python3
"""Reconstruct station events and storm linkage with filled M values removed.

The script runs the published 0.1 mm and 120 min station-event definition twice
from the retained 15 min archive:

1. baseline: all finite rainfall values, reproducing the executed workflow;
2. no-fill: finite values whose retained pre-fill flag is not ``M``.

It then applies the authoritative temporal storm-clustering implementation and
compares the no-fill clusters with the retained storm table and final 157-storm
analysis sample. Outputs are deliberately small and portable; the restricted
rainfall observations are not copied into the revision workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


GAP = pd.Timedelta(minutes=120)
THRESHOLD = 0.1
STORM_MERGE_GAP_MINUTES = 30


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rain", help="Retained cleaned CSV (historical fallback)")
    p.add_argument("--parquet-dataset", help="Canonical station/year parquet dataset")
    p.add_argument("--retained-events", required=True)
    p.add_argument("--retained-storms", required=True)
    p.add_argument("--final-storms", required=True)
    p.add_argument("--cluster-script", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--chunksize", type=int, default=2_000_000)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def station_name(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype("Int64")
    return numeric.map(lambda x: f"S{int(x):03d}" if pd.notna(x) else pd.NA)


def _events_from_station_frame(df: pd.DataFrame, station: str) -> tuple[list[dict], list[dict], dict]:
    """Reproduce the retained event extractor for one station."""
    df = df.sort_values("time").reset_index(drop=True)
    df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")
    flags = df["rainfall_flag"].fillna("<NA>").astype(str)
    totals = {
        "rows": int(len(df)),
        "baseline_rainy_rows": int(df["rainfall"].gt(THRESHOLD).sum()),
        "filled_rainy_rows": int((df["rainfall"].gt(THRESHOLD) & flags.eq("M")).sum()),
    }

    def build(remove_filled: bool) -> list[dict]:
        mask = df["rainfall"].gt(THRESHOLD)
        if remove_filled:
            mask &= flags.ne("M")
        rainy = df.loc[mask, ["time", "rainfall"]]
        if rainy.empty:
            return []
        groups = rainy["time"].diff().gt(GAP).cumsum()
        output = []
        for _, g in rainy.groupby(groups, sort=False):
            start = g["time"].iloc[0]
            end = g["time"].iloc[-1]
            values = g["rainfall"]
            output.append({
                "event_id": f"EV_S{station}_{start:%Y%m%d%H%M}",
                "stationid": station,
                "event_start": start,
                "event_end": end,
                "event_length_minutes": int((end - start).total_seconds() // 60),
                "n_rain_slots": int(len(g)),
                "total_rainfall": float(values.sum()),
                "average_rainfall": float(values.mean()),
                "max_rainfall": float(values.max()),
                "min_rainfall": float(values.min()),
            })
        return output

    return build(False), build(True), totals


def _process_station_dir(path_str: str) -> tuple[list[dict], list[dict], dict]:
    path = Path(path_str)
    station = path.name.split("=", 1)[1]
    files = sorted(path.rglob("*.parquet"))
    parts = [pd.read_parquet(f, columns=["time", "rainfall", "rainfall_flag"]) for f in files]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return _events_from_station_frame(df, station)


def reconstruct_events_parquet(dataset: str, workers: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    station_dirs = sorted(Path(dataset).glob("stationid=*") )
    if not station_dirs:
        raise FileNotFoundError(f"No station partitions under {dataset}")
    baseline_rows: list[dict] = []
    nofill_rows: list[dict] = []
    totals = Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (base, nofill, counts) in enumerate(
            pool.map(_process_station_dir, map(str, station_dirs)), 1
        ):
            baseline_rows.extend(base)
            nofill_rows.extend(nofill)
            totals.update(counts)
            print(f"processed station {i}/{len(station_dirs)}", flush=True)
    frames = []
    for rows in (baseline_rows, nofill_rows):
        frames.append(pd.DataFrame(rows).sort_values(
            ["stationid", "event_start", "event_end"]
        ).reset_index(drop=True))
    return frames[0], frames[1], dict(totals)


def reconstruct_events(rain_path: str, chunksize: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    states: dict[str, dict[str, dict]] = {"baseline": {}, "no_fill": {}}
    rows: dict[str, list[dict]] = {"baseline": [], "no_fill": []}
    totals = Counter()
    last_time: pd.Timestamp | None = None

    def finish(mode: str, station: str) -> None:
        state = states[mode].pop(station, None)
        if state is None:
            return
        rows[mode].append(
            {
                "event_id": f"EV_S{station}_{state['start']:%Y%m%d%H%M}",
                "stationid": station,
                "event_start": state["start"],
                "event_end": state["end"],
                "event_length_minutes": int((state["end"] - state["start"]).total_seconds() // 60),
                "n_rain_slots": state["n"],
                "total_rainfall": state["sum"],
                "average_rainfall": state["sum"] / state["n"],
                "max_rainfall": state["max"],
                "min_rainfall": state["min"],
            }
        )

    def ingest(mode: str, station: str, time: pd.Timestamp, rain: float) -> None:
        state = states[mode].get(station)
        if state is None or time - state["end"] > GAP:
            finish(mode, station)
            states[mode][station] = {
                "start": time, "end": time, "n": 1,
                "sum": float(rain), "max": float(rain), "min": float(rain),
            }
            return
        state["end"] = time
        state["n"] += 1
        state["sum"] += float(rain)
        state["max"] = max(state["max"], float(rain))
        state["min"] = min(state["min"], float(rain))

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            rain_path,
            usecols=["time", "stationid", "rainfall", "rainfall_flag"],
            parse_dates=["time"],
            chunksize=chunksize,
        ),
        1,
    ):
        chunk = chunk.dropna(subset=["time", "stationid"])
        if not chunk.empty:
            chunk_min = chunk["time"].min()
            chunk_max = chunk["time"].max()
            if last_time is not None and chunk_min < last_time:
                raise RuntimeError("Input is not globally chronological")
            last_time = chunk_max
        chunk["stationid"] = station_name(chunk["stationid"])
        chunk["rainfall"] = pd.to_numeric(chunk["rainfall"], errors="coerce")
        chunk["rainfall_flag"] = chunk["rainfall_flag"].fillna("<NA>").astype(str)
        rainy = chunk[chunk["rainfall"].gt(THRESHOLD)].copy()
        totals["rows"] += len(chunk)
        totals["baseline_rainy_rows"] += len(rainy)
        totals["filled_rainy_rows"] += int(rainy["rainfall_flag"].eq("M").sum())
        for row in rainy.itertuples(index=False):
            ingest("baseline", row.stationid, row.time, row.rainfall)
            if row.rainfall_flag != "M":
                ingest("no_fill", row.stationid, row.time, row.rainfall)
        print(f"processed chunk {chunk_no}: {totals['rows']:,} rows", flush=True)

    for mode in states:
        for station in list(states[mode]):
            finish(mode, station)

    frames = []
    for mode in ["baseline", "no_fill"]:
        d = pd.DataFrame(rows[mode]).sort_values(
            ["stationid", "event_start", "event_end"]
        ).reset_index(drop=True)
        frames.append(d)
    return frames[0], frames[1], dict(totals)


def load_cluster_module(path: str):
    spec = importlib.util.spec_from_file_location("authoritative_cluster", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load clustering code: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cluster_ready(events: pd.DataFrame) -> pd.DataFrame:
    """Match the authoritative script's read_events chronological ordering."""
    return events.sort_values(
        ["event_start", "event_end", "stationid", "event_id"]
    ).reset_index(drop=True)


def compare_events(actual: pd.DataFrame, retained: pd.DataFrame) -> dict:
    retained = retained.copy()
    retained["stationid"] = retained["stationid"].astype(str)
    for c in ["event_start", "event_end"]:
        retained[c] = pd.to_datetime(retained[c])
    keys = ["stationid", "event_start", "event_end"]
    a = set(map(tuple, actual[keys].itertuples(index=False, name=None)))
    b = set(map(tuple, retained[keys].itertuples(index=False, name=None)))
    return {
        "reconstructed_rows": len(actual),
        "retained_rows": len(retained),
        "exact_key_matches": len(a & b),
        "reconstructed_only": len(a - b),
        "retained_only": len(b - a),
    }


def compare_storms(actual: pd.DataFrame, retained: pd.DataFrame) -> dict:
    retained = retained.copy()
    for c in ["storm_start", "storm_end"]:
        retained[c] = pd.to_datetime(retained[c])
    cols = ["storm_id", "storm_start", "storm_end", "n_member_events", "n_member_stations"]
    merged = actual[cols].merge(retained[cols], on="storm_id", suffixes=("_new", "_retained"), how="outer", indicator=True)
    exact = merged["_merge"].eq("both")
    for c in ["storm_start", "storm_end", "n_member_events", "n_member_stations"]:
        exact &= merged[f"{c}_new"].eq(merged[f"{c}_retained"])
    return {
        "reconstructed_rows": len(actual),
        "retained_rows": len(retained),
        "exact_rows": int(exact.sum()),
        "non_exact_or_unmatched_rows": int((~exact).sum()),
    }


def add_storm_to_events(events: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    return events.merge(members[["event_id", "storm_id"]], on="event_id", how="left", validate="one_to_one")


def event_crosswalk(base_events: pd.DataFrame, nofill_events: pd.DataFrame) -> pd.DataFrame:
    """Map no-fill station events to the containing baseline station event."""
    output = []
    by_station = {s: g.sort_values("event_start") for s, g in base_events.groupby("stationid")}
    for station, group in nofill_events.groupby("stationid"):
        candidates = by_station.get(station)
        if candidates is None:
            continue
        starts = candidates["event_start"].to_numpy(dtype="datetime64[ns]")
        ends = candidates["event_end"].to_numpy(dtype="datetime64[ns]")
        ids = candidates["event_id"].astype(str).to_numpy()
        for row in group.itertuples(index=False):
            t = np.datetime64(row.event_start)
            idx = int(np.searchsorted(starts, t, side="right") - 1)
            if idx < 0 or t > ends[idx] or np.datetime64(row.event_end) > ends[idx]:
                raise RuntimeError(f"No containing baseline event for {row.event_id}")
            output.append({"baseline_event_id": ids[idx], "no_fill_event_id": row.event_id})
    return pd.DataFrame(output)


def main() -> None:
    a = args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if a.parquet_dataset:
        base_events, nofill_events, totals = reconstruct_events_parquet(
            a.parquet_dataset, a.workers
        )
        rain_input = a.parquet_dataset
    elif a.rain:
        base_events, nofill_events, totals = reconstruct_events(a.rain, a.chunksize)
        rain_input = a.rain
    else:
        raise ValueError("Provide --parquet-dataset or --rain")
    retained_events = pd.read_parquet(a.retained_events)
    event_validation = compare_events(base_events, retained_events)
    if event_validation["reconstructed_only"] or event_validation["retained_only"]:
        raise RuntimeError(f"Baseline event reconstruction failed: {event_validation}")

    cluster = load_cluster_module(a.cluster_script)
    base_storms_raw, base_members_raw = cluster.build_storm_clusters(
        cluster_ready(base_events), merge_gap_minutes=STORM_MERGE_GAP_MINUTES
    )
    nofill_storms_raw, nofill_members_raw = cluster.build_storm_clusters(
        cluster_ready(nofill_events), merge_gap_minutes=STORM_MERGE_GAP_MINUTES
    )
    base_storms, base_members = cluster.filter_storms(base_storms_raw, base_members_raw, 2, 2)
    nofill_storms, nofill_members = cluster.filter_storms(nofill_storms_raw, nofill_members_raw, 2, 2)

    retained_storms = pd.read_parquet(a.retained_storms)
    storm_validation = compare_storms(base_storms, retained_storms)
    if storm_validation["non_exact_or_unmatched_rows"]:
        raise RuntimeError(f"Baseline storm reconstruction failed: {storm_validation}")

    base_events = add_storm_to_events(base_events, base_members_raw)
    nofill_events = add_storm_to_events(nofill_events, nofill_members_raw)
    xwalk = event_crosswalk(base_events, nofill_events)
    xwalk = xwalk.merge(
        base_events[["event_id", "storm_id"]].rename(columns={"event_id": "baseline_event_id", "storm_id": "baseline_storm_id"}),
        on="baseline_event_id", how="left", validate="many_to_one",
    ).merge(
        nofill_events[["event_id", "storm_id"]].rename(columns={"event_id": "no_fill_event_id", "storm_id": "no_fill_storm_id"}),
        on="no_fill_event_id", how="left", validate="many_to_one",
    )

    base_to_nf = xwalk.groupby("baseline_storm_id")["no_fill_storm_id"].agg(lambda s: sorted(set(s.dropna().astype(str))))
    nf_to_base = xwalk.groupby("no_fill_storm_id")["baseline_storm_id"].agg(lambda s: sorted(set(s.dropna().astype(str))))

    final = pd.read_csv(a.final_storms, parse_dates=["storm_start", "storm_end"])
    base_index = base_storms_raw.set_index("storm_id")
    nf_index = nofill_storms_raw.set_index("storm_id")
    detail = []
    for row in final.itertuples(index=False):
        bid = str(row.storm_id)
        mapped = base_to_nf.get(bid, [])
        rec = {
            "baseline_storm_id": bid,
            "baseline_label": str(row.convective_rule_label),
            "baseline_start": row.storm_start,
            "baseline_end": row.storm_end,
            "no_fill_storm_count": len(mapped),
            "no_fill_storm_ids": "|".join(mapped),
            "split": len(mapped) > 1,
            "dropped": len(mapped) == 0,
        }
        if len(mapped) == 1:
            nr = nf_index.loc[mapped[0]]
            rec.update({
                "no_fill_start": nr.storm_start,
                "no_fill_end": nr.storm_end,
                "start_shift_minutes": (nr.storm_start - row.storm_start).total_seconds() / 60,
                "end_shift_minutes": (nr.storm_end - row.storm_end).total_seconds() / 60,
                "no_fill_member_events": int(nr.n_member_events),
                "no_fill_member_stations": int(nr.n_member_stations),
            })
        detail.append(rec)
    detail_df = pd.DataFrame(detail)

    reverse_merges = {k: v for k, v in nf_to_base.items() if len(v) > 1}
    summary = {
        "inputs": {
            "rain": rain_input,
            "retained_events": a.retained_events,
            "retained_storms": a.retained_storms,
            "final_storms": a.final_storms,
            "cluster_script": a.cluster_script,
        },
        "sha256": {p: sha256(p) for p in [a.retained_events, a.retained_storms, a.final_storms, a.cluster_script]},
        "parameters": {"rain_threshold_mm_per_15min": THRESHOLD, "station_event_gap_minutes": 120, "storm_merge_gap_minutes": STORM_MERGE_GAP_MINUTES, "retained_min_member_events": 2, "retained_min_member_stations": 2},
        "archive_counts": totals,
        "event_validation": event_validation,
        "storm_validation": storm_validation,
        "counts": {
            "baseline_station_events": len(base_events),
            "no_fill_station_events": len(nofill_events),
            "baseline_raw_storms": len(base_storms_raw),
            "no_fill_raw_storms": len(nofill_storms_raw),
            "baseline_retained_storms": len(base_storms),
            "no_fill_retained_storms": len(nofill_storms),
            "final_storms": len(final),
            "final_storms_split": int(detail_df["split"].sum()),
            "final_storms_dropped": int(detail_df["dropped"].sum()),
            "final_storms_start_shifted": int(detail_df["start_shift_minutes"].fillna(0).ne(0).sum()),
            "final_storms_end_shifted": int(detail_df["end_shift_minutes"].fillna(0).ne(0).sum()),
            "no_fill_storms_mapping_to_multiple_baseline_storms": len(reverse_merges),
        },
        "reverse_merge_examples": reverse_merges,
    }

    base_events.drop(columns=["storm_id"]).to_parquet(out / "baseline_station_events.parquet", index=False)
    nofill_events.drop(columns=["storm_id"]).to_parquet(out / "no_fill_station_events.parquet", index=False)
    nofill_storms.to_parquet(out / "no_fill_storm_episodes.parquet", index=False)
    nofill_members.to_parquet(out / "no_fill_storm_episode_members.parquet", index=False)
    detail_df.to_csv(out / "final_157_event_linkage_comparison.csv", index=False)
    (out / "event_linkage_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
