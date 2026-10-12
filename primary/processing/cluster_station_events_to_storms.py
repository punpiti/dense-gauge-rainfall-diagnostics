#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cluster_station_events_to_storms.py

Group station-wise rain events into storm episodes using temporal overlap.

Input
-----
rain_events.parquet from extract_rain_events.py
Expected columns:
- event_id
- stationid
- event_start
- event_end

Optional columns used for aggregation if present:
- event_length_minutes
- n_rain_slots
- total_rainfall
- average_rainfall
- max_rainfall
- min_rainfall

Main idea
---------
Two station-events belong to the same storm episode if:
- their time windows overlap, or
- the gap between the current cluster end and the next event start
  is <= merge_gap_minutes

This is a 1D temporal clustering, not a spatial clustering.

Outputs
-------
1) storm_episodes.parquet / .csv
   One row per storm episode

2) storm_episode_members.parquet / .csv
   One row per station-event membership

Examples
--------
python cluster_station_events_to_storms.py \
  --events results/rain_events_stationwise/rain_events.parquet \
  --out-storms results/storm_clusters/storm_episodes.parquet \
  --out-members results/storm_clusters/storm_episode_members.parquet

Use a 30-minute merge gap:
python cluster_station_events_to_storms.py \
  --events results/rain_events_stationwise/rain_events.parquet \
  --merge-gap-minutes 30 \
  --out-storms results/storm_clusters/storm_episodes.parquet \
  --out-members results/storm_clusters/storm_episode_members.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="rain_events.parquet from extract_rain_events.py")
    ap.add_argument(
        "--merge-gap-minutes",
        type=int,
        default=0,
        help="Merge adjacent events into the same storm if the gap is <= this many minutes.",
    )
    ap.add_argument(
        "--min-member-events",
        type=int,
        default=1,
        help="Keep only storm episodes with at least this many member station-events.",
    )
    ap.add_argument(
        "--min-member-stations",
        type=int,
        default=1,
        help="Keep only storm episodes with at least this many unique stations.",
    )
    ap.add_argument("--out-storms", required=True, help="output storm episodes (.parquet or .csv)")
    ap.add_argument("--out-members", required=True, help="output storm membership (.parquet or .csv)")
    return ap.parse_args()


def read_events(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    required = ["event_id", "stationid", "event_start", "event_end"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in events parquet: {missing}")

    df["event_id"] = df["event_id"].astype("string")
    df["stationid"] = df["stationid"].astype("string").str.strip()
    df["event_start"] = pd.to_datetime(df["event_start"], errors="coerce")
    df["event_end"] = pd.to_datetime(df["event_end"], errors="coerce")

    df = df.dropna(subset=["event_id", "stationid", "event_start", "event_end"]).copy()
    df = df[df["event_end"] >= df["event_start"]].copy()

    df = df.sort_values(["event_start", "event_end", "stationid", "event_id"]).reset_index(drop=True)
    return df


def build_storm_clusters(events_df: pd.DataFrame, merge_gap_minutes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    gap = pd.Timedelta(minutes=merge_gap_minutes)

    members = []
    storms = []

    current_events = []
    current_start = None
    current_end = None
    storm_index = 0

    def finalize_current_cluster():
        nonlocal storm_index, current_events, current_start, current_end, storms, members

        if not current_events:
            return

        storm_id = f"STORM_{storm_index:06d}"

        cdf = pd.DataFrame(current_events).copy()
        cdf["storm_id"] = storm_id

        storm_start = cdf["event_start"].min()
        storm_end = cdf["event_end"].max()
        storm_duration_minutes = int((storm_end - storm_start).total_seconds() // 60)

        n_member_events = int(len(cdf))
        n_member_stations = int(cdf["stationid"].nunique())

        row = {
            "storm_id": storm_id,
            "storm_start": storm_start,
            "storm_end": storm_end,
            "storm_duration_minutes": storm_duration_minutes,
            "n_member_events": n_member_events,
            "n_member_stations": n_member_stations,
            "member_stationids": "|".join(sorted(cdf["stationid"].dropna().astype(str).unique().tolist())),
            "member_event_ids": "|".join(cdf["event_id"].dropna().astype(str).tolist()),
        }

        # Optional aggregate fields if available
        if "total_rainfall" in cdf.columns:
            row["sum_total_rainfall"] = float(pd.to_numeric(cdf["total_rainfall"], errors="coerce").fillna(0).sum())
            row["max_total_rainfall"] = float(pd.to_numeric(cdf["total_rainfall"], errors="coerce").max())
        if "max_rainfall" in cdf.columns:
            row["max_peak_rainfall"] = float(pd.to_numeric(cdf["max_rainfall"], errors="coerce").max())
        if "n_rain_slots" in cdf.columns:
            row["sum_n_rain_slots"] = int(pd.to_numeric(cdf["n_rain_slots"], errors="coerce").fillna(0).sum())
        if "event_length_minutes" in cdf.columns:
            row["max_event_length_minutes"] = int(pd.to_numeric(cdf["event_length_minutes"], errors="coerce").max())

        storms.append(row)

        mem_cols = ["storm_id", "event_id", "stationid", "event_start", "event_end"]
        extra_cols = [
            c for c in [
                "event_length_minutes",
                "n_rain_slots",
                "total_rainfall",
                "average_rainfall",
                "max_rainfall",
                "min_rainfall",
            ] if c in cdf.columns
        ]
        members.append(cdf[mem_cols + extra_cols].copy())

        storm_index += 1
        current_events = []
        current_start = None
        current_end = None

    for _, row in events_df.iterrows():
        ev = row.to_dict()
        ev_start = pd.Timestamp(ev["event_start"])
        ev_end = pd.Timestamp(ev["event_end"])

        if current_start is None:
            current_events = [ev]
            current_start = ev_start
            current_end = ev_end
            continue

        # overlap or gap <= merge_gap
        if ev_start <= (current_end + gap):
            current_events.append(ev)
            if ev_end > current_end:
                current_end = ev_end
        else:
            finalize_current_cluster()
            current_events = [ev]
            current_start = ev_start
            current_end = ev_end

    finalize_current_cluster()

    storms_df = pd.DataFrame(storms)
    members_df = pd.concat(members, ignore_index=True) if members else pd.DataFrame()

    if not storms_df.empty:
        storms_df = storms_df.sort_values(["storm_start", "storm_end"]).reset_index(drop=True)

    if not members_df.empty:
        members_df = members_df.sort_values(["storm_id", "event_start", "stationid"]).reset_index(drop=True)

    return storms_df, members_df


def filter_storms(
    storms_df: pd.DataFrame,
    members_df: pd.DataFrame,
    min_member_events: int,
    min_member_stations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if storms_df.empty:
        return storms_df, members_df

    keep = storms_df[
        (storms_df["n_member_events"] >= min_member_events) &
        (storms_df["n_member_stations"] >= min_member_stations)
    ].copy()

    keep_ids = set(keep["storm_id"].astype(str).tolist())

    if members_df.empty:
        return keep.reset_index(drop=True), members_df

    mem = members_df[members_df["storm_id"].astype(str).isin(keep_ids)].copy()
    return keep.reset_index(drop=True), mem.reset_index(drop=True)


def save_df(df: pd.DataFrame, path_str: str) -> None:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()

    events_df = read_events(args.events)
    print(f"[INFO] input station-events       : {len(events_df):,}")

    storms_df, members_df = build_storm_clusters(
        events_df=events_df,
        merge_gap_minutes=args.merge_gap_minutes,
    )

    print(f"[INFO] raw storm episodes        : {len(storms_df):,}")

    storms_df, members_df = filter_storms(
        storms_df=storms_df,
        members_df=members_df,
        min_member_events=args.min_member_events,
        min_member_stations=args.min_member_stations,
    )

    print(f"[INFO] filtered storm episodes   : {len(storms_df):,}")

    if not storms_df.empty:
        print(f"[INFO] mean member events/storm  : {storms_df['n_member_events'].mean():.2f}")
        print(f"[INFO] mean stations/storm       : {storms_df['n_member_stations'].mean():.2f}")
        print(f"[INFO] max member events/storm   : {storms_df['n_member_events'].max():,}")
        print(f"[INFO] max stations/storm        : {storms_df['n_member_stations'].max():,}")

    save_df(storms_df, args.out_storms)
    save_df(members_df, args.out_members)

    print(f"[DONE] saved storms  : {args.out_storms}")
    print(f"[DONE] saved members : {args.out_members}")


if __name__ == "__main__":
    main()