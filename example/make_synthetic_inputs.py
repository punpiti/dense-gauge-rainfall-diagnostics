#!/usr/bin/env python3
"""Create unrestricted smoke-test inputs for the released processing code."""

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    out = Path(__file__).resolve().parent / "generated"
    out.mkdir(parents=True, exist_ok=True)

    station_rows = []
    station_id = 1
    for iy in range(6):
        for ix in range(6):
            station_rows.append(
                {
                    "id": station_id,
                    "nameShort": f"S{station_id:02d}",
                    "lat": 13.64 + 0.03 * iy,
                    "lon": 100.43 + 0.03 * ix,
                }
            )
            station_id += 1
    stations = pd.DataFrame(station_rows)
    stations.to_csv(out / "stations.csv", index=False)

    times = pd.date_range("2020-01-01T00:00:00", periods=15, freq="15min")
    rows = []
    coords = stations[["lat", "lon"]].to_numpy(dtype=float)
    for k, timestamp in enumerate(times):
        center = np.array([13.66 + 0.006 * k, 100.45 + 0.007 * k])
        dist2 = ((coords - center) ** 2).sum(axis=1)
        rain = 10.0 * np.exp(-dist2 / (2.0 * 0.050**2))
        # A broad, increasing pulse creates a transparent positive-residual
        # example that passes the manuscript's operational rule. It is not
        # intended to emulate Bangkok climatology.
        if k >= 1:
            growth_dist2 = ((coords - np.array([13.715, 100.505])) ** 2).sum(axis=1)
            rain += (2.0 * k) * np.exp(-growth_dist2 / (2.0 * 0.085**2))
        for station_id, value in zip(stations["id"], rain):
            rows.append(
                {
                    "time": timestamp,
                    "stationid": int(station_id),
                    "rainfall": float(value),
                    "rainfall_flag": "N",
                }
            )

    pd.DataFrame(rows).to_parquet(out / "rainfall.parquet", index=False)

    print(f"  station metadata      : {out / 'stations.csv'}")
    print(f"  rainfall observations : {out / 'rainfall.parquet'}")


if __name__ == "__main__":
    main()
