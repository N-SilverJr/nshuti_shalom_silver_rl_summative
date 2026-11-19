"""Download real daily weather for a Rwandan site from the Open-Meteo archive
(free, no API key) and save it to data/weather_daily.csv. When that file
exists, UmuhinziEnv replays real seasons instead of synthetic ones
(train: seasons starting <= 2015, held-out eval: 2016+).

    uv run python scripts/fetch_weather.py                 # Musanze (default)
    uv run python scripts/fetch_weather.py --lat -1.95 --lon 30.06 --name kigali
"""

import argparse
import csv
import json
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "weather_daily.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=-1.4996)
    ap.add_argument("--lon", type=float, default=29.6344)
    ap.add_argument("--name", default="musanze")
    ap.add_argument("--start", default="1995-01-01")
    ap.add_argument("--end", default="2025-06-30")
    a = ap.parse_args()
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={a.lat}&longitude={a.lon}&start_date={a.start}&end_date={a.end}"
        "&daily=precipitation_sum,temperature_2m_mean&timezone=Africa%2FKigali"
    )
    print("GET", url)
    data = json.load(urllib.request.urlopen(url, timeout=60))["daily"]
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "precipitation_sum", "temperature_2m_mean"])
        for row in zip(data["time"], data["precipitation_sum"], data["temperature_2m_mean"]):
            w.writerow(row)
    print(f"saved {len(data['time'])} days for {a.name} -> {OUT}")
    print("Note: models trained on synthetic weather should be retrained after switching.")


if __name__ == "__main__":
    main()
