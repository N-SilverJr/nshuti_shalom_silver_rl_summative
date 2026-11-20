"""Download real daily weather for a Rwandan site from the Open-Meteo archive
(free, no API key) and save it to data/weather_daily.csv.

IMPORTANT: we request a single, consistent reanalysis (ERA5-Seamless =
ERA5 + ERA5-Land). The API's default "best match" silently switches to the
ECMWF IFS model from 2017 onwards (at Musanze this roughly halved the
rainfall from that date) and would make any train/test split across 2017 meaningless.

When the CSV exists, UmuhinziEnv
  * calibrates its stochastic weather generator on seasons before 2016-08-20
    (training + validation), and
  * replays the real seasons 2016/17 ... 2024/25 as the held-out test set.

    uv run python scripts/fetch_weather.py                 # Nyagatare (default)
    uv run python scripts/fetch_weather.py --lat -1.4996 --lon 29.6344 --name musanze
"""

import argparse
import csv
import json
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "weather_daily.csv"
MODELS = ["era5_seamless", "era5"]  # first one that the API accepts is used


def fetch(lat, lon, start, end, model):
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
        "&daily=precipitation_sum,temperature_2m_mean&timezone=Africa%2FKigali"
        f"&models={model}"
    )
    print("GET", url)
    try:
        payload = json.load(urllib.request.urlopen(url, timeout=120))
    except urllib.error.HTTPError as e:
        print(f"  {model} rejected: {e.read().decode(errors='ignore')[:200]}")
        return None
    daily = payload.get("daily", {})
    # some API versions suffix variable names with the model name
    def pick(var):
        for k, v in daily.items():
            if k.startswith(var):
                return v
        return None

    p, t = pick("precipitation_sum"), pick("temperature_2m_mean")
    if not daily.get("time") or p is None or t is None or all(v is None for v in p):
        print(f"  {model}: no data returned")
        return None
    return daily["time"], p, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=-1.2938)  # Nyagatare, Eastern Province (maize belt)
    ap.add_argument("--lon", type=float, default=30.3275)
    ap.add_argument("--name", default="nyagatare")
    ap.add_argument("--start", default="1995-01-01")
    ap.add_argument("--end", default="2025-06-30")
    a = ap.parse_args()
    for model in MODELS:
        got = fetch(a.lat, a.lon, a.start, a.end, model)
        if got:
            break
    else:
        raise SystemExit("Could not download weather data - check your internet connection.")
    time, prec, temp = got
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "precipitation_sum", "temperature_2m_mean"])
        n_missing = 0
        for row in zip(time, prec, temp):
            n_missing += row[1] is None
            w.writerow(["" if v is None else v for v in row])
    meta = {"site": a.name, "lat": a.lat, "lon": a.lon, "model": model, "start": a.start, "end": a.end,
            "source": "Open-Meteo Historical Weather API (archive-api.open-meteo.com)"}
    json.dump(meta, open(OUT.with_suffix(".json"), "w"), indent=2)
    print(f"saved {len(time)} days for {a.name} using model={model} -> {OUT} ({n_missing} missing rain values)")
    print("Next: uv run python scripts/check_weather.py   (checks the data and the calibrated generator)")


if __name__ == "__main__":
    main()
