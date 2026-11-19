"""Weather source for UmuhinziEnv.

Two modes:

1. Historical (preferred): if ``data/weather_daily.csv`` exists (created by
   ``scripts/fetch_weather.py`` from the Open-Meteo archive), each episode
   replays one real Season A (20 Aug -> early Mar) for the chosen site.
   Seasons <= 2015 are used for training (the last 3 of them for validation),
   2016+ are held out as the test set.

2. Synthetic (fallback): a stochastic daily generator calibrated to an
   approximate Northern-Province (Musanze) bimodal climatology. Every
   "weather year" is an integer id that seeds its own RNG, so the same id
   always produces the same season. Training, validation and test ids never
   overlap: validation picks checkpoints/runs, test is only used for reporting.

Both modes return a :class:`Season` with daily rainfall (mm) and mean
temperature (deg C) arrays of equal length.
"""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SEASON_START = (8, 20)  # 20 August: before Season A onset
SEASON_DAYS = 198  # 66 decision steps of 3 days -> ~6 March

# Approximate monthly rainfall (mm) for a highland Northern-Province site.
# Index 0 = January. Used only by the synthetic generator.
MONTHLY_RAIN_MM = np.array([75, 95, 140, 180, 130, 30, 20, 50, 100, 130, 140, 100], dtype=float)
MEAN_TEMP_C = 19.0

TRAIN_YEAR_IDS = range(0, 1000)
VAL_YEAR_IDS = range(5_000, 5_030)  # checkpoint / run selection
EVAL_YEAR_IDS = range(10_000, 10_050)  # final test, never used for selection
HIST_SPLIT_YEAR = 2015
HIST_VAL_YEARS = 3  # last N training seasons held out for validation

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "weather_daily.csv"


@dataclass
class Season:
    year: int
    rain: np.ndarray  # mm/day, length SEASON_DAYS + 14 (extra days for forecasts)
    temp: np.ndarray  # deg C
    source: str  # "synthetic" or "historical"
    start_date: dt.date


def _month_of_day(start: dt.date, d: int) -> int:
    return (start + dt.timedelta(days=d)).month - 1


def synthetic_season(year_id: int, rain_scale: float = 1.0) -> Season:
    """Generate one deterministic synthetic season for ``year_id``."""
    rng = np.random.default_rng(1_000_003 * (year_id + 17))
    n = SEASON_DAYS + 14
    start = dt.date(2001, *SEASON_START)

    onset_shift = int(np.clip(rng.normal(0, 10), -20, 25))  # days late (+) or early (-)
    wetness = float(np.exp(rng.normal(0, 0.2))) * rain_scale  # ENSO-like year factor

    mean_daily = np.empty(n)
    for d in range(n):
        # Shift the Aug-Oct ramp by the onset anomaly: a late onset means the
        # first weeks look like August for longer.
        shifted = d - onset_shift if d < 75 else d
        shifted = max(shifted, 0)
        m = _month_of_day(start, shifted)
        mean_daily[d] = MONTHLY_RAIN_MM[m] / 30.0
    if rng.random() < 0.35:
        # False onset: a burst of rain before the real onset, then dry.
        burst_day = int(rng.integers(3, 20))
        mean_daily[burst_day : burst_day + 3] += 8.0
        mean_daily[burst_day + 3 : burst_day + 18] *= 0.25
    if rng.random() < 0.45:
        # Mid-season dry spell of 10-25 days.
        s = int(rng.integers(35, 150))
        mean_daily[s : s + int(rng.integers(10, 26))] *= 0.15
    mean_daily *= wetness

    p_wet = np.clip(mean_daily / 9.0 + 0.05, 0.03, 0.85)
    wet = rng.random(n) < p_wet
    shape = 0.8
    scale = mean_daily / (p_wet * shape)
    rain = np.where(wet, rng.gamma(shape, np.maximum(scale, 1e-6)), 0.0)
    rain = np.clip(rain, 0, 90)

    seasonal = 0.8 * np.sin(np.linspace(0, 2 * np.pi, n))
    temp = MEAN_TEMP_C + seasonal + rng.normal(0, 1.2, n) + rng.normal(0, 0.6)
    return Season(year_id, rain.astype(np.float32), temp.astype(np.float32), "synthetic", start)


class HistoricalWeather:
    """Replays real seasons from an Open-Meteo daily CSV."""

    def __init__(self, csv_path: Path):
        rows = list(csv.DictReader(open(csv_path, newline="")))
        self.by_date = {
            dt.date.fromisoformat(r["date"]): (
                float(r["precipitation_sum"] or 0.0),
                float(r["temperature_2m_mean"] or MEAN_TEMP_C),
            )
            for r in rows
        }
        years = sorted({d.year for d in self.by_date})
        n = SEASON_DAYS + 14
        self.years = [
            y
            for y in years
            if dt.date(y, *SEASON_START) in self.by_date
            and dt.date(y, *SEASON_START) + dt.timedelta(days=n - 1) in self.by_date
        ]
        pre = [y for y in self.years if y <= HIST_SPLIT_YEAR]
        self.train_years = pre[:-HIST_VAL_YEARS]
        self.val_years = pre[-HIST_VAL_YEARS:]
        self.eval_years = [y for y in self.years if y > HIST_SPLIT_YEAR]

    def season(self, year: int, rain_scale: float = 1.0) -> Season:
        start = dt.date(year, *SEASON_START)
        n = SEASON_DAYS + 14
        vals = [self.by_date[start + dt.timedelta(days=i)] for i in range(n)]
        rain = np.array([v[0] for v in vals], dtype=np.float32) * rain_scale
        temp = np.array([v[1] for v in vals], dtype=np.float32)
        return Season(year, rain, temp, "historical", start)


class WeatherProvider:
    """Chooses historical data when available, otherwise synthetic seasons."""

    def __init__(self, csv_path: Path | None = None, force_synthetic: bool = False):
        path = Path(csv_path) if csv_path else DEFAULT_CSV
        self.hist = None
        if not force_synthetic and path.exists():
            try:
                h = HistoricalWeather(path)
                if len(h.train_years) >= 5 and len(h.eval_years) >= 2:
                    self.hist = h
            except (KeyError, ValueError):
                self.hist = None

    @property
    def source(self) -> str:
        return "historical" if self.hist else "synthetic"

    def year_ids(self, split: str) -> list[int]:
        if self.hist:
            return {"train": self.hist.train_years, "val": self.hist.val_years, "eval": self.hist.eval_years}[split]
        return list({"train": TRAIN_YEAR_IDS, "val": VAL_YEAR_IDS, "eval": EVAL_YEAR_IDS}[split])

    def season(self, year: int, rain_scale: float = 1.0) -> Season:
        if self.hist:
            return self.hist.season(year, rain_scale)
        return synthetic_season(year, rain_scale)
