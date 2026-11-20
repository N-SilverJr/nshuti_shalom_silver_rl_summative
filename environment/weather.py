"""Weather for UmuhinziEnv: a stochastic generator calibrated on real
Nyagatare data, plus replay of real held-out seasons.

Data: ``data/weather_daily.csv`` - daily precipitation and mean temperature
for Nyagatare (Eastern Province, Rwanda's main maize belt) from the
Open-Meteo archive, ERA5-Seamless reanalysis (see ``scripts/fetch_weather.py``). Days with < 1 mm are treated
as dry (standard wet-day threshold; removes reanalysis drizzle).

Splits (no overlap, nothing from the test period is used for fitting):

    train      generated seasons, ids 0-999      } generator fitted on data
    val        generated seasons, ids 5000-5029  } before 20 Aug 2016
    eval       REAL seasons 2016/17 ... 2024/25  (held-out test set)
    test_gen   generated seasons, ids 10000-10049 (in-distribution test)

Generator (fitted per calendar month):
  * rain occurrence: two-state first-order Markov chain,
    P(wet | yesterday dry) and P(wet | yesterday wet);
  * wet-day amounts: gamma distribution (method of moments);
  * onset variability: the Aug-Nov rainfall calendar is shifted by an onset
    anomaly sampled from the observed onset dates of the calibration years;
  * season wetness: a log-normal season factor whose spread is chosen so the
    generated Sep-Feb totals have the same year-to-year variability as the
    observed ones (daily Markov models alone under-disperse);
  * temperature: monthly mean + AR(1) daily anomaly.

If the CSV is missing, an uncalibrated fallback generator is used.
"""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

SEASON_START = (8, 20)  # 20 August: before Season A onset
SEASON_DAYS = 198  # 66 decision steps of 3 days -> ~6 March
SEASON_LEN = SEASON_DAYS + 14  # extra days so 7-day forecasts exist at the end
WET_MM = 1.0
CALIB_END = dt.date(2016, *SEASON_START)  # first real test season starts here
FIRST_TEST_SEASON = 2016

TRAIN_YEAR_IDS = range(0, 1000)
VAL_YEAR_IDS = range(5_000, 5_030)
TEST_GEN_IDS = range(10_000, 10_050)

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "weather_daily.csv"

# Fallback climatology (only used without the CSV).
FALLBACK_MONTHLY_MM = np.array([150, 135, 190, 175, 115, 28, 16, 57, 105, 175, 195, 175], dtype=float)
FALLBACK_TEMP_C = 18.6


@dataclass
class Season:
    year: int
    rain: np.ndarray  # mm/day, length SEASON_LEN
    temp: np.ndarray  # deg C
    source: str  # "historical" | "generated" | "fallback"
    start_date: dt.date


# ----------------------------------------------------------------- helpers
def onset_day(rain: np.ndarray) -> int | None:
    """Agronomic onset after 20 Aug: first 3-day spell with >= 20 mm that is
    not followed by a 10-day dry spell (< 5 mm) within the next 30 days."""
    for d in range(0, 120):
        if rain[d : d + 3].sum() >= 20 and all(rain[k : k + 10].sum() >= 5 for k in range(d + 3, d + 30)):
            return d
    return None


def season_total(rain: np.ndarray) -> float:
    return float(rain[11:195].sum())  # 1 Sep - end Feb


def load_csv(path: Path):
    rows = list(csv.DictReader(open(path, newline="")))
    dates = np.array([dt.date.fromisoformat(r["date"]) for r in rows])
    rain = np.array([float(r["precipitation_sum"]) if r["precipitation_sum"] not in ("", None) else np.nan for r in rows])
    temp = np.array([float(r["temperature_2m_mean"]) if r["temperature_2m_mean"] not in ("", None) else np.nan for r in rows])
    rain = np.where(np.isnan(rain), 0.0, rain)
    temp = np.where(np.isnan(temp), np.nanmean(temp), temp)
    rain = np.where(rain < WET_MM, 0.0, rain)
    return dates, rain, temp


# --------------------------------------------------------------- climate fit
@dataclass
class Climate:
    p01: np.ndarray  # P(wet | prev dry) per month (index 0 = Jan)
    p11: np.ndarray  # P(wet | prev wet)
    shape: np.ndarray  # gamma shape of wet-day amounts
    scale: np.ndarray  # gamma scale
    t_mean: np.ndarray
    t_sd: float
    t_ar: float
    onset_anoms: np.ndarray  # observed onset - mean onset (days)
    season_sigma: float  # sd of log season factor
    onset_bias: float = 0.0  # days added to every shift so generated mean onset matches observed
    stats: dict = field(default_factory=dict)  # observed calibration statistics


def fit_climate(dates, rain, temp, end: dt.date = CALIB_END) -> Climate:
    keep = dates < end
    dates, rain, temp = dates[keep], rain[keep], temp[keep]
    month = np.array([d.month - 1 for d in dates])
    wet = rain > 0
    p01, p11, shape, scale, t_mean = (np.zeros(12) for _ in range(5))
    for m in range(12):
        idx = np.where(month[1:] == m)[0] + 1
        prev, cur = wet[idx - 1], wet[idx]
        p01[m] = cur[~prev].mean() if (~prev).any() else 0.1
        p11[m] = cur[prev].mean() if prev.any() else 0.5
        amt = rain[(month == m) & wet]
        mu, var = amt.mean(), amt.var()
        shape[m] = np.clip(mu**2 / var, 0.3, 3.0)
        scale[m] = mu / shape[m]
        t_mean[m] = temp[month == m].mean()
    anom = temp - t_mean[month]
    t_ar = float(np.corrcoef(anom[:-1], anom[1:])[0, 1])
    t_sd = float(np.std(anom[1:] - t_ar * anom[:-1]))

    by_date = dict(zip(dates, rain))
    onsets, totals = [], []
    for y in range(dates[0].year, end.year):
        s = dt.date(y, *SEASON_START)
        days = [s + dt.timedelta(days=i) for i in range(SEASON_LEN)]
        if days[0] in by_date and days[-1] in by_date:
            x = np.array([by_date[d] for d in days])
            o = onset_day(x)
            if o is not None:
                onsets.append(o)
            totals.append(season_total(x))
    onsets, totals = np.array(onsets, float), np.array(totals)
    clim = Climate(p01, p11, shape, scale, t_mean, t_sd, t_ar, onsets - onsets.mean(), 0.0)
    clim.stats = {
        "years": f"{dates[0].year}-{end.year} (seasons before {end})",
        "n_seasons": len(totals),
        "onset_mean": float(onsets.mean()),
        "onset_sd": float(onsets.std()),
        "season_total_mean": float(totals.mean()),
        "season_total_logsd": float(np.std(np.log(totals))),
        "monthly_mm": [float(rain[month == m].sum() / len(set(d.year for d in dates[month == m]))) for m in range(12)],
        "monthly_wetdays": [float(wet[month == m].sum() / len(set(d.year for d in dates[month == m]))) for m in range(12)],
    }
    # 1) correct the mean onset date (a few fixed-point iterations)
    for _ in range(3):
        sims = [_generate(clim, 900_000 + i) for i in range(300)]
        gen_on = [o for o in (onset_day(g.rain) for g in sims) if o is not None]
        clim.onset_bias += float(onsets.mean() - np.mean(gen_on))
    # 2) choose the season factor so generated totals match the observed spread
    sims = np.array([season_total(_generate(clim, 900_000 + i).rain) for i in range(300)])
    sim_sd = float(np.std(np.log(sims)))
    clim.season_sigma = float(np.sqrt(max(clim.stats["season_total_logsd"] ** 2 - sim_sd**2, 0.0)))
    return clim


def _generate(clim: Climate, year_id: int, rain_scale: float = 1.0) -> Season:
    rng = np.random.default_rng(1_000_003 * (year_id + 17))
    start = dt.date(2001, *SEASON_START)
    shift = clim.onset_bias + (float(rng.choice(clim.onset_anoms) + rng.normal(0, 4)) if len(clim.onset_anoms) else 0.0)
    factor = float(np.exp(rng.normal(0, clim.season_sigma))) if clim.season_sigma > 0 else 1.0
    rain = np.zeros(SEASON_LEN)
    temp = np.zeros(SEASON_LEN)
    wet_prev = False
    t_anom = rng.normal(0, clim.t_sd)
    for d in range(SEASON_LEN):
        # a late (positive) onset stretches the dry Aug/Sep calendar; the effect fades by ~day 100
        eff = d - shift * max(0.0, 1.0 - d / 100.0)
        m = (start + dt.timedelta(days=int(round(max(eff, -60))))).month - 1
        p = clim.p11[m] if wet_prev else clim.p01[m]
        wet_prev = rng.random() < p
        if wet_prev:
            rain[d] = rng.gamma(clim.shape[m], clim.scale[m])
        m_true = (start + dt.timedelta(days=d)).month - 1
        t_anom = clim.t_ar * t_anom + rng.normal(0, clim.t_sd)
        temp[d] = clim.t_mean[m_true] + t_anom
    rain = np.clip(rain * factor * rain_scale, 0, 150)
    return Season(year_id, rain.astype(np.float32), temp.astype(np.float32), "generated", start)


# ------------------------------------------------------------ fallback only
def fallback_season(year_id: int, rain_scale: float = 1.0) -> Season:
    """Uncalibrated generator used only when the CSV is absent."""
    rng = np.random.default_rng(1_000_003 * (year_id + 17))
    start = dt.date(2001, *SEASON_START)
    shift = int(np.clip(rng.normal(0, 15), -20, 40))
    wetness = float(np.exp(rng.normal(0, 0.2))) * rain_scale
    mean_daily = np.array(
        [FALLBACK_MONTHLY_MM[(start + dt.timedelta(days=max(d - shift, 0) if d < 75 else d)).month - 1] / 30.0
         for d in range(SEASON_LEN)]
    )
    p_wet = np.clip(mean_daily / 9.0 + 0.05, 0.03, 0.85)
    wet = rng.random(SEASON_LEN) < p_wet
    rain = np.where(wet, rng.gamma(0.8, np.maximum(mean_daily / (p_wet * 0.8), 1e-6)), 0.0) * wetness
    temp = FALLBACK_TEMP_C + rng.normal(0, 1.0, SEASON_LEN)
    return Season(year_id, np.clip(rain, 0, 150).astype(np.float32), temp.astype(np.float32), "fallback", start)


# ------------------------------------------------------------------ provider
@lru_cache(maxsize=4)
def _load_provider_data(path_str: str):
    dates, rain, temp = load_csv(Path(path_str))
    clim = fit_climate(dates, rain, temp)
    by_date = {d: (r, t) for d, r, t in zip(dates, rain, temp)}
    test_years = []
    for y in range(FIRST_TEST_SEASON, dates[-1].year + 1):
        s = dt.date(y, *SEASON_START)
        if s in by_date and s + dt.timedelta(days=SEASON_LEN - 1) in by_date:
            test_years.append(y)
    return clim, by_date, test_years


class WeatherProvider:
    def __init__(self, csv_path: Path | None = None, force_synthetic: bool = False):
        path = Path(csv_path) if csv_path else DEFAULT_CSV
        self.calibrated = False
        if not force_synthetic and path.exists():
            self.climate, self._by_date, self.test_years = _load_provider_data(str(path.resolve()))
            self.calibrated = len(self.test_years) > 0
        if not self.calibrated:
            self.test_years = []

    @property
    def source(self) -> str:
        return "calibrated (ERA5) / real test seasons" if self.calibrated else "fallback synthetic"

    def year_ids(self, split: str) -> list[int]:
        if split == "eval" and self.calibrated:
            return list(self.test_years)
        if split == "eval":
            return list(TEST_GEN_IDS)
        return list({"train": TRAIN_YEAR_IDS, "val": VAL_YEAR_IDS, "test_gen": TEST_GEN_IDS}[split])

    def season(self, year: int, rain_scale: float = 1.0) -> Season:
        if self.calibrated and year in self.test_years:
            start = dt.date(year, *SEASON_START)
            vals = [self._by_date[start + dt.timedelta(days=i)] for i in range(SEASON_LEN)]
            rain = np.array([v[0] for v in vals], dtype=np.float32) * rain_scale
            temp = np.array([v[1] for v in vals], dtype=np.float32)
            return Season(year, rain, temp, "historical", start)
        if self.calibrated:
            return _generate(self.climate, year, rain_scale)
        return fallback_season(year, rain_scale)


# --------------------------------------------------------------- diagnostics
def report(csv_path: Path = DEFAULT_CSV):
    """Data quality + calibration check: uv run python scripts/check_weather.py"""
    dates, rain, temp = load_csv(csv_path)
    years = np.array([d.year for d in dates])
    print(f"data: {dates[0]} .. {dates[-1]}  ({len(dates)} days)  file={csv_path}")
    meta = csv_path.with_suffix(".json")
    if meta.exists():
        print("source:", meta.read_text().replace("\n", " "))
    else:
        print("WARNING: no data/weather_daily.json - file may come from the old script (mixed models).")
    ann = {y: rain[years == y].sum() for y in range(dates[0].year, dates[-1].year) if (years == y).sum() >= 365}
    pre = np.mean([v for y, v in ann.items() if y < 2017])
    post = np.mean([v for y, v in ann.items() if y >= 2017] or [np.nan])
    print(f"annual rain (>=1 mm days): before 2017 {pre:.0f} mm/yr, 2017+ {post:.0f} mm/yr")
    if post < 0.75 * pre:
        print("WARNING: rainfall drops sharply in 2017 - this looks like the ECMWF IFS model switch. "
              "Re-run scripts/fetch_weather.py (it requests ERA5-Seamless).")
    clim, by_date, test_years = _load_provider_data(str(csv_path.resolve()))
    st = clim.stats
    print(f"\ncalibration: {st['years']}, {st['n_seasons']} seasons")
    print(f"  onset after 20 Aug: mean {st['onset_mean']:.0f} d, sd {st['onset_sd']:.0f} d | "
          f"Sep-Feb total: mean {st['season_total_mean']:.0f} mm, log-sd {st['season_total_logsd']:.2f} "
          f"| season factor sigma {clim.season_sigma:.2f}, onset bias {clim.onset_bias:+.1f} d")
    gens = [_generate(clim, i) for i in TRAIN_YEAR_IDS[:300]]
    g_tot = np.array([season_total(g.rain) for g in gens])
    g_on = np.array([o for o in (onset_day(g.rain) for g in gens) if o is not None])
    print(f"  generated:   onset mean {g_on.mean():.0f} d, sd {g_on.std():.0f} d | "
          f"Sep-Feb total mean {g_tot.mean():.0f} mm, log-sd {np.std(np.log(g_tot)):.2f}")
    print("\nreal held-out test seasons:")
    for y in test_years:
        x = np.array([by_date[dt.date(y, *SEASON_START) + dt.timedelta(days=i)][0] for i in range(SEASON_LEN)])
        o = onset_day(x)
        print(f"  {y}/{(y + 1) % 100:02d}: Sep-Feb {season_total(x):5.0f} mm, onset "
              f"{(dt.date(y, *SEASON_START) + dt.timedelta(days=o)).strftime('%d %b') if o is not None else '-'}")


if __name__ == "__main__":
    report()
