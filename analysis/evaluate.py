"""Final comparison: best model of each algorithm vs rule-based baselines on
held-out weather years, in normal and drought (60% rain) conditions.

    uv run main.py evaluate        -> logs/final_comparison.csv (+ printed table)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from environment.baselines import BASELINES
from environment.custom_env import UmuhinziEnv
from training.common import LOG_DIR, MODEL_DIR, evaluate, load_policy_net

ALGOS = ["dqn", "reinforce", "ppo", "a2c"]


def _baseline_eval(name: str, n: int, rain_scale: float, seed: int = 12345) -> dict:
    env = UmuhinziEnv(split="eval", rain_scale=rain_scale)
    pol = BASELINES[name]()
    R, H, P, F = [], [], [], 0
    for i in range(n):
        obs, _ = env.reset(seed=seed + i, options={"year": env.year_ids[i % len(env.year_ids)]})
        done = False
        while not done:
            obs, _, te, tr, _ = env.step(pol(env, obs))
            done = te or tr
        R.append(env.ep_return)
        H.append(env.harvest_kg)
        P.append(env.totals["revenue"] - env.totals["costs"])
        F += env.harvest_kg < 1
    return {"eval_return_mean": np.mean(R), "eval_return_std": np.std(R), "eval_harvest_kg": np.mean(H),
            "eval_profit_krwf": np.mean(P), "eval_failure_rate": F / n}


def run(n_episodes: int = 50) -> list[dict]:
    rows = []
    for algo in ALGOS:
        p = MODEL_DIR / ("dqn" if algo == "dqn" else "pg") / f"best_{algo}.pt"
        if not p.exists():
            print(f"[skip] no {p.name} yet - train/sweep {algo} first")
            continue
        net, ck = load_policy_net(p)
        for cond, scale in (("normal", 1.0), ("drought", 0.6)):
            ev = evaluate(net, n_episodes=n_episodes, rain_scale=scale)
            rows.append({"policy": algo, "run": ck["run_name"], "condition": cond, **ev})
    for name in ("threshold_farmer", "calendar_farmer", "random"):
        for cond, scale in (("normal", 1.0), ("drought", 0.6)):
            rows.append({"policy": name, "run": "-", "condition": cond, **_baseline_eval(name, n_episodes, scale)})

    out = LOG_DIR / "final_comparison.csv"
    keys = ["policy", "run", "condition", "eval_return_mean", "eval_return_std", "eval_harvest_kg",
            "eval_profit_krwf", "eval_failure_rate"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'policy':<18}{'cond':<9}{'return':>14}{'harvest kg':>12}{'profit kRWF':>13}{'fail %':>8}")
    for r in rows:
        print(f"{r['policy']:<18}{r['condition']:<9}{r['eval_return_mean']:>8.2f}±{r['eval_return_std']:<5.2f}"
              f"{r['eval_harvest_kg']:>12.0f}{r['eval_profit_krwf']:>13.1f}{100 * r['eval_failure_rate']:>8.0f}")
    print(f"\nsaved -> {out}")
    return rows


if __name__ == "__main__":
    run()
