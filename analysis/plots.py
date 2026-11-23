"""Report figures from the sweep logs (all saved to assets/figures/).

    uv run main.py plots

fig1_comparative_rewards.png   held-out return vs frames, best run per algorithm + baselines
fig2_dqn_loss.png              DQN TD loss and mean max-Q for all 10 DQN runs
fig3_pg_entropy.png            policy entropy vs frames for every REINFORCE/PPO/A2C run
fig4_convergence.png           training-episode return (smoothed) vs episodes, best runs
fig5_generalization.png        held-out normal vs drought return, best models + baselines
fig6_sweeps.png                final held-out return of all 10 runs per algorithm
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
OUT = ROOT / "assets" / "figures"

# Validated categorical palette (fixed order, light mode) - one colour per algorithm.
COLORS = {"dqn": "#2a78d6", "reinforce": "#eb6834", "ppo": "#1baf7a", "a2c": "#eda100"}
NAMES = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
ALGOS = ["dqn", "reinforce", "ppo", "a2c"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 150,
})


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path))) if path.exists() else []


def col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) if r.get(key) not in (None, "") else np.nan for r in rows])


def runs(algo: str) -> dict[str, list[dict]]:
    d = LOGS / algo
    return {p.parent.name: read_csv(p) for p in sorted(d.glob("*/train.csv"))} if d.exists() else {}


def best_run(algo: str) -> str | None:
    rows = read_csv(LOGS / algo / "sweep_summary.csv")
    return max(rows, key=lambda r: float(r["best_val_return"]))["run"] if rows else None


def baselines() -> dict[str, dict]:
    rows = read_csv(LOGS / "final_comparison.csv")
    return {(r["policy"], r["condition"]): r for r in rows}


def smooth(y: np.ndarray, k: int = 3) -> np.ndarray:
    if len(y) < k:
        return y
    return np.convolve(np.pad(y, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def fig_comparative():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ends = []
    for a in ALGOS:
        b = best_run(a)
        if not b:
            continue
        r = runs(a)[b]
        x, y = col(r, "frames") / 1000, col(r, "val_return_mean")
        ax.plot(x, smooth(y), color=COLORS[a], label=f"{NAMES[a]} ({b})")
        k = int(np.nanargmax(y))  # checkpoint kept = best validation evaluation
        ax.plot(x[k], smooth(y)[k], "o", ms=8, color=COLORS[a], mec=SURFACE, mew=2, zorder=5)
        ends.append([smooth(y)[-1], x[-1], NAMES[a]])
    # direct end labels, nudged apart so they never overlap
    ends.sort()
    span = np.ptp([e[0] for e in ends]) if ends else 1
    gap = max(0.07 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 1e-3)
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for yv, xv, name in ends:
        ax.annotate(name, (xv, yv), xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=9, va="center")
    _ = span
    base = baselines()
    # baselines are measured on the test years; shown as reference levels
    for name, ls in (("threshold_farmer", "--"), ("calendar_farmer", ":"), ("random", "-.")):
        if (name, "normal") in base:
            v = float(base[(name, "normal")]["eval_return_mean"])
            ax.axhline(v, color=MUTED, ls=ls, lw=1.2)
            ax.text(ax.get_xlim()[0], v, f" {name.replace('_', ' ')}", color=INK2, fontsize=8, va="bottom")
    ax.set(xlabel="training frames (thousands)", ylabel="validation return (greedy, 20 unseen seasons)",
           title="Validation return during training (best run per algorithm)")
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.99, 0.30, "●  checkpoint kept (best validation)\nbaselines: dashed/dotted, test seasons",
            transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    save(fig, "fig1_comparative_rewards.png")


def fig_dqn_loss():
    rs = runs("dqn")
    if not rs:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    best = best_run("dqn")
    for name, r in rs.items():
        x = col(r, "frames") / 1000
        hl = name == best
        kw = dict(color=COLORS["dqn"] if hl else MUTED, lw=2 if hl else 1, alpha=1 if hl else 0.55,
                  label=f"{name} (best)" if hl else None)
        a1.plot(x, col(r, "loss"), **kw)
        q = col(r, "q_mean")
        a2.plot(x, q, **kw)
        if np.nanmax(q) > 20:  # name the runs whose value estimates blew up
            a2.annotate(name, (x[-1], q[-1]), xytext=(-4, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=8, color=INK2)
    a1.set(xlabel="frames (thousands)", ylabel="mean TD loss", title="DQN objective (TD loss)", yscale="log")
    a2.set(xlabel="frames (thousands)", ylabel="mean max-Q of sampled batch (log)", title="DQN value estimates", yscale="log")
    a1.legend(fontsize=8)
    fig.suptitle("DQN loss and Q-values across all 10 runs (grey) - best run highlighted", fontsize=10, color=INK)
    save(fig, "fig2_dqn_loss.png")


def fig_entropy():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, a in zip(axes, ["reinforce", "ppo", "a2c"]):
        best = best_run(a)
        for name, r in runs(a).items():
            if "entropy" not in r[0]:
                continue
            hl = name == best
            ax.plot(col(r, "frames") / 1000, col(r, "entropy"), color=COLORS[a] if hl else MUTED,
                    lw=2 if hl else 1, alpha=1 if hl else 0.5, label=f"{name} (best)" if hl else None)
        ax.axhline(np.log(8), color=INK2, ls=":", lw=1)
        ax.text(0, np.log(8), " uniform (ln 8)", fontsize=8, color=INK2, va="bottom")
        ax.set(title=NAMES[a], xlabel="frames (thousands)")
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("policy entropy (nats)")
    fig.suptitle("Policy entropy during training - all runs (grey), best highlighted", fontsize=10, color=INK)
    save(fig, "fig3_pg_entropy.png")


def fig_convergence():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for a in ALGOS:
        b = best_run(a)
        if not b:
            continue
        r = runs(a)[b]
        ax.plot(col(r, "episodes"), smooth(col(r, "train_return"), 5), color=COLORS[a], label=NAMES[a])
    ax.set(xlabel="training episodes (seasons)", ylabel="training return (last 20 episodes, smoothed)",
           title="Convergence on training weather years (best runs)")
    ax.legend(fontsize=9)
    save(fig, "fig4_convergence.png")


def fig_generalization():
    rows = read_csv(LOGS / "final_comparison.csv")
    if not rows:
        return
    pols = list(dict.fromkeys(r["policy"] for r in rows))
    by = {(r["policy"], r["condition"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(pols))
    w = 0.38
    for i, cond in enumerate(["normal", "drought"]):
        vals = [float(by[(p, cond)]["eval_return_mean"]) for p in pols]
        errs = [float(by[(p, cond)]["eval_return_std"]) for p in pols]
        colors = [COLORS.get(p, MUTED) for p in pols]
        ax.bar(x + (i - 0.5) * (w + 0.02), vals, w, yerr=errs, color=colors,
               alpha=1.0 if cond == "normal" else 0.45, edgecolor=SURFACE, linewidth=2,
               error_kw=dict(ecolor=INK2, lw=1, capsize=2), hatch=None if cond == "normal" else "//")
    ax.axhline(0, color=INK2, lw=0.8)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(facecolor=MUTED, label="normal rainfall"),
                       Patch(facecolor=MUTED, alpha=0.45, hatch="//", label="drought (−40% rain)")],
              loc="lower left", fontsize=8)
    ax.set_xticks(x, [NAMES.get(p, p.replace("_", "\n")) for p in pols], fontsize=9)
    ax.set(ylabel="test return (mean ± sd)\n50 episodes over 9 real seasons",
           title="Generalisation to unseen real seasons: normal vs drought")
    save(fig, "fig5_generalization.png")


def fig_sweeps():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), gridspec_kw={"wspace": 0.85})
    for ax, a in zip(axes, ALGOS):
        rows = read_csv(LOGS / a / "sweep_summary.csv")
        if not rows:
            ax.set_title(NAMES[a])
            continue
        rows.sort(key=lambda r: r["run"])
        v = [float(r["test_return"]) for r in rows]
        best = int(np.argmax([float(r["best_val_return"]) for r in rows]))
        ax.barh(range(len(v)), v, color=[COLORS[a] if i == best else MUTED for i in range(len(v))],
                edgecolor=SURFACE, linewidth=2)
        ax.set_yticks(range(len(v)), [r["run"] for r in rows], fontsize=7.5)  # each panel its own run names
        ax.invert_yaxis()
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set(title=NAMES[a], xlabel="test return (best-val checkpoint)")
    fig.suptitle("Hyperparameter sweeps - test return of each run (run selected on validation highlighted)", fontsize=10, color=INK)
    save(fig, "fig6_sweeps.png")


def main():
    for f in (fig_comparative, fig_dqn_loss, fig_entropy, fig_convergence, fig_generalization, fig_sweeps):
        try:
            f()
        except Exception as e:  # keep going so partial results still plot
            print(f"[warn] {f.__name__}: {e}")
    summary = {a: best_run(a) for a in ALGOS}
    json.dump(summary, open(OUT / "best_runs.json", "w"), indent=2)


if __name__ == "__main__":
    main()
