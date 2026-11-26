"""Run a trained agent in the 3D simulation with verbose terminal output.

    uv run play.py                          # best agent, demo pacing (holds on each SMS)
    uv run play.py --pace steady --speed 3  # fixed speed instead
    uv run play.py --algo ppo --episodes 2
    uv run play.py --algo dqn --drought     # 40% less rain (stress test)
    uv run play.py --algo threshold_farmer  # rule-based baseline for comparison
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from environment.baselines import BASELINES
from environment.custom_env import ACTION_LABELS, Action, UmuhinziEnv
from environment.rendering import get_viewer

ROOT = Path(__file__).resolve().parent
ALGOS = ["dqn", "ppo", "a2c", "reinforce"]

C = {"g": "\033[92m", "y": "\033[93m", "r": "\033[91m", "c": "\033[96m", "b": "\033[1m", "d": "\033[2m", "x": "\033[0m"}


def find_model(algo: str | None) -> tuple[str, Path | None]:
    """Return (algo, checkpoint). With algo=None pick the best promoted model."""
    cands = []
    for a in [algo] if algo else ALGOS:
        sub = "dqn" if a == "dqn" else "pg"
        p = ROOT / "models" / sub / f"best_{a}.pt"
        if p.exists():
            m = torch.load(p, map_location="cpu", weights_only=False)["metrics"]
            cands.append((m.get("best_val_return", -1e9), a, p))
    if not cands:
        return algo or "threshold_farmer", None
    _, a, p = max(cands)
    return a, p


def make_policy(algo: str, ckpt: Path | None, mode: str):
    if ckpt is None:
        base = BASELINES[algo if algo in BASELINES else "threshold_farmer"]()
        return base.name, (lambda env, obs: int(base(env, obs))), None
    from training.common import load_policy_net

    net, meta = load_policy_net(ckpt)
    gen = torch.Generator().manual_seed(0)

    def raw(obs: np.ndarray) -> int:
        with torch.no_grad():
            out = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        if mode == "sample" and algo != "dqn":
            return int(torch.multinomial(torch.softmax(out, -1), 1, generator=gen).item())
        return int(out.argmax(-1).item())

    label = f"{algo.upper()} ({meta['run_name']})"
    return label, (lambda env, obs: raw(obs)), raw


PART_LABELS = {
    "cost": "cost",
    "toxicity": "toxicity",
    "progress": "crop progress",
    "harvest": "harvest value",
    "invalid": "invalid advice",
    "failure": "season failed",
}


def explain_reward(parts: dict) -> str:
    """'cost -0.25 · toxicity -0.60 · crop progress +0.15' (non-zero parts only)."""
    out = []
    for k, label in PART_LABELS.items():
        v = parts.get(k, 0.0)
        if abs(v) < 0.005:
            continue
        if k == "progress" and parts.get("harvest", 0.0) > 0 and v < 0:
            label = "progress points cashed in"
        out.append(f"{label} {v:+.2f}")
    return " · ".join(out) if out else "nothing happened"


def is_highlight(a: int, s: dict, done: bool) -> bool:
    """Steps worth pausing on: a real SMS was sent, or something important happened."""
    ev = s["event"]
    return a != Action.WAIT or done or "!" in ev or "Germination" in ev or "lost" in ev or "stress" in ev


def bar(v: float, width: int = 10) -> str:
    n = int(round(max(0.0, min(1.0, v)) * width))
    return "█" * n + "·" * (width - n)


def play(args):
    algo, ckpt = find_model(None if args.algo in (None, "best") else args.algo)
    if args.algo in BASELINES:
        algo, ckpt = args.algo, None
    label, policy, raw = make_policy(algo, ckpt, args.mode)
    speed = args.speed or (5.0 if args.pace == "demo" else 3.0)

    env = UmuhinziEnv(split="eval", rain_scale=0.6 if args.drought else 1.0)
    viewer = None
    if not args.headless:
        viewer = get_viewer(port=args.port, open_browser=not args.no_browser)
        viewer.policy_fn = raw
        viewer.set_meta(agent=label, rain_scale=env.rain_scale, pace=args.pace, hold=args.hold)

    print(f"{C['b']}Umuhinzi advisory agent{C['x']}  agent={label}  checkpoint={ckpt or 'rule-based baseline'}")
    print(f"weather={env.weather.source}  split=held-out eval years  rain_scale={env.rain_scale}  mode={args.mode}")
    print(f"{C['d']}score: 1 reward point = 10,000 RWF · pace={args.pace} "
          f"({'monitoring steps fast-forward, each SMS advisory is held ' + str(args.hold) + 's' if args.pace == 'demo' else str(speed) + ' steps/s'}){C['x']}")
    if viewer and args.wait > 0:
        print(f"{C['d']}waiting {args.wait}s for the browser to connect…{C['x']}")
        time.sleep(args.wait)

    results = []
    first = max(args.start_episode, 1) - 1  # replay episode N of a longer run exactly (same start-state seed)
    for ep in range(first, first + args.episodes):
        year = env.year_ids[(args.year_index + ep - first) % len(env.year_ids)]
        obs, _ = env.reset(seed=1000 + ep, options={"year": year})
        if viewer:
            viewer.set_meta(episode=ep + 1)
            viewer.push(env.state_dict())
            if args.pace == "demo":
                time.sleep(min(args.hold, 2.0))  # let the viewer show the new season
        print(f"\n{C['c']}=== Episode {ep + 1}/{first + args.episodes} · weather year {year} ==={C['x']}")
        print(f"{C['d']}{'date':>7} {'t':>3}  {'advisory':<28} {'reward':>7}  soil        crop-health  armyworm(true/est)  budget  event{C['x']}")
        done = False
        while not done:
            a = policy(env, obs)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            s = env.state_dict()
            colour = C["g"] if r > 0.05 else C["r"] if r < -0.3 else ""
            act = ACTION_LABELS[Action(a)]
            highlight = is_highlight(a, s, done)
            if not highlight and not args.all_steps:
                act_disp = f"{C['d']}{act:<28}{C['x']}"
            else:
                act_disp = f"{C['y']}{act:<28}{C['x']}"
            print(
                f"{s['date']:>7} {s['t']:>3}  {act_disp} {colour}{r:+7.2f}{C['x']}  "
                f"{bar(s['soil_moisture'])} {bar(s['health']) if s['planted'] else '·' * 10}  "
                f"{s['pest_true']:.2f}/{s['pest_estimate']:.2f}           {s['budget']:5.1f}  {s['event']}"
            )
            if highlight:
                print(f"{C['d']}{'':>13}↳ reward {r:+.2f} = {explain_reward(info['reward_parts'])}{C['x']}")
            if viewer:
                viewer.push(s)
                if args.pace == "demo":
                    time.sleep(args.hold if highlight else 1.0 / speed)
                else:
                    time.sleep(1.0 / speed)
        prof = env.totals["revenue"] - env.totals["costs"]
        results.append((env.ep_return, env.harvest_kg, prof))
        print(
            f"{C['b']}Episode {ep + 1}: return {env.ep_return:+.2f} · harvest {env.harvest_kg:.0f} kg · "
            f"profit {prof:.1f} kRWF · cheap sprays {env.cheap_sprays} · toxicity {env.totals['toxicity']:.1f}{C['x']}"
        )
        if viewer:
            time.sleep(2.5)
    R = np.array(results)
    print(f"\n{C['b']}Summary over {len(R)} held-out seasons:{C['x']} mean return {R[:, 0].mean():+.2f} · "
          f"mean harvest {R[:, 1].mean():.0f} kg · mean profit {R[:, 2].mean():.1f} kRWF")
    if viewer and not args.exit:
        print(f"{C['d']}Simulation still served at {viewer.url} — press Ctrl+C to quit.{C['x']}")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


def add_play_args(ap: argparse.ArgumentParser):
    ap.add_argument("--algo", default="best", help="best | dqn | ppo | a2c | reinforce | " + " | ".join(BASELINES))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--pace", choices=["demo", "steady"], default="demo",
                    help="demo: fast-forward monitoring, hold on each SMS advisory; steady: fixed speed")
    ap.add_argument("--speed", type=float, default=None,
                    help="steps per second (demo: monitoring steps, default 5; steady: all steps, default 3)")
    ap.add_argument("--hold", type=float, default=2.5, help="demo pace: seconds to hold on each advisory")
    ap.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    ap.add_argument("--drought", action="store_true", help="evaluate with 40%% less rain")
    ap.add_argument("--year-index", type=int, default=0)
    ap.add_argument("--start-episode", type=int, default=1,
                    help="number the first season as episode N and reuse that episode's start-state seed, "
                         "so one season can be replayed exactly as it played inside a longer run")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--wait", type=float, default=4.0, help="seconds to wait for the browser before starting")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--headless", action="store_true", help="terminal only, no 3D viewer")
    ap.add_argument("--all-steps", action="store_true", help="highlight every step, including monitoring")
    ap.add_argument("--exit", action="store_true", help="exit after the episodes instead of keeping the viewer up")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_play_args(ap)
    play(ap.parse_args())
