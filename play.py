"""Run a trained agent in the 3D simulation with verbose terminal output.

    uv run play.py                          # best available agent, opens browser
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


def bar(v: float, width: int = 10) -> str:
    n = int(round(max(0.0, min(1.0, v)) * width))
    return "█" * n + "·" * (width - n)


def play(args):
    algo, ckpt = find_model(None if args.algo in (None, "best") else args.algo)
    if args.algo in BASELINES:
        algo, ckpt = args.algo, None
    label, policy, raw = make_policy(algo, ckpt, args.mode)

    env = UmuhinziEnv(split="eval", rain_scale=0.6 if args.drought else 1.0)
    viewer = None
    if not args.headless:
        viewer = get_viewer(port=args.port, open_browser=not args.no_browser)
        viewer.policy_fn = raw
        viewer.set_meta(agent=label, rain_scale=env.rain_scale)

    print(f"{C['b']}Umuhinzi advisory agent{C['x']}  agent={label}  checkpoint={ckpt or 'rule-based baseline'}")
    print(f"weather={env.weather.source}  split=held-out eval years  rain_scale={env.rain_scale}  mode={args.mode}")
    if viewer and args.wait > 0:
        print(f"{C['d']}waiting {args.wait}s for the browser to connect…{C['x']}")
        time.sleep(args.wait)

    results = []
    for ep in range(args.episodes):
        year = env.year_ids[(args.year_index + ep) % len(env.year_ids)]
        obs, _ = env.reset(seed=1000 + ep, options={"year": year})
        if viewer:
            viewer.set_meta(episode=ep + 1)
            viewer.push(env.state_dict())
        print(f"\n{C['c']}=== Episode {ep + 1}/{args.episodes} · weather year {year} ==={C['x']}")
        print(f"{C['d']}{'date':>7} {'t':>3}  {'advisory':<28} {'reward':>7}  soil        crop-health  armyworm(true/est)  budget  event{C['x']}")
        done = False
        while not done:
            a = policy(env, obs)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            s = env.state_dict()
            colour = C["g"] if r > 0.05 else C["r"] if r < -0.3 else ""
            act = ACTION_LABELS[Action(a)]
            if a == Action.WAIT and not args.all_steps and "!" not in s["event"] and "Germination" not in s["event"] and not done:
                act_disp = f"{C['d']}{act:<28}{C['x']}"
            else:
                act_disp = f"{C['y']}{act:<28}{C['x']}"
            print(
                f"{s['date']:>7} {s['t']:>3}  {act_disp} {colour}{r:+7.2f}{C['x']}  "
                f"{bar(s['soil_moisture'])} {bar(s['health']) if s['planted'] else '·' * 10}  "
                f"{s['pest_true']:.2f}/{s['pest_estimate']:.2f}           {s['budget']:5.1f}  {s['event']}"
            )
            if viewer:
                viewer.push(s)
                time.sleep(1.0 / args.speed)
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
    ap.add_argument("--speed", type=float, default=3.0, help="decision steps per second in the GUI")
    ap.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    ap.add_argument("--drought", action="store_true", help="evaluate with 40%% less rain")
    ap.add_argument("--year-index", type=int, default=0)
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
