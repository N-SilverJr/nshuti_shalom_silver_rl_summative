"""Umuhinzi RL summative - single entry point.

    uv run main.py                                   # 3D demo of the best trained agent
    uv run main.py play --algo ppo --episodes 2      # (same options as play.py)
    uv run main.py train --algo dqn --run 1          # one config from the sweep file
    uv run main.py sweep --algo ppo                  # all 10 configs of one algorithm
    uv run main.py sweep --algo all                  # all 40 runs
    uv run main.py evaluate                          # best models vs baselines, normal + drought
    uv run main.py plots                             # report figures -> assets/figures/
    uv run main.py baselines                         # rule-based farmer policies only
"""

from __future__ import annotations

import argparse
import sys

ALGOS = ["dqn", "reinforce", "ppo", "a2c"]


def _train(algo: str, runs: list[int] | None, frames: int | None, quiet: bool):
    from training.common import load_sweep, promote_best

    configs = load_sweep(algo)
    idx = runs or list(range(1, len(configs) + 1))
    for i in idx:
        cfg = configs[i - 1]
        if frames:
            cfg = {**cfg, "total_frames": frames, "eval_every": max(frames // 10, 1000)}
        print(f"\n=== {algo.upper()} {cfg['name']} ({i}/{len(configs)}) ===", flush=True)
        if algo == "dqn":
            from training.dqn_training import train_dqn

            train_dqn(cfg, verbose=not quiet)
        else:
            from training.pg_training import train_pg

            train_pg(algo, cfg, verbose=not quiet)
    promote_best(algo)


def main(argv=None):
    from play import add_play_args, play

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    p_play = sub.add_parser("play", help="run an agent in the 3D simulation")
    add_play_args(p_play)

    p_train = sub.add_parser("train", help="train one or more configs")
    p_train.add_argument("--algo", required=True, choices=ALGOS)
    p_train.add_argument("--run", type=int, nargs="+", help="1-based config index(es)")
    p_train.add_argument("--frames", type=int, help="override total_frames")
    p_train.add_argument("--quiet", action="store_true")

    p_sweep = sub.add_parser("sweep", help="run the full 10-config sweep")
    p_sweep.add_argument("--algo", required=True, choices=ALGOS + ["all"])
    p_sweep.add_argument("--frames", type=int, help="override total_frames")
    p_sweep.add_argument("--quiet", action="store_true")

    sub.add_parser("evaluate", help="best models vs baselines on held-out years")
    sub.add_parser("plots", help="generate report figures")
    sub.add_parser("baselines", help="evaluate rule-based policies")

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help")):
        argv = ["play", *argv]  # `uv run main.py [play options]` defaults to play
    args = ap.parse_args(argv)
    if args.cmd == "play":
        play(args)
    elif args.cmd == "train":
        _train(args.algo, args.run, args.frames, args.quiet)
    elif args.cmd == "sweep":
        for a in ALGOS if args.algo == "all" else [args.algo]:
            _train(a, None, args.frames, args.quiet)
    elif args.cmd == "evaluate":
        from analysis.evaluate import run

        run()
    elif args.cmd == "plots":
        from analysis.plots import main as plots

        plots()
    elif args.cmd == "baselines":
        import numpy as np

        from environment.baselines import BASELINES, run_policy

        for name, cls in BASELINES.items():
            for scale in (1.0, 0.6):
                res = run_policy(cls(), n_episodes=50, rain_scale=scale)
                R = np.array([r["return"] for r in res])
                print(f"{name:18s} rain x{scale}: return {R.mean():6.2f} ± {R.std():.2f}  "
                      f"harvest {np.mean([r['harvest_kg'] for r in res]):5.0f} kg  "
                      f"profit {np.mean([r['profit'] for r in res]):6.1f} kRWF")


if __name__ == "__main__":
    sys.exit(main())
