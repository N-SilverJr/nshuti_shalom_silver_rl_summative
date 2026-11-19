"""Shared helpers for all four TorchRL trainers: env factory, evaluation,
logging, checkpointing and sweep bookkeeping."""

from __future__ import annotations

import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchrl.envs import Compose, GymWrapper, RewardSum, SerialEnv, StepCounter, TransformedEnv
from torchrl.modules import MLP

from environment.custom_env import OBS_NAMES, Action, UmuhinziEnv

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "models"
OBS_DIM = len(OBS_NAMES)
N_ACTIONS = len(Action)
ALGO_MODEL_SUBDIR = {"dqn": "dqn", "reinforce": "pg", "ppo": "pg", "a2c": "pg"}


def make_env(split: str = "train", seed: int = 0, num_envs: int = 1) -> TransformedEnv:
    """TorchRL view of UmuhinziEnv. Every algorithm trains on exactly this."""

    def _single():
        return GymWrapper(UmuhinziEnv(split=split), device="cpu")

    base = _single() if num_envs == 1 else SerialEnv(num_envs, _single)
    env = TransformedEnv(base, Compose(StepCounter(), RewardSum()))
    env.set_seed(seed)
    return env


def build_mlp(out_features: int, hidden: list[int], activation: str = "tanh") -> nn.Module:
    act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
    return MLP(in_features=OBS_DIM, out_features=out_features, num_cells=list(hidden), activation_class=act)


def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(
    net: nn.Module,
    n_episodes: int = 30,
    split: str = "eval",
    rain_scale: float = 1.0,
    seed: int = 12345,
    mode: str = "greedy",
):
    """Evaluate on held-out weather years. ``net`` maps obs -> logits or
    Q-values. ``mode='greedy'`` takes the argmax (deployment behaviour, used for
    every algorithm); ``mode='sample'`` samples from softmax(logits), i.e. the
    stochastic policy a policy-gradient method actually optimises."""
    gen = torch.Generator().manual_seed(seed)
    env = UmuhinziEnv(split=split, rain_scale=rain_scale)
    years = env.year_ids
    returns, harvests, tox, sprays, profits, fails, actions = [], [], [], [], [], 0, np.zeros(N_ACTIONS)
    net.eval()
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i, options={"year": years[i % len(years)]})
        done = False
        while not done:
            out = net(torch.as_tensor(obs).unsqueeze(0))
            if mode == "greedy":
                a = int(out.argmax(-1).item())
            else:
                a = int(torch.multinomial(torch.softmax(out, -1), 1, generator=gen).item())
            actions[a] += 1
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
        returns.append(env.ep_return)
        harvests.append(env.harvest_kg)
        tox.append(env.totals["toxicity"])
        sprays.append(env.cheap_sprays)
        profits.append(env.totals["revenue"] - env.totals["costs"])
        fails += env.harvest_kg < 1
    net.train()
    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_harvest_kg": float(np.mean(harvests)),
        "eval_profit_krwf": float(np.mean(profits)),
        "eval_toxicity": float(np.mean(tox)),
        "eval_cheap_sprays": float(np.mean(sprays)),
        "eval_failure_rate": fails / n_episodes,
        "eval_action_dist": (actions / max(actions.sum(), 1)).round(3).tolist(),
    }


# ------------------------------------------------------------------- logging
class RunLogger:
    def __init__(self, algo: str, run_name: str, config: dict):
        self.dir = LOG_DIR / algo / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        self.t0 = time.time()
        json.dump(config, open(self.dir / "config.json", "w"), indent=2)

    def log(self, row: dict, verbose: bool = True):
        row = {"wall_s": round(time.time() - self.t0, 1), **row}
        self.rows.append(row)
        keys = []
        for r in self.rows:
            keys += [k for k in r if k not in keys]
        with open(self.dir / "train.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)
        if verbose:
            show = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "eval_action_dist"}
            print("  ".join(f"{k}={v}" for k, v in show.items()), flush=True)

    def finish(self, summary: dict):
        json.dump(summary, open(self.dir / "summary.json", "w"), indent=2)


def save_model(net: nn.Module, algo: str, run_name: str, config: dict, metrics: dict) -> Path:
    d = MODEL_DIR / ALGO_MODEL_SUBDIR[algo]
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{algo}_{run_name}.pt"
    torch.save(
        {
            "algo": algo,
            "run_name": run_name,
            "state_dict": net.state_dict(),
            "hidden": config["hidden"],
            "activation": config.get("activation", "tanh"),
            "config": config,
            "metrics": metrics,
            "obs_names": OBS_NAMES,
        },
        path,
    )
    return path


def load_policy_net(path: Path) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = build_mlp(N_ACTIONS, ckpt["hidden"], ckpt["activation"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt


def update_sweep_summary(algo: str, run_name: str, config: dict, summary: dict):
    path = LOG_DIR / algo / "sweep_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(path))) if path.exists() else []
    rows = [r for r in rows if r["run"] != run_name]
    flat = {"run": run_name}
    flat.update({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in config.items()})
    flat.update({k: v for k, v in summary.items() if not isinstance(v, (list, dict))})
    rows.append(flat)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_sweep(algo: str) -> list[dict]:
    cfg = json.load(open(ROOT / "training" / "configs" / f"{algo}_sweep.json"))
    base = cfg["base"]
    return [{**base, **run} for run in cfg["runs"]]


class CheckpointKeeper:
    """Keeps the weights with the best *validation* return seen so far."""

    def __init__(self, net: nn.Module):
        self.net, self.best, self.state, self.frames = net, -1e9, None, 0

    def update(self, val_return: float, frames: int):
        if val_return > self.best:
            self.best, self.frames = val_return, frames
            self.state = copy.deepcopy(self.net.state_dict())


def finish_run(algo: str, net: nn.Module, cfg: dict, logger: "RunLogger", keeper: CheckpointKeeper, frames: int) -> dict:
    """Test the final and the best-validation weights on unseen years, save the
    best-validation checkpoint (the deployable model) and record the summary."""
    n = cfg["final_eval_episodes"]
    last = evaluate(net, n_episodes=n, split="eval")
    if keeper.state is not None:
        net.load_state_dict(keeper.state)
    test = evaluate(net, n_episodes=n, split="eval")
    drought = evaluate(net, n_episodes=n, split="eval", rain_scale=0.6)
    summary = {
        "best_val_return": keeper.best,
        "best_val_frames": keeper.frames,
        "test_return": test["eval_return_mean"],
        "test_std": test["eval_return_std"],
        "test_harvest_kg": test["eval_harvest_kg"],
        "test_profit_krwf": test["eval_profit_krwf"],
        "test_failure_rate": test["eval_failure_rate"],
        "test_toxicity": test["eval_toxicity"],
        "test_cheap_sprays": test["eval_cheap_sprays"],
        "drought_test_return": drought["eval_return_mean"],
        "final_weights_test_return": last["eval_return_mean"],
        "frames": frames,
        "wall_s": logger.rows[-1]["wall_s"] if logger.rows else 0,
    }
    if algo != "dqn":
        summary["test_sample_return"] = evaluate(net, n_episodes=n, split="eval", mode="sample")["eval_return_mean"]
    logger.finish(summary)
    save_model(net, algo, cfg["name"], cfg, summary)
    update_sweep_summary(algo, cfg["name"], cfg, summary)
    print(
        f"[{algo}:{cfg['name']}] best val {keeper.best:.2f} @ {keeper.frames} frames | test {summary['test_return']:.2f} "
        f"({summary['test_harvest_kg']:.0f} kg, {summary['test_profit_krwf']:.0f} kRWF) | drought {summary['drought_test_return']:.2f} "
        f"| final weights {summary['final_weights_test_return']:.2f}",
        flush=True,
    )
    return summary


def promote_best(algo: str) -> Path | None:
    """Copy the run with the best *validation* return to best_<algo>.pt
    (selection never looks at the test years)."""
    path = LOG_DIR / algo / "sweep_summary.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    best = max(rows, key=lambda r: float(r["best_val_return"]))
    src = MODEL_DIR / ALGO_MODEL_SUBDIR[algo] / f"{algo}_{best['run']}.pt"
    dst = MODEL_DIR / ALGO_MODEL_SUBDIR[algo] / f"best_{algo}.pt"
    if src.exists():
        dst.write_bytes(src.read_bytes())
        print(f"[{algo}] best run = {best['run']} (val {float(best['best_val_return']):.2f}, "
              f"test {float(best['test_return']):.2f}) -> {dst}")
        return dst
    return None
