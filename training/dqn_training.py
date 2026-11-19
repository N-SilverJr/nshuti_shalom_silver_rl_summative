"""Deep Q-Network on UmuhinziEnv with TorchRL.

Usage:
    uv run python -m training.dqn_training --run 1            # one sweep config
    uv run python -m training.dqn_training --sweep            # all 10 configs
    uv run python -m training.dqn_training --run 1 --frames 20000   # quick test
"""

from __future__ import annotations

import argparse

import torch
from tensordict.nn import TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate

from training.common import (
    N_ACTIONS,
    CheckpointKeeper,
    RunLogger,
    build_mlp,
    evaluate,
    finish_run,
    load_sweep,
    make_env,
    promote_best,
    set_seeds,
)


def train_dqn(cfg: dict, verbose: bool = True) -> dict:
    run_name = cfg["name"]
    set_seeds(cfg["seed"])
    env = make_env("train", seed=cfg["seed"])

    q_net = build_mlp(N_ACTIONS, cfg["hidden"], cfg.get("activation", "relu"))
    actor = QValueActor(q_net, in_keys=["observation"], spec=env.action_spec)
    explore = EGreedyModule(
        env.action_spec,
        eps_init=cfg["eps_start"],
        eps_end=cfg["eps_end"],
        annealing_num_steps=int(cfg["eps_decay_frac"] * cfg["total_frames"]),
    )
    policy = TensorDictSequential(actor, explore)

    collector = Collector(
        env,
        policy,
        frames_per_batch=cfg["frames_per_batch"],
        total_frames=cfg["total_frames"],
        init_random_frames=cfg["learning_starts"],
        auto_register_policy_transforms=False,
    )
    rb = TensorDictReplayBuffer(storage=LazyTensorStorage(cfg["buffer_size"]), batch_size=cfg["batch_size"])

    loss_fn = DQNLoss(actor, action_space=env.action_spec, delay_value=True, double_dqn=cfg["double_dqn"])
    loss_fn.make_value_estimator(gamma=cfg["gamma"])
    if cfg["target_update"] == "soft":
        updater = SoftUpdate(loss_fn, tau=cfg["tau"])
    else:
        updater = HardUpdate(loss_fn, value_network_update_interval=cfg["target_update_interval"])
    optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])

    logger = RunLogger("dqn", run_name, cfg)
    frames, episodes = 0, 0
    keeper = CheckpointKeeper(q_net)
    recent_returns: list[float] = []
    losses, qvals = [], []
    next_eval = cfg["eval_every"]

    for data in collector:
        frames += data.numel()
        done = data["next", "done"].squeeze(-1)
        if done.any():
            recent_returns += data["next", "episode_reward"][done].squeeze(-1).tolist()
            episodes += int(done.sum())
        rb.extend(data.reshape(-1))
        explore.step(data.numel())

        if frames >= cfg["learning_starts"]:
            for _ in range(cfg["updates_per_batch"]):
                sample = rb.sample()
                loss_td = loss_fn(sample)
                loss = loss_td["loss"]
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), cfg["max_grad_norm"])
                optim.step()
                updater.step()
                losses.append(loss.item())
            with torch.no_grad():
                qvals.append(q_net(sample["observation"]).max(-1).values.mean().item())
        collector.update_policy_weights_()

        if frames >= next_eval or frames >= cfg["total_frames"]:
            next_eval += cfg["eval_every"]
            ev = evaluate(q_net, n_episodes=cfg["eval_episodes"], split="val")
            row = {
                "frames": frames,
                "episodes": episodes,
                "train_return": sum(recent_returns[-20:]) / max(len(recent_returns[-20:]), 1),
                "loss": sum(losses) / max(len(losses), 1),
                "q_mean": sum(qvals) / max(len(qvals), 1),
                "epsilon": float(explore.eps),
                **{k.replace("eval_", "val_"): v for k, v in ev.items()},
            }
            logger.log(row, verbose)
            losses, qvals = [], []
            keeper.update(ev["eval_return_mean"], frames)
    collector.shutdown()
    return finish_run("dqn", q_net, cfg, logger, keeper, frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, help="1-based index into training/configs/dqn_sweep.json")
    ap.add_argument("--sweep", action="store_true", help="run all configs")
    ap.add_argument("--frames", type=int, help="override total_frames (quick tests)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    configs = load_sweep("dqn")
    chosen = configs if args.sweep else [configs[(args.run or 1) - 1]]
    for cfg in chosen:
        if args.frames:
            cfg = {**cfg, "total_frames": args.frames, "eval_every": max(args.frames // 5, 1000)}
        print(f"\n=== DQN {cfg['name']} ===")
        train_dqn(cfg, verbose=not args.quiet)
    promote_best("dqn")


if __name__ == "__main__":
    main()
