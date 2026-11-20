"""Policy-gradient methods on UmuhinziEnv with TorchRL: REINFORCE, PPO, A2C.

Usage:
    uv run python -m training.pg_training --algo ppo --run 1
    uv run python -m training.pg_training --algo a2c --sweep
    uv run python -m training.pg_training --algo reinforce --run 3 --frames 20000
"""

from __future__ import annotations

import argparse

import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import OneHotCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss, ReinforceLoss
from torchrl.objectives.value import GAE

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


def build_actor_critic(cfg: dict, env):
    act = cfg.get("activation", "tanh")
    policy_net = build_mlp(N_ACTIONS, cfg["hidden"], act)
    value_net = build_mlp(1, cfg["hidden"], act)
    actor = ProbabilisticActor(
        TensorDictModule(policy_net, in_keys=["observation"], out_keys=["logits"]),
        in_keys=["logits"],
        out_keys=["action"],
        spec=env.action_spec,
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )
    critic = ValueOperator(value_net, in_keys=["observation"])
    return policy_net, actor, critic


def _episode_stats(data, recent: list[float]) -> int:
    done = data["next", "done"].squeeze(-1)
    if done.any():
        recent += data["next", "episode_reward"][done].squeeze(-1).tolist()
    return int(done.sum())


def _discounted_returns(reward: torch.Tensor, done: torch.Tensor, gamma: float) -> torch.Tensor:
    """Monte-Carlo returns along the last (time) dim, resetting at episode ends."""
    g = torch.zeros_like(reward)
    running = torch.zeros_like(reward[..., -1, :])
    for t in reversed(range(reward.shape[-2])):
        running = reward[..., t, :] + gamma * running * (~done[..., t, :]).float()
        g[..., t, :] = running
    return g


def train_pg(algo: str, cfg: dict, verbose: bool = True) -> dict:
    run_name = cfg["name"]
    set_seeds(cfg["seed"])
    num_envs = cfg.get("num_envs", 1)
    env = make_env("train", seed=cfg["seed"], num_envs=num_envs)
    policy_net, actor, critic = build_actor_critic(cfg, env)
    ent_coef = cfg["entropy_coef"]

    if algo == "ppo":
        loss_fn = ClipPPOLoss(
            actor,
            critic,
            clip_epsilon=cfg["clip_epsilon"],
            entropy_coeff=ent_coef,
            critic_coeff=cfg["value_coef"],
            normalize_advantage=cfg["normalize_advantage"],
            loss_critic_type="smooth_l1",
        )
    elif algo == "a2c":
        loss_fn = A2CLoss(actor, critic, entropy_coeff=ent_coef, critic_coeff=cfg["value_coef"])
    elif algo == "reinforce":
        loss_fn = ReinforceLoss(actor, critic)
    else:
        raise ValueError(algo)

    if algo == "ppo":
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(cfg["frames_per_batch"]),
            sampler=SamplerWithoutReplacement(),
            batch_size=cfg["minibatch_size"],
        )
    if algo in ("ppo", "a2c"):
        advantage = GAE(gamma=cfg["gamma"], lmbda=cfg["gae_lambda"], value_network=critic, average_gae=False)

    collector = Collector(
        env, actor, frames_per_batch=cfg["frames_per_batch"], total_frames=cfg["total_frames"],
        auto_register_policy_transforms=False,
    )
    optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])
    n_updates_total = cfg["total_frames"] // cfg["frames_per_batch"]
    sched = (
        torch.optim.lr_scheduler.LinearLR(optim, 1.0, 0.1, total_iters=n_updates_total)
        if cfg.get("lr_anneal")
        else None
    )

    logger = RunLogger(algo, run_name, cfg)
    frames, episodes = 0, 0
    keeper = CheckpointKeeper(policy_net)
    recent: list[float] = []
    stats = {"loss_actor": [], "loss_value": [], "entropy": [], "kl_approx": [], "clip_frac": []}
    next_eval = cfg["eval_every"]

    for data in collector:
        frames += data.numel()
        episodes += _episode_stats(data, recent)

        if algo == "ppo":
            for _ in range(cfg["n_epochs"]):
                with torch.no_grad():
                    advantage(data)
                rb.empty()
                rb.extend(data.reshape(-1))
                for _ in range(max(data.numel() // cfg["minibatch_size"], 1)):
                    mb = rb.sample()
                    out = loss_fn(mb)
                    loss = sum(v for k, v in out.items() if k.startswith("loss_"))  # loss_entropy is absent when its coef is 0
                    optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), cfg["max_grad_norm"])
                    optim.step()
                    stats["loss_actor"].append(out["loss_objective"].item())
                    stats["loss_value"].append(out["loss_critic"].item())
                    stats["entropy"].append(out["entropy"].item() if "entropy" in out.keys() else float("nan"))
                    if "kl_approx" in out.keys():
                        stats["kl_approx"].append(out["kl_approx"].item())
                    if "clip_fraction" in out.keys():
                        stats["clip_frac"].append(out["clip_fraction"].item())

        elif algo == "a2c":
            with torch.no_grad():
                advantage(data)
            flat = data.reshape(-1)
            if cfg["normalize_advantage"]:
                adv = flat["advantage"]
                flat["advantage"] = (adv - adv.mean()) / (adv.std() + 1e-8)
            out = loss_fn(flat)
            loss = sum(v for k, v in out.items() if k.startswith("loss_"))  # loss_entropy is absent when its coef is 0
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), cfg["max_grad_norm"])
            optim.step()
            stats["loss_actor"].append(out["loss_objective"].item())
            stats["loss_value"].append(out["loss_critic"].item())
            if "entropy" in out.keys():
                stats["entropy"].append(out["entropy"].item())
            else:  # entropy_coeff=0: A2CLoss skips it, but we still log it for the entropy figure
                with torch.no_grad():
                    stats["entropy"].append(actor.get_dist(flat).entropy().mean().item())

        else:  # REINFORCE: Monte-Carlo returns over complete episodes in the batch
            td = data if data.ndim == 2 else data.unsqueeze(0)
            reward = td["next", "reward"]
            done = td["next", "done"]
            g = _discounted_returns(reward, done, cfg["gamma"])
            # Keep only steps whose episode finished inside this batch, so every
            # return is a true Monte-Carlo return (no bootstrapping).
            finished_after = torch.flip(torch.cummax(torch.flip(done.float(), [-2]), -2).values, [-2]).bool()
            mask = finished_after.squeeze(-1)
            flat = td[mask]
            if flat.numel() > 0:
                returns = g[mask]
                with torch.no_grad():
                    v = critic(flat.select("observation").clone())["state_value"]
                adv = returns - v if cfg["baseline"] else returns.clone()
                if cfg["normalize_returns"]:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                flat.set("advantage", adv)
                flat.set("value_target", returns)
                out = loss_fn(flat)
                dist = actor.get_dist(flat)
                entropy = dist.entropy().mean()
                loss = out["loss_actor"] + cfg["value_coef"] * out["loss_value"] - ent_coef * entropy
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), cfg["max_grad_norm"])
                optim.step()
                stats["loss_actor"].append(out["loss_actor"].item())
                stats["loss_value"].append(out["loss_value"].item())
                stats["entropy"].append(entropy.item())

        if sched:
            sched.step()
        collector.update_policy_weights_()

        if frames >= next_eval or frames >= cfg["total_frames"]:
            next_eval += cfg["eval_every"]
            ev = evaluate(policy_net, n_episodes=cfg["eval_episodes"], split="val")
            ev_s = evaluate(policy_net, n_episodes=cfg["eval_episodes"], split="val", mode="sample")
            ev["eval_sample_return_mean"] = ev_s["eval_return_mean"]  # logged as val_sample_return_mean
            row = {
                "frames": frames,
                "episodes": episodes,
                "train_return": sum(recent[-20:]) / max(len(recent[-20:]), 1),
                **{k: sum(v) / len(v) for k, v in stats.items() if v},
                "lr": optim.param_groups[0]["lr"],
                **{k.replace("eval_", "val_"): v for k, v in ev.items()},
            }
            logger.log(row, verbose)
            stats = {k: [] for k in stats}
            keeper.update(ev["eval_return_mean"], frames)
    collector.shutdown()
    return finish_run(algo, policy_net, cfg, logger, keeper, frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["reinforce", "ppo", "a2c"])
    ap.add_argument("--run", type=int, help="1-based index into training/configs/<algo>_sweep.json")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--frames", type=int, help="override total_frames (quick tests)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    configs = load_sweep(args.algo)
    chosen = configs if args.sweep else [configs[(args.run or 1) - 1]]
    for cfg in chosen:
        if args.frames:
            cfg = {**cfg, "total_frames": args.frames, "eval_every": max(args.frames // 5, 1000)}
        print(f"\n=== {args.algo.upper()} {cfg['name']} ===")
        with set_exploration_type(ExplorationType.RANDOM):
            train_pg(args.algo, cfg, verbose=not args.quiet)
    promote_best(args.algo)


if __name__ == "__main__":
    main()
