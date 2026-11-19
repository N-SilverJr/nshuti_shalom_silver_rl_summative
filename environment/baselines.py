"""Non-learning reference policies used to sanity-check the environment and
as comparison points in the report ("what farmers do today")."""

from __future__ import annotations

import numpy as np

from .custom_env import Action, UmuhinziEnv


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, env: UmuhinziEnv, obs: np.ndarray) -> int:
        return int(self.rng.integers(len(Action)))


class CalendarFarmer:
    """Plants on a fixed date, sprays the cheap pesticide every 2 weeks once
    the crop is up, weeds twice, harvests at maturity. Mirrors common practice."""

    name = "calendar_farmer"

    def __call__(self, env: UmuhinziEnv, obs: np.ndarray) -> int:
        if not env.planted:
            return Action.PLANT if env.day >= 21 else Action.WAIT
        dsp = env.day - env.plant_day
        if env.stage >= 1.0:
            return Action.HARVEST
        if dsp in range(18, 21) or dsp in range(42, 45):
            return Action.WEED_FERTILIZE
        if 15 <= dsp <= 80 and env.days_since_spray >= 14:
            return Action.SPRAY_CHEAP
        return Action.WAIT


class ThresholdFarmer:
    """An extension-officer style rule set that only uses observable signals:
    plant after ~50 mm cumulative rain with moist soil, scout weekly, spray
    when scouted infestation passes 20% (rotating products), irrigate from the
    tank under stress, weed twice, harvest at maturity."""

    name = "threshold_farmer"

    def __call__(self, env: UmuhinziEnv, obs: np.ndarray) -> int:
        if not env.planted:
            return Action.PLANT if (env.cum_rain >= 50 and env.soil >= 0.4) else Action.WAIT
        if env.stage >= 1.0:
            return Action.HARVEST
        dsp = env.day - env.plant_day
        if env.steps_since_scout >= 2 and env.stage < 0.95:
            return Action.SCOUT
        if env.pest_scouted > 0.2 and env.days_since_spray > 6 and env.steps_since_scout == 0:
            return Action.SPRAY_BIO if env.cheap_sprays >= 2 else Action.SPRAY_CHEAP
        if env.soil < 0.22 and env.budget > 10 and env.tank >= env.p.tank_per_irrigation:
            return Action.IRRIGATE
        if dsp in range(18, 21) or dsp in range(42, 45):
            return Action.WEED_FERTILIZE
        return Action.WAIT


BASELINES = {c.name: c for c in (RandomPolicy, CalendarFarmer, ThresholdFarmer)}


def run_policy(policy, n_episodes: int = 50, split: str = "eval", seed: int = 0, rain_scale: float = 1.0):
    """Roll out a callable ``policy(env, obs) -> action`` and collect stats."""
    env = UmuhinziEnv(split=split, rain_scale=rain_scale)
    years = env.year_ids
    out = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i, options={"year": years[i % len(years)]})
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(policy(env, obs))
            done = term or trunc
        out.append(
            {
                "return": env.ep_return,
                "harvest_kg": env.harvest_kg,
                "steps": env.t,
                "cheap_sprays": env.cheap_sprays,
                "toxicity": env.totals["toxicity"],
                "costs": env.totals["costs"],
                "profit": env.totals["revenue"] - env.totals["costs"],
            }
        )
    return out
