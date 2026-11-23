import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environment.baselines import BASELINES, run_policy
from environment.custom_env import OBS_NAMES, Action, UmuhinziEnv
from environment.weather import WeatherProvider


def test_gymnasium_api():
    check_env(UmuhinziEnv(), skip_render_check=True)


def test_observation_bounds_and_length():
    env = UmuhinziEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (len(OBS_NAMES),)
    rng = np.random.default_rng(0)
    done, steps = False, 0
    while not done:
        obs, r, term, trunc, _ = env.step(int(rng.integers(len(Action))))
        assert env.observation_space.contains(obs)
        assert np.isfinite(r)
        done = term or trunc
        steps += 1
    assert steps <= env.max_steps


def test_weather_years_deterministic_and_disjoint():
    w = WeatherProvider()
    a, b = w.season(5), w.season(5)
    assert np.allclose(a.rain, b.rain) and len(a.rain) == len(b.rain)
    tr, va, ev, tg = (set(UmuhinziEnv(split=s).year_ids) for s in ("train", "val", "eval", "test_gen"))
    assert not (tr & va) and not (tr & ev) and not (va & ev) and not (tr & tg) and not (va & tg)


def test_invalid_actions_are_penalised_noops():
    env = UmuhinziEnv()
    env.reset(seed=1)
    budget = env.budget
    _, r, *_ = env.step(Action.SPRAY_CHEAP)  # nothing planted yet
    assert r < 0 and env.budget == budget and env.totals["invalid"] == 1


def test_cheap_spray_builds_resistance():
    env = UmuhinziEnv()
    env.reset(seed=2)
    env.step(Action.PLANT)
    r0 = env.resistance
    env.step(Action.SPRAY_CHEAP)
    assert env.resistance > r0 and env.cheap_sprays == 1


def test_scouting_reveals_true_pest_level():
    env = UmuhinziEnv()
    env.reset(seed=3)
    env.step(Action.PLANT)
    # force an established crop so the test does not depend on germination luck
    env.planted, env.plant_day, env.health, env.stage = True, env.day - 20, 1.0, 0.2
    env.pest = 0.4
    env.step(Action.SCOUT)
    assert abs(env.pest_scouted - env.pest) < 0.1 and env.steps_since_scout == 0


def test_never_planting_is_a_failure():
    env = UmuhinziEnv()
    env.reset(seed=4)
    total, done = 0.0, False
    while not done:
        _, r, te, tr, _ = env.step(Action.WAIT)
        total += r
        done = te or tr
    assert total == pytest.approx(-env.p.failure_penalty)


def test_sensible_baseline_beats_random():
    rnd = np.mean([r["return"] for r in run_policy(BASELINES["random"](), 20)])
    thr = np.mean([r["return"] for r in run_policy(BASELINES["threshold_farmer"](), 20)])
    assert thr > rnd + 2


def test_reward_breakdown_adds_up():
    """The itemised reward shown in the viewer must equal the real reward."""
    env = UmuhinziEnv(split="eval")
    for ep in range(3):
        env.reset(seed=ep, options={"year": env.year_ids[ep]})
        rng = np.random.default_rng(ep)
        done = False
        while not done:
            _, r, te, tr, info = env.step(int(rng.choice([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])))
            assert sum(info["reward_parts"].values()) == pytest.approx(r, abs=1e-9)
            done = te or tr
