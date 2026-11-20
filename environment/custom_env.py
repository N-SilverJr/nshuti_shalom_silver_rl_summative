"""UmuhinziEnv: one Season-A maize season on a Rwandan smallholder plot.

The agent plays the role of an early-detection / advisory system. Every
decision step (3 days, a realistic SMS cadence) it chooses ONE advisory to
send the farmer. Each advisory maps to a concrete real-world action.

Persistent, action-dependent state carried across steps:
    soil moisture, crop stage, crop health, fall-armyworm (FAW) infestation,
    pest resistance to the cheap pesticide, weed pressure, soil fertility,
    budget, whether the crop is planted.

Delayed trade-offs the agent must learn:
    * Planting before reliable rains risks germination failure (seed lost);
      planting late leaves too little season for the crop to mature.
    * The cheap broad-spectrum spray works now but builds resistance that
      makes later outbreaks harder to control; the targeted/bio spray costs
      more but does not.
    * The farmer's daily pest impression is noisy; scouting costs money and a
      turn but returns an accurate infestation reading.
    * Money spent early cannot be spent later; most reward arrives at harvest.

Units: money in thousands of Rwandan francs (kRWF) on a 0.1 ha plot.
Reward = net cash flow / 10 kRWF, minus pesticide-toxicity and shaping terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .weather import SEASON_DAYS, Season, WeatherProvider


class Action(IntEnum):
    WAIT = 0  # monitor only, send no advisory
    PLANT = 1  # "Rains are established: plant maize now"
    IRRIGATE = 2  # "Water the plot from your rainwater tank today"
    SPRAY_CHEAP = 3  # "Spray broad-spectrum pesticide (e.g. cypermethrin)"
    SPRAY_BIO = 4  # "Apply targeted/bio control (e.g. Bt / emamectin)"
    SCOUT = 5  # "Inspect 20 plants in a W pattern and report FAW damage"
    WEED_FERTILIZE = 6  # "Weed and top-dress with manure/urea"
    HARVEST = 7  # "Harvest now"


ACTION_NAMES = [a.name for a in Action]
ACTION_LABELS = {
    Action.WAIT: "Monitor (no advisory)",
    Action.PLANT: "Plant maize",
    Action.IRRIGATE: "Irrigate from tank",
    Action.SPRAY_CHEAP: "Spray cheap pesticide",
    Action.SPRAY_BIO: "Apply bio/targeted control",
    Action.SCOUT: "Scout field for armyworm",
    Action.WEED_FERTILIZE: "Weed + top-dress",
    Action.HARVEST: "Harvest",
}

OBS_NAMES = [
    "day_of_season",
    "rain_last_step",
    "forecast_rain_7d",
    "cum_rain_since_start",
    "soil_moisture",
    "water_tank",
    "temperature",
    "planted",
    "crop_stage",
    "crop_health",
    "pest_estimate",
    "pest_scouted_last",
    "steps_since_scout",
    "weed_pressure",
    "soil_fertility",
    "budget",
    "cheap_sprays_used",
    "days_since_spray",
]


@dataclass
class EnvParams:
    step_days: int = 3
    budget0: float = 45.0  # kRWF available for inputs this season
    costs: dict = field(
        default_factory=lambda: {
            Action.WAIT: 0.0,
            Action.PLANT: 12.0,  # hybrid seed + labour for 0.1 ha
            Action.IRRIGATE: 1.0,  # labour; water comes from the tank
            Action.SPRAY_CHEAP: 2.5,
            Action.SPRAY_BIO: 6.0,
            Action.SCOUT: 0.3,
            Action.WEED_FERTILIZE: 4.0,
            Action.HARVEST: 2.0,
        }
    )
    toxicity: dict = field(
        default_factory=lambda: {Action.SPRAY_CHEAP: 0.6, Action.SPRAY_BIO: 0.1}
    )
    max_yield_kg: float = 450.0  # 4.5 t/ha potential on 0.1 ha
    price_kg: float = 0.35  # kRWF per kg of grain
    days_to_maturity: int = 120
    soil_capacity_mm: float = 70.0
    irrigation_mm: float = 12.0
    tank_per_irrigation: float = 0.34  # a full tank holds ~3 irrigations
    tank_fill_per_mm: float = 0.015  # roof catchment: ~65 mm of rain fills it
    reward_scale: float = 10.0  # kRWF per reward unit
    shaping_alpha: float = 0.5  # weight of potential-based crop-value shaping
    shaping_gamma: float = 0.99
    invalid_penalty: float = 0.1
    failure_penalty: float = 3.0
    pest_pressure: float = 1.0  # multiplies moth immigration


class UmuhinziEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(
        self,
        split: str = "train",
        rain_scale: float = 1.0,
        params: EnvParams | None = None,
        render_mode: str | None = None,
        weather: WeatherProvider | None = None,
        force_synthetic: bool = False,
    ):
        super().__init__()
        assert split in ("train", "val", "eval", "test_gen")
        self.p = params or EnvParams()
        self.split = split
        self.rain_scale = rain_scale
        self.render_mode = render_mode
        self.weather = weather or WeatherProvider(force_synthetic=force_synthetic)
        self.year_ids = self.weather.year_ids(split)
        self.max_steps = SEASON_DAYS // self.p.step_days

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(0.0, 1.0, shape=(len(OBS_NAMES),), dtype=np.float32)
        self._viewer = None
        self.season: Season | None = None

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        year = options.get("year")
        if year is None:
            year = int(self.np_random.choice(self.year_ids))
        self.season = self.weather.season(int(year), options.get("rain_scale", self.rain_scale))

        self.day = 0
        self.t = 0
        self.soil = float(self.np_random.uniform(0.15, 0.3))  # dry end of August
        self.tank = float(self.np_random.uniform(0.0, 0.15))  # nearly empty after the dry season
        self.cum_rain = 0.0
        self.rain_last = 0.0
        self.planted = False
        self.plant_day = -1
        self.stage = 0.0
        self.health = 0.0
        self.germ_soil: list[float] = []
        self.pest = 0.0
        self.resistance = float(self.np_random.uniform(0.0, 0.15))
        self.weeds = 0.0
        self.fertility = float(self.np_random.uniform(0.6, 0.8))
        self.budget = self.p.budget0
        self.cheap_sprays = 0
        self.days_since_spray = 30
        self.pest_scouted = 0.0
        self.steps_since_scout = 10
        self.pest_estimate = 0.0
        self.harvest_kg = 0.0
        self.last_action = None
        self.last_event = "Season start: waiting for the rains"
        self.ep_return = 0.0
        self.last_reward = 0.0
        self.totals = {"revenue": 0.0, "costs": 0.0, "toxicity": 0.0, "shaping": 0.0, "invalid": 0}
        self._update_pest_estimate(scouted=False)
        return self._obs(), self._info()

    # ------------------------------------------------------------------- step
    def valid_actions(self) -> np.ndarray:
        mask = np.zeros(len(Action), dtype=bool)
        mask[Action.WAIT] = True
        cost = self.p.costs
        if not self.planted:
            mask[Action.PLANT] = self.budget >= cost[Action.PLANT]
        else:
            for a in (Action.IRRIGATE, Action.SPRAY_CHEAP, Action.SPRAY_BIO, Action.SCOUT, Action.WEED_FERTILIZE):
                mask[a] = self.budget >= cost[a]
            mask[Action.IRRIGATE] &= self.tank >= self.p.tank_per_irrigation
            mask[Action.HARVEST] = self.stage >= 0.75
        return mask

    def step(self, action):
        action = Action(int(action))
        p = self.p
        reward = 0.0
        terminated = False
        events: list[str] = []
        phi_before = self._potential()
        scouted = False

        if not self.valid_actions()[action]:
            reward -= p.invalid_penalty
            self.totals["invalid"] += 1
            events.append(f"Invalid advisory ({ACTION_LABELS[action]}) ignored")
            action_effective = Action.WAIT
        else:
            action_effective = action
            cost = p.costs[action]
            self.budget -= cost
            reward -= cost / p.reward_scale
            self.totals["costs"] += cost
            tox = p.toxicity.get(action, 0.0)
            reward -= tox
            self.totals["toxicity"] += tox

        a = action_effective
        if a == Action.PLANT:
            self.planted = True
            self.plant_day = self.day
            self.stage = 0.0
            self.health = 1.0
            self.germ_soil = []
            self.pest = 0.0
            self.weeds = 0.05
            events.append("Maize planted")
        elif a == Action.IRRIGATE:
            self.soil = min(1.0, self.soil + p.irrigation_mm / p.soil_capacity_mm)
            self.tank -= p.tank_per_irrigation
            events.append("Plot irrigated from tank")
        elif a in (Action.SPRAY_CHEAP, Action.SPRAY_BIO):
            next_rain = float(self.season.rain[self.day : self.day + 2].sum())
            washoff = 0.5 if next_rain > 15 else 1.0
            if a == Action.SPRAY_CHEAP:
                eff = 0.85 * (1.0 - self.resistance) * washoff
                self.resistance += 0.12 * (1.0 - self.resistance)
                self.cheap_sprays += 1
            else:
                eff = 0.70 * washoff
            self.pest *= 1.0 - eff
            self.days_since_spray = 0
            events.append(f"Sprayed ({'cheap' if a == Action.SPRAY_CHEAP else 'bio'}), kill rate {eff:.0%}")
        elif a == Action.SCOUT:
            scouted = True
            events.append("Field scouted")
        elif a == Action.WEED_FERTILIZE:
            self.weeds *= 0.15
            self.fertility = min(1.0, self.fertility + 0.12)
            events.append("Weeded and top-dressed")
        elif a == Action.HARVEST:
            reward += self._harvest(events)
            terminated = True

        # ---- simulate the days until the next decision
        if not terminated:
            for _ in range(p.step_days):
                ev = self._simulate_day()
                if ev:
                    events.append(ev)
                if self.planted and self.health <= 0.05:
                    events.append("Crop lost (health collapsed)")
                    reward -= p.failure_penalty
                    terminated = True
                    break

        # Potential-based shaping F = gamma*Phi(s') - Phi(s) (Ng et al., 1999):
        # dense feedback on crop progress that leaves the optimal policy unchanged.
        phi_after = 0.0 if terminated else self._potential()
        shaping = p.shaping_gamma * phi_after - phi_before
        reward += shaping
        self.totals["shaping"] += shaping

        self.t += 1
        truncated = False
        if not terminated and self.t >= self.max_steps:
            if self.planted:
                events.append("Season over: forced harvest")
                reward += self._harvest(events)
                reward -= p.shaping_gamma * phi_after  # potential of harvested field is 0
                self.totals["shaping"] -= p.shaping_gamma * phi_after
            else:
                events.append("Season over: nothing was planted")
                reward -= p.failure_penalty
            truncated = True
        if not terminated and not truncated and not self.planted and self.budget < p.costs[Action.PLANT]:
            events.append("Budget exhausted before a crop could be established")
            reward -= p.failure_penalty
            terminated = True

        self._update_pest_estimate(scouted)
        self.last_action = int(action)
        self.last_event = "; ".join(events) if events else "Monitoring"
        self.ep_return += reward
        self.last_reward = float(reward)
        info = self._info()
        info["reward"] = reward
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, info

    # -------------------------------------------------------------- dynamics
    def _simulate_day(self) -> str | None:
        p = self.p
        rain = float(self.season.rain[self.day])
        temp = float(self.season.temp[self.day])
        self.cum_rain += rain
        self.tank = min(1.0, self.tank + rain * p.tank_fill_per_mm)
        event = None

        # Soil water bucket: rain in, crop/evaporation out, excess drains.
        kc = 0.35 if not self.planted else 0.35 + 0.8 * min(1.0, self.stage / 0.5)
        et = 3.6 * kc
        self.soil = self.soil + (rain - et) / p.soil_capacity_mm
        if self.soil > 0.95:
            self.soil = 0.95 - 0.3 * (self.soil - 0.95)
        self.soil = float(np.clip(self.soil, 0.0, 1.0))

        self.days_since_spray = min(self.days_since_spray + 1, 60)
        self.resistance = max(0.0, self.resistance - 0.002)  # slow reversion

        if self.planted:
            dsp = self.day - self.plant_day
            # Germination check over the first 6 days.
            if dsp < 6:
                self.germ_soil.append(self.soil)
                if dsp == 5:
                    m = float(np.mean(self.germ_soil))
                    p_fail = float(np.clip((0.38 - m) / 0.2, 0.0, 1.0))
                    if self.np_random.random() < p_fail:
                        self.planted = False
                        self.health = 0.0
                        self.stage = 0.0
                        self.pest = 0.0
                        self.weeds = 0.0
                        event = "Germination failed (dry soil): seed lost"
                        self.day += 1
                        return event

            temp_f = float(np.clip((temp - 12.0) / 10.0, 0.3, 1.3))
            self.stage += (0.75 + 0.25 * temp_f) / p.days_to_maturity

            # Water stress, most damaging around flowering (stage 0.45-0.7).
            stress = float(np.clip((0.3 - self.soil) / 0.3, 0.0, 1.0))
            flowering = 1.0 + 1.5 * float(0.45 <= self.stage <= 0.7)
            waterlog = 0.004 if self.soil > 0.92 else 0.0
            dmg_water = 0.007 * stress * flowering + waterlog

            # Fall armyworm: moth immigration + logistic larval growth.
            if self.stage < 1.0:
                if self.np_random.random() < 0.06 * p.pest_pressure:
                    self.pest += float(self.np_random.uniform(0.01, 0.04))
                r = 0.055 * temp_f
                self.pest += r * self.pest * (1.0 - self.pest)
                if rain > 25:  # heavy rain drowns young larvae
                    self.pest *= 0.85
            else:
                self.pest *= 0.95
            self.pest = float(np.clip(self.pest, 0.0, 1.0))
            if self.stage < 0.1:
                suscept = 0.3
            elif self.stage < 0.55:
                suscept = 1.0  # whorl stage: FAW does most damage
            elif self.stage < 1.0:
                suscept = 0.5  # ear damage
            else:
                suscept = 0.1
            dmg_pest = 0.025 * max(0.0, self.pest - 0.08) * suscept  # below ~8% infestation damage is negligible

            # Weeds compete for water and nutrients in the first ~10 weeks.
            if self.stage < 0.6:
                self.weeds += 0.05 * (0.5 + self.soil) * self.weeds * (1.0 - self.weeds) + 0.002
            self.weeds = float(np.clip(self.weeds, 0.0, 1.0))
            dmg_weed = 0.006 * self.weeds * float(self.stage < 0.6)

            # Grain left in the field after maturity deteriorates.
            dmg_late = 0.008 if self.stage > 1.05 else 0.0

            self.health -= dmg_water + dmg_pest + dmg_weed + dmg_late
            if stress < 0.05 and self.pest < 0.1 and self.stage < 1.0:
                self.health += 0.0015  # mild recovery when unstressed
            self.health = float(np.clip(self.health, 0.0, 1.0))

            if self.pest > 0.5 and event is None:
                event = "Armyworm outbreak!"
            if stress > 0.6 and event is None:
                event = "Crop under drought stress"

        self.rain_last = rain  # last day's rain is shown separately in render
        self.day += 1
        return event

    def _potential(self) -> float:
        """Phi(s): fraction of the expected harvest value already 'in the field'."""
        if not self.planted or self.harvest_kg > 0:
            return 0.0
        p = self.p
        v_max = p.max_yield_kg * p.price_kg / p.reward_scale
        return p.shaping_alpha * v_max * self.health * min(self.stage, 1.0) * (0.6 + 0.4 * self.fertility)

    def _harvest(self, events: list[str]) -> float:
        p = self.p
        maturity = float(np.clip(self.stage, 0.0, 1.0)) ** 2
        fert = 0.6 + 0.4 * self.fertility
        self.harvest_kg = p.max_yield_kg * self.health * maturity * fert
        revenue = self.harvest_kg * p.price_kg
        self.totals["revenue"] += revenue
        events.append(f"Harvested {self.harvest_kg:.0f} kg (~{revenue:.0f} kRWF)")
        return revenue / p.reward_scale

    def _update_pest_estimate(self, scouted: bool):
        if not self.planted:
            self.pest_estimate = 0.0
            return
        if scouted:
            self.pest_scouted = float(np.clip(self.pest + self.np_random.normal(0, 0.02), 0, 1))
            self.steps_since_scout = 0
            self.pest_estimate = self.pest_scouted
        else:
            self.steps_since_scout = min(self.steps_since_scout + 1, 10)
            # Farmers notice armyworm late and inconsistently: biased low + noisy.
            self.pest_estimate = float(np.clip(0.6 * self.pest + self.np_random.normal(0, 0.12), 0, 1))

    # ------------------------------------------------------------ observation
    def _forecast(self) -> float:
        true7 = float(self.season.rain[self.day : self.day + 7].sum())
        return true7 * float(np.exp(self.np_random.normal(0, 0.35)))

    def _obs(self) -> np.ndarray:
        last_step_rain = float(self.season.rain[max(0, self.day - self.p.step_days) : self.day].sum())
        o = np.array(
            [
                self.day / SEASON_DAYS,
                min(last_step_rain / 60.0, 1.0),
                min(self._forecast() / 100.0, 1.0),
                min(self.cum_rain / 400.0, 1.0),
                self.soil,
                self.tank,
                np.clip((float(self.season.temp[min(self.day, len(self.season.temp) - 1)]) - 10.0) / 25.0, 0, 1),
                float(self.planted),
                min(self.stage / 1.2, 1.0),
                self.health,
                self.pest_estimate,
                self.pest_scouted,
                self.steps_since_scout / 10.0,
                float(np.clip(self.weeds + self.np_random.normal(0, 0.03), 0, 1)) if self.planted else 0.0,
                self.fertility,
                float(np.clip(self.budget / self.p.budget0, 0, 1)),
                min(self.cheap_sprays / 6.0, 1.0),
                min(self.days_since_spray / 30.0, 1.0),
            ],
            dtype=np.float32,
        )
        return np.clip(o, 0.0, 1.0)

    def _info(self) -> dict:
        return {
            "day": self.day,
            "year": self.season.year if self.season else None,
            "planted": self.planted,
            "stage": self.stage,
            "health": self.health,
            "pest_true": self.pest,
            "resistance": self.resistance,
            "budget": self.budget,
            "harvest_kg": self.harvest_kg,
            "action_mask": self.valid_actions(),
            "totals": dict(self.totals),
        }

    # ---------------------------------------------------------------- render
    def state_dict(self) -> dict:
        """JSON-serialisable snapshot used by the 3D front-end and the API."""
        date = self.season.start_date.toordinal() + self.day
        import datetime as _dt

        return {
            "t": self.t,
            "day": self.day,
            "date": _dt.date.fromordinal(date).strftime("%d %b"),
            "year": self.season.year,
            "weather_source": self.season.source,
            "rain_today": float(self.season.rain[max(self.day - 1, 0)]),
            "forecast_7d": float(self.season.rain[self.day : self.day + 7].sum()),
            "cum_rain": self.cum_rain,
            "temperature": float(self.season.temp[min(self.day, len(self.season.temp) - 1)]),
            "soil_moisture": self.soil,
            "water_tank": self.tank,
            "planted": self.planted,
            "stage": self.stage,
            "health": self.health,
            "pest_true": self.pest,
            "pest_estimate": self.pest_estimate,
            "pest_scouted": self.pest_scouted,
            "resistance": self.resistance,
            "weeds": self.weeds,
            "fertility": self.fertility,
            "budget": self.budget,
            "budget0": self.p.budget0,
            "cheap_sprays": self.cheap_sprays,
            "harvest_kg": self.harvest_kg,
            "last_action": self.last_action,
            "last_action_label": ACTION_LABELS[Action(self.last_action)] if self.last_action is not None else "-",
            "event": self.last_event,
            "episode_return": self.ep_return,
            "step_reward": self.last_reward,
            "max_steps": self.max_steps,
        }

    def render(self):
        if self.render_mode == "ansi":
            s = self.state_dict()
            return (
                f"[{s['date']}] t={s['t']:2d} act={s['last_action_label']:<28} "
                f"soil={s['soil_moisture']:.2f} stage={s['stage']:.2f} health={s['health']:.2f} "
                f"pest={s['pest_true']:.2f} budget={s['budget']:.1f} | {s['event']}"
            )
        if self.render_mode == "human":
            from .rendering import get_viewer

            if self._viewer is None:
                self._viewer = get_viewer()
            self._viewer.push(self.state_dict())
        return None

    def close(self):
        self._viewer = None
