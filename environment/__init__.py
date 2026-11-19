from gymnasium.envs.registration import register

from .custom_env import ACTION_LABELS, ACTION_NAMES, OBS_NAMES, Action, EnvParams, UmuhinziEnv

register(id="Umuhinzi-v0", entry_point="environment.custom_env:UmuhinziEnv")

__all__ = ["UmuhinziEnv", "EnvParams", "Action", "ACTION_NAMES", "ACTION_LABELS", "OBS_NAMES"]
