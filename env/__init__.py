"""visual-reasoning-rlvr environment package.

An agentic RL environment where a VLM investigates a face image under a
resolution/action budget — committing a falsifiable hypothesis before each
high-resolution reveal, reconciling it afterward, and finally committing a
verdict. The reward (``env.reward``) scores only mechanically verifiable
outcomes; reasoning is forced by the trajectory structure, never judged for
eloquence.
"""

from env.reward import RewardConfig, compute_episode_reward
from env.trajectory import Trajectory, TurnEntry, parse_turn

__all__ = [
    "InvestigationEnv",
    "RewardConfig",
    "compute_episode_reward",
    "Trajectory",
    "TurnEntry",
    "parse_turn",
]


def __getattr__(name):
    # Lazily import the environment so the lightweight reward/trajectory modules
    # (and their tests / CI) don't pull in PIL + gymnasium unless actually needed.
    if name == "InvestigationEnv":
        from env.environment import InvestigationEnv

        return InvestigationEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
