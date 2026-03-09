from SDK.actions import ActionBundle, ActionCatalog
from SDK.engine import GameState
from SDK.features import FeatureExtractor

__all__ = [
    "ActionBundle",
    "ActionCatalog",
    "AntWarParallelEnv",
    "FeatureExtractor",
    "GameState",
    "env",
]


def __getattr__(name: str):
    if name not in {"AntWarParallelEnv", "env"}:
        raise AttributeError(f"module 'SDK' has no attribute {name!r}")
    from SDK.pettingzoo_env import AntWarParallelEnv, env

    globals().update(
        {
            "AntWarParallelEnv": AntWarParallelEnv,
            "env": env,
        }
    )
    return globals()[name]
