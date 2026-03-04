try:
    from .pettingzoo_env import GeneralsAECEnv, env, parallel_env
except Exception as exc:  # pragma: no cover - optional runtime dependency
    _IMPORT_ERROR = str(exc)
    GeneralsAECEnv = None  # type: ignore[assignment]

    def env(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError(f"PettingZoo environment is unavailable: {_IMPORT_ERROR}")

    def parallel_env(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError(f"PettingZoo environment is unavailable: {_IMPORT_ERROR}")

__all__ = ["GeneralsAECEnv", "env", "parallel_env"]
