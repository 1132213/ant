from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from SDK.engine import GameState


class EngineBackend(Protocol):
    name: str

    def initial_state(self, seed: int = 0) -> GameState: ...


@dataclass(slots=True)
class PythonBackend:
    name: str = "python"

    def initial_state(self, seed: int = 0) -> GameState:
        return GameState.initial(seed=seed)


class NativeBackendUnavailable(RuntimeError):
    pass


@dataclass(slots=True)
class NativeBackend:
    module: object
    name: str = "native"

    def initial_state(self, seed: int = 0) -> GameState:
        from SDK.native_adapter import NativeGameStateAdapter

        return NativeGameStateAdapter.initial(seed)


def load_backend(prefer_native: bool = False) -> EngineBackend:
    if not prefer_native:
        return PythonBackend()
    try:
        from SDK import native_antwar  # type: ignore
    except Exception as exc:  # pragma: no cover - optional acceleration path
        raise NativeBackendUnavailable(str(exc)) from exc
    return NativeBackend(native_antwar)
