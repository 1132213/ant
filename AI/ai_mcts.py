from __future__ import annotations

import os

from logic.gamestate import GameState

from SDK.mcts_agent import LinearValueModel, MCTSAgent, SearchConfig


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _default_model_path() -> str:
    env_path = os.environ.get("MCTS_MODEL")
    if env_path:
        return env_path
    candidates = [
        os.path.join(_THIS_DIR, "selfplay", "mcts_value.npz"),
        os.path.join(_THIS_DIR, "AI", "selfplay", "mcts_value.npz"),
        os.path.join("AI", "selfplay", "mcts_value.npz"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


MODEL_PATH = _default_model_path()
SIMULATIONS = int(os.environ.get("MCTS_SIMULATIONS", "16"))
SEARCH_DEPTH = int(os.environ.get("MCTS_DEPTH", "3"))
HEURISTIC_WEIGHT = float(os.environ.get("MCTS_HEURISTIC_WEIGHT", "0.7"))
CPUCT = float(os.environ.get("MCTS_CPUCT", "1.35"))

_AGENT: MCTSAgent | None = None


def _build_agent() -> MCTSAgent:
    model = LinearValueModel.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    config = SearchConfig(
        simulations=SIMULATIONS,
        max_depth=SEARCH_DEPTH,
        c_puct=CPUCT,
        heuristic_weight=HEURISTIC_WEIGHT,
    )
    return MCTSAgent(model=model, search_config=config)


def policy(round_idx: int, my_seat: int, state: GameState) -> list[list[int]]:
    del round_idx
    global _AGENT
    if _AGENT is None:
        _AGENT = _build_agent()
    return _AGENT.policy(state.round, my_seat, state)


def ai_func(state: GameState) -> list[list[int]]:
    return policy(getattr(state, "round", 1), 0, state)
