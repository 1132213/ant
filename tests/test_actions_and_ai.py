from __future__ import annotations

import json
from pathlib import Path

from AI.ai_greedy import GreedyAgent, GreedyHeuristicFallbackAgent
from AI.ai_mcts import MCTSAgent
from AI.ai_random import RandomAgent
from SDK.actions import ActionCatalog
from SDK.backend import load_backend
from SDK.constants import OperationType
from SDK.engine import GameState, PublicRoundState
from SDK.model import Ant, Operation

FIXTURE_PATH = Path("tests/fixtures/greedy_oracle_corpus.json")


def _normalize_op(operation) -> tuple[int, int, int]:
    return (int(operation.op_type), int(operation.arg0), int(operation.arg1))


def _normalize_ops(operations) -> list[tuple[int, int, int]]:
    return [_normalize_op(operation) for operation in operations]


def _load_cases() -> list[dict]:
    return json.loads(FIXTURE_PATH.read_text())


def _ops_from_rows(rows: list[list[int]]) -> list[Operation]:
    return [Operation(OperationType(row[0]), row[1], row[2]) for row in rows]


def _public_state_from_dict(payload: dict) -> PublicRoundState:
    return PublicRoundState(
        round_index=int(payload["round_index"]),
        towers=[tuple(row) for row in payload["towers"]],
        ants=[tuple(row) for row in payload["ants"]],
        coins=tuple(payload["coins"]),
        camps_hp=tuple(payload["camps_hp"]),
    )


def _replay_case(case: dict) -> list[tuple[int, int, int]]:
    player = int(case["player"])
    seed = int(case["seed"])
    agent = GreedyAgent(seed=seed)
    state = GameState.initial(seed=seed)
    agent.on_match_start(player, seed)
    for event in case["events"]:
        if event["kind"] == "self_ops":
            agent.on_self_operations(_ops_from_rows(event["operations"]))
        elif event["kind"] == "opponent_ops":
            agent.on_opponent_operations(_ops_from_rows(event["operations"]))
        elif event["kind"] == "round_state":
            public_state = _public_state_from_dict(event["state"])
            state.sync_public_round_state(public_state)
            agent.on_round_state(public_state)
        else:  # pragma: no cover - fixture schema guard
            raise AssertionError(f"unknown event kind: {event['kind']}")
    return _normalize_ops(agent.choose_operations(state, player))


def test_action_catalog_returns_legal_bundles() -> None:
    state = GameState.initial(seed=11)
    catalog = ActionCatalog(max_actions=32)
    bundles = catalog.build(state, 0)
    assert bundles
    assert bundles[0].name
    for bundle in bundles[:10]:
        accepted = []
        for operation in bundle.operations:
            assert state.can_apply_operation(0, operation, accepted)
            accepted.append(operation)


def test_random_agent_selects_non_empty_legal_bundle() -> None:
    state = GameState.initial(seed=5)
    agent = RandomAgent(seed=5)
    bundles = agent.list_bundles(state, 0)
    bundle = agent.choose_bundle(state, 0, bundles=bundles)
    assert bundle in bundles


def test_greedy_module_has_no_native_runtime_import() -> None:
    content = Path("AI/ai_greedy.py").read_text()
    assert "native_antwar" not in content


def test_repo_sources_no_longer_reference_ai_expert_runtime() -> None:
    targets = [
        Path("SDK/native_antwar.cpp"),
        Path("SDK/native_adapter.py"),
        Path("SDK/backend.py"),
        Path("tools/setup_native.py"),
    ]
    for path in targets:
        content = path.read_text()
        assert "AI_expert" not in content
        assert "expert_oracle" not in content
        assert "expert_reset" not in content


def test_default_backend_stays_python() -> None:
    assert load_backend().name == "python"


def test_native_backend_can_boot_and_advance() -> None:
    state = load_backend(prefer_native=True).initial_state(seed=7)
    state.resolve_turn([], [])
    assert state.round_index == 1
    assert len(state.ants) == 2
    assert state.coins == [51, 51]


def test_greedy_runs_on_python_state_without_native_backend() -> None:
    state = GameState.initial(seed=9)
    agent = GreedyAgent(seed=9)
    agent.on_match_start(0, 9)
    operations = agent.choose_operations(state, 0)
    assert isinstance(operations, list)
    assert all(hasattr(operation, "to_protocol_tokens") for operation in operations)


def test_greedy_matches_recorded_oracle_corpus() -> None:
    for case in _load_cases():
        assert _replay_case(case) == [tuple(row) for row in case["expected"]], case["name"]


def test_fallback_greedy_builds_under_pressure() -> None:
    state = GameState.initial(seed=9)
    state.ants.append(Ant(0, 1, 6, 8, hp=10, level=0))
    agent = GreedyHeuristicFallbackAgent(seed=2)
    bundle = agent.choose_bundle(state, 0)
    assert bundle.name != "hold"


def test_mcts_agent_returns_legal_choice() -> None:
    state = GameState.initial(seed=13)
    state.ants.append(Ant(1, 1, 6, 8, hp=10, level=0))
    agent = MCTSAgent(iterations=6, max_depth=2, seed=2)
    bundles = agent.list_bundles(state, 0)
    bundle = agent.choose_bundle(state, 0, bundles=bundles)
    assert bundle in bundles
    assert all(op.op_type in OperationType for op in bundle.operations)
