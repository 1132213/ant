from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from typing import Callable

import numpy as np

from logic.game_rules import is_game_over
from logic.gamestate import GameState, init_generals, update_round

from SDK.mcts_features import (
    apply_ops_in_place,
    candidate_turn_plans,
    clone_state,
    extract_features,
    heuristic_value,
    simulate_turn,
)
from SDK.training_base import SelfPlayTrainerBase, TrainLoopConfig, ValueModelBase, seed_python


AI = Callable[[int, int, GameState], list[list[int]]]


@dataclass(slots=True)
class SearchConfig:
    simulations: int = 12
    max_depth: int = 3
    c_puct: float = 1.35
    max_candidates: int = 8
    heuristic_weight: float = 0.7


TrainConfig = TrainLoopConfig


@dataclass(slots=True)
class SearchNode:
    state: GameState
    to_play: int
    prior: float
    ops: list[list[int]] | None = None
    visits: int = 0
    value_sum: float = 0.0
    children: list["SearchNode"] = field(default_factory=list)
    winner: int = -1

    @property
    def mean_value(self) -> float:
        if self.visits <= 0:
            return 0.0
        return self.value_sum / self.visits


class LinearValueModel(ValueModelBase):
    def __init__(self, feature_dim: int, weights: np.ndarray | None = None, bias: float = 0.0, updates: int = 0):
        self.feature_dim = int(feature_dim)
        self.weights = np.zeros(self.feature_dim, dtype=np.float32) if weights is None else weights.astype(np.float32)
        self.bias = float(bias)
        self.updates = int(updates)

    def predict_raw(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        return x @ self.weights + self.bias

    def predict(self, features: np.ndarray) -> np.ndarray:
        raw = self.predict_raw(features)
        return np.tanh(raw)

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
        l2: float,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        if len(features) == 0:
            return {"loss": 0.0}
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32).reshape(-1)
        n = len(x)
        last_loss = 0.0
        for _ in range(max(1, epochs)):
            order = np.arange(n)
            rng.shuffle(order)
            for start in range(0, n, max(1, batch_size)):
                idx = order[start:start + max(1, batch_size)]
                xb = x[idx]
                yb = y[idx]
                raw = xb @ self.weights + self.bias
                pred = np.tanh(raw)
                err = pred - yb
                last_loss = float(np.mean(err * err) + l2 * np.sum(self.weights * self.weights))
                grad_scale = (2.0 / max(1, len(idx))) * err * (1.0 - pred * pred)
                grad_w = xb.T @ grad_scale + 2.0 * l2 * self.weights
                grad_b = float(np.sum(grad_scale))
                self.weights -= lr * grad_w.astype(np.float32)
                self.bias -= lr * grad_b
                self.updates += 1
        return {"loss": last_loss}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(
            path,
            feature_dim=np.array([self.feature_dim], dtype=np.int32),
            weights=self.weights.astype(np.float32),
            bias=np.array([self.bias], dtype=np.float32),
            updates=np.array([self.updates], dtype=np.int32),
        )

    @classmethod
    def load(cls, path: str) -> "LinearValueModel":
        with np.load(path) as data:
            feature_dim = int(data["feature_dim"][0])
            weights = data["weights"].astype(np.float32)
            bias = float(data["bias"][0])
            updates = int(data["updates"][0]) if "updates" in data else 0
        return cls(feature_dim=feature_dim, weights=weights, bias=bias, updates=updates)


class MCTSAgent:
    def __init__(self, model: LinearValueModel | None = None, search_config: SearchConfig | None = None):
        self.search_config = search_config or SearchConfig()
        if model is None:
            sample = extract_features(GameState(), 0)
            model = LinearValueModel(feature_dim=int(sample.shape[0]))
        self.model = model

    def evaluate(self, state: GameState, player: int) -> float:
        winner = is_game_over(state)
        if winner != -1:
            return 1.0 if winner == player else -1.0
        features = extract_features(state, player)
        learned = float(self.model.predict(features))
        heuristic = heuristic_value(state, player)
        weight = float(np.clip(self.search_config.heuristic_weight, 0.0, 1.0))
        return float(np.clip(weight * heuristic + (1.0 - weight) * learned, -1.0, 1.0))

    def _expand(self, node: SearchNode) -> None:
        if node.children:
            return
        winner = is_game_over(node.state)
        if winner != -1:
            node.winner = winner
            return
        plans = candidate_turn_plans(
            node.state,
            node.to_play,
            node.state.round,
            max_candidates=self.search_config.max_candidates,
        )
        if not plans:
            return
        scored: list[tuple[float, list[list[int]], GameState]] = []
        for ops in plans:
            child_state = simulate_turn(node.state, node.to_play, ops)
            child_score = self.evaluate(child_state, node.to_play)
            scored.append((child_score, ops, child_state))
        logits = np.array([score for score, _, _ in scored], dtype=np.float32)
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs_sum = float(np.sum(probs))
        if probs_sum <= 0:
            priors = np.full(len(scored), 1.0 / len(scored), dtype=np.float32)
        else:
            priors = probs / probs_sum
        for prior, (_, ops, child_state) in zip(priors, scored):
            node.children.append(
                SearchNode(
                    state=child_state,
                    to_play=1 - node.to_play,
                    prior=float(prior),
                    ops=ops,
                )
            )

    def _select_child(self, node: SearchNode) -> SearchNode:
        sqrt_visits = math.sqrt(max(1, node.visits))
        best_score = -1e18
        best_child = node.children[0]
        for child in node.children:
            exploit = -child.mean_value
            explore = self.search_config.c_puct * child.prior * sqrt_visits / (1 + child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _search(self, node: SearchNode, depth: int) -> float:
        winner = is_game_over(node.state)
        if winner != -1:
            value = 1.0 if winner == node.to_play else -1.0
            node.visits += 1
            node.value_sum += value
            node.winner = winner
            return value
        if depth >= self.search_config.max_depth:
            value = self.evaluate(node.state, node.to_play)
            node.visits += 1
            node.value_sum += value
            return value
        if not node.children:
            self._expand(node)
            value = self.evaluate(node.state, node.to_play)
            node.visits += 1
            node.value_sum += value
            return value
        child = self._select_child(node)
        value = -self._search(child, depth + 1)
        node.visits += 1
        node.value_sum += value
        return value

    def search(self, state: GameState, player: int) -> tuple[list[list[int]], dict[str, float]]:
        root = SearchNode(state=clone_state(state), to_play=player, prior=1.0)
        self._expand(root)
        if not root.children:
            return [[8]], {"root_value": self.evaluate(state, player), "root_children": 0}
        if len(root.children) == 1:
            return root.children[0].ops or [[8]], {"root_value": root.children[0].mean_value, "root_children": 1}
        for _ in range(max(1, self.search_config.simulations)):
            self._search(root, 0)
        best = max(root.children, key=lambda child: (child.visits, child.mean_value))
        return best.ops or [[8]], {
            "root_value": float(root.mean_value),
            "root_children": float(len(root.children)),
            "best_visits": float(best.visits),
        }

    def policy(self, round_idx: int, my_seat: int, state: GameState) -> list[list[int]]:
        del round_idx
        ops, _ = self.search(state, my_seat)
        return ops


def load_ai_callable(spec: str) -> AI:
    if ":" in spec:
        mod_name, func_name = spec.split(":", 1)
    else:
        mod_name, func_name = f"AI.ai_{spec}", "policy"
    module = __import__(mod_name, fromlist=[func_name])
    return getattr(module, func_name)


def _init_training_state(seed: int | None = None) -> GameState:
    seed_python(seed)
    state = GameState()
    state.disable_replay = True
    init_generals(state)
    return state


class MCTSSelfPlayTrainer(SelfPlayTrainerBase):
    def __init__(self, agent: MCTSAgent, config: TrainConfig):
        super().__init__(agent.model, config)
        self.agent = agent
        self.fixed_pool: dict[str, AI] = {
            "handcraft": load_ai_callable("handcraft"),
            "greedy": load_ai_callable("greedy"),
            "random_safe": load_ai_callable("random_safe"),
        }

    def build_trainable_policy(self) -> AI:
        return self.agent.policy

    def resolve_fixed_opponent(self, name: str) -> AI:
        return self.fixed_pool[name] if name in self.fixed_pool else load_ai_callable(name)

    def build_initial_state(self, seed: int | None) -> GameState:
        return _init_training_state(seed)

    def feature_vector(self, state: GameState, player: int) -> np.ndarray:
        return extract_features(state, player)

    def apply_turn(self, state: GameState, player: int, ops: list[list[int]]) -> None:
        apply_ops_in_place(state, player, ops)

    def finish_round(self, state: GameState) -> None:
        update_round(state)


def train_mcts_selfplay(
    save_path: str,
    *,
    search_config: SearchConfig | None = None,
    train_config: TrainConfig | None = None,
    warm_start_path: str | None = None,
) -> dict[str, float]:
    search = search_config or SearchConfig()
    cfg = train_config or TrainConfig()
    rng = np.random.default_rng(cfg.seed)

    if warm_start_path and os.path.exists(warm_start_path):
        model = LinearValueModel.load(warm_start_path)
    else:
        sample = extract_features(GameState(), 0)
        model = LinearValueModel(feature_dim=int(sample.shape[0]))
    agent = MCTSAgent(model=model, search_config=search)
    trainer = MCTSSelfPlayTrainer(agent, cfg)
    trainer.rng = rng
    return trainer.train(save_path)
