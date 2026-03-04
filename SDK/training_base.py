from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
from typing import Callable

import numpy as np

from logic.game_rules import is_game_over, tiebreak_now
from logic.gamestate import GameState


AI = Callable[[int, int, GameState], list[list[int]]]


@dataclass(slots=True)
class TrainLoopConfig:
    games: int = 40
    max_rounds: int = 80
    seed: int | None = None
    train_every: int = 4
    buffer_size: int = 4096
    fit_epochs: int = 8
    batch_size: int = 128
    lr: float = 0.05
    l2: float = 1e-4
    opponents: tuple[str, ...] = ("self", "handcraft")
    log_every: int = 1


@dataclass(slots=True)
class PolicyController:
    name: str
    policy: AI
    trainable: bool


@dataclass(slots=True)
class EpisodeSummary:
    winner: int
    state: GameState
    samples: list[tuple[np.ndarray, float]]
    opponent_name: str
    tracked_player: int | None


class ValueModelBase(ABC):
    feature_dim: int
    updates: int

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError


class SelfPlayTrainerBase(ABC):
    def __init__(self, model: ValueModelBase, config: TrainLoopConfig):
        self.model = model
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    @abstractmethod
    def build_trainable_policy(self) -> AI:
        raise NotImplementedError

    @abstractmethod
    def resolve_fixed_opponent(self, name: str) -> AI:
        raise NotImplementedError

    @abstractmethod
    def build_initial_state(self, seed: int | None) -> GameState:
        raise NotImplementedError

    @abstractmethod
    def feature_vector(self, state: GameState, player: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def apply_turn(self, state: GameState, player: int, ops: list[list[int]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def finish_round(self, state: GameState) -> None:
        raise NotImplementedError

    def log(self, message: str) -> None:
        print(message)

    def current_winner(self, state: GameState) -> int:
        return is_game_over(state)

    def final_winner(self, state: GameState) -> int:
        return tiebreak_now(state)

    def _play_episode(
        self,
        player0: PolicyController,
        player1: PolicyController,
        opponent_name: str,
        tracked_player: int | None,
        seed: int | None,
    ) -> EpisodeSummary:
        state = self.build_initial_state(seed)
        samples: list[tuple[np.ndarray, float]] = []
        round_idx = 1
        winner = -1
        controllers = {0: player0, 1: player1}
        while round_idx <= self.config.max_rounds and winner == -1:
            for player in (0, 1):
                ctrl = controllers[player]
                if ctrl.trainable:
                    features = self.feature_vector(state, player)
                    ops = ctrl.policy(round_idx, player, state)
                    samples.append((features, float(player)))
                else:
                    ops = ctrl.policy(round_idx, player, state)
                self.apply_turn(state, player, ops)
                winner = self.current_winner(state)
                if winner != -1:
                    break
            if winner != -1:
                break
            self.finish_round(state)
            round_idx += 1
        if winner == -1:
            winner = self.final_winner(state)
        labeled = [(features, 1.0 if int(player) == winner else -1.0) for features, player in samples]
        return EpisodeSummary(
            winner=winner,
            state=state,
            samples=labeled,
            opponent_name=opponent_name,
            tracked_player=tracked_player,
        )

    def _fit_buffer(self, replay_buffer: list[tuple[np.ndarray, float]]) -> dict[str, float]:
        if replay_buffer:
            x = np.stack([feat for feat, _ in replay_buffer], axis=0)
            y = np.array([target for _, target in replay_buffer], dtype=np.float32)
        else:
            x = np.empty((0, self.model.feature_dim), dtype=np.float32)
            y = np.empty((0,), dtype=np.float32)
        return self.model.fit(
            x,
            y,
            epochs=self.config.fit_epochs,
            batch_size=self.config.batch_size,
            lr=self.config.lr,
            l2=self.config.l2,
            rng=self.rng,
        )

    def train(self, save_path: str) -> dict[str, float]:
        replay_buffer: list[tuple[np.ndarray, float]] = []
        summary = {
            "games": 0.0,
            "selfplay_games": 0.0,
            "versus_games": 0.0,
            "wins": 0.0,
            "losses": 0.0,
            "draws": 0.0,
            "avg_rounds": 0.0,
            "last_loss": 0.0,
            "model_updates": float(self.model.updates),
        }
        self.log(
            "[train] "
            f"games={self.config.games} rounds={self.config.max_rounds} "
            f"train_every={self.config.train_every} buffer={self.config.buffer_size} "
            f"epochs={self.config.fit_epochs} batch={self.config.batch_size} "
            f"lr={self.config.lr} l2={self.config.l2} opponents={','.join(self.config.opponents)}"
        )

        trainable_policy = self.build_trainable_policy()
        for game_idx in range(self.config.games):
            opponent_name = self.config.opponents[game_idx % len(self.config.opponents)] if self.config.opponents else "self"
            if opponent_name == "self":
                player0 = PolicyController(name="self_p0", policy=trainable_policy, trainable=True)
                player1 = PolicyController(name="self_p1", policy=trainable_policy, trainable=True)
                tracked_player = None
            else:
                fixed_ai = self.resolve_fixed_opponent(opponent_name)
                if game_idx % 2 == 0:
                    player0 = PolicyController(name="learner", policy=trainable_policy, trainable=True)
                    player1 = PolicyController(name=opponent_name, policy=fixed_ai, trainable=False)
                    tracked_player = 0
                else:
                    player0 = PolicyController(name=opponent_name, policy=fixed_ai, trainable=False)
                    player1 = PolicyController(name="learner", policy=trainable_policy, trainable=True)
                    tracked_player = 1

            episode = self._play_episode(
                player0,
                player1,
                opponent_name=opponent_name,
                tracked_player=tracked_player,
                seed=None if self.config.seed is None else self.config.seed + game_idx,
            )
            replay_buffer.extend(episode.samples)
            if len(replay_buffer) > self.config.buffer_size:
                replay_buffer = replay_buffer[-self.config.buffer_size:]

            summary["games"] += 1.0
            summary["avg_rounds"] += float(episode.state.round)
            if episode.tracked_player is None:
                summary["selfplay_games"] += 1.0
            elif episode.winner == episode.tracked_player:
                summary["versus_games"] += 1.0
                summary["wins"] += 1.0
            elif episode.winner == 1 - episode.tracked_player:
                summary["versus_games"] += 1.0
                summary["losses"] += 1.0
            else:
                summary["versus_games"] += 1.0
                summary["draws"] += 1.0

            if (game_idx + 1) % max(1, self.config.log_every) == 0:
                seat = "self-play" if episode.tracked_player is None else f"seat={episode.tracked_player}"
                self.log(
                    f"[game {game_idx + 1}/{self.config.games}] "
                    f"opponent={episode.opponent_name} {seat} "
                    f"winner={episode.winner} rounds={episode.state.round} "
                    f"buffer={len(replay_buffer)}"
                )

            if (game_idx + 1) % max(1, self.config.train_every) == 0 or game_idx + 1 == self.config.games:
                fit_info = self._fit_buffer(replay_buffer)
                summary["last_loss"] = float(fit_info.get("loss", 0.0))
                summary["model_updates"] = float(self.model.updates)
                self.log(
                    f"[fit] after_game={game_idx + 1} "
                    f"samples={len(replay_buffer)} loss={summary['last_loss']:.6f} "
                    f"updates={int(summary['model_updates'])}"
                )

        if summary["games"] > 0:
            summary["avg_rounds"] /= summary["games"]
        self.model.save(save_path)
        self.log(f"[save] path={save_path}")
        return summary


def seed_python(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
