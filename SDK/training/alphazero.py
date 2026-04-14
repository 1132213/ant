from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from SDK.alphazero import (
    PolicyValueNet,
    PolicyValueNetConfig,
    PriorGuidedMCTS,
    SearchConfig,
    build_policy_value_net,
)
from SDK.training.env import AntWarParallelEnv
from SDK.training.logging_utils import TrainingLogger
from SDK.utils.actions import ActionCatalog
from SDK.utils.features import FeatureExtractor


import sys
def compat_dataclass(**kwargs):
    if sys.version_info < (3, 10) and 'slots' in kwargs:
        del kwargs['slots']
    return dataclass(**kwargs)

@compat_dataclass(slots=True)
class AlphaZeroTrainerConfig:
    batches: int = 1
    episodes: int = 4
    workers: int = 4
    learning_rate: float = 1e-3
    value_weight: float = 1.0
    l2_weight: float = 1e-5
    search_iterations: int = 48
    max_depth: int = 4
    c_puct: float = 1.25
    root_action_limit: int = 16
    child_action_limit: int = 10
    dirichlet_alpha: float = 0.35
    dirichlet_epsilon: float = 0.25
    prior_mix: float = 0.7
    value_mix: float = 0.7
    value_scale: float = 350.0
    root_temperature: float = 1.0
    temperature_drop_round: int = 96
    seed: int = 0
    max_rounds: int = 128
    max_actions: int = 96
    hidden_dim: int = 128
    hidden_dim2: int = 64
    checkpoint_path: str = "checkpoints/ai_mcts_latest.npz"
    resume_from: str | None = None
    start_batch: int = 0
    evaluation_episodes: int = 2
    save_interval: int = 1


import sys
def compat_dataclass(**kwargs):
    if sys.version_info < (3, 10) and 'slots' in kwargs:
        del kwargs['slots']
    return dataclass(**kwargs)

@compat_dataclass(slots=True)
class SelfPlaySample:
    observation: np.ndarray
    mask: np.ndarray
    policy: np.ndarray
    value: float


import sys
def compat_dataclass(**kwargs):
    if sys.version_info < (3, 10) and 'slots' in kwargs:
        del kwargs['slots']
    return dataclass(**kwargs)

@compat_dataclass(slots=True)
class SelfPlayBatch:
    observations: np.ndarray
    masks: np.ndarray
    policies: np.ndarray
    values: np.ndarray


import sys
def compat_dataclass(**kwargs):
    if sys.version_info < (3, 10) and 'slots' in kwargs:
        del kwargs['slots']
    return dataclass(**kwargs)

@compat_dataclass(slots=True)
class EpisodeSummary:
    seed: int
    rounds: int
    winner: int | None
    reward_player_0: float
    reward_player_1: float
    outcome_player_0: float
    outcome_player_1: float


class AlphaZeroSelfPlayTrainer:
    def __init__(
        self,
        env_factory,
        config: AlphaZeroTrainerConfig | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        self.env_factory = env_factory
        self.config = config or AlphaZeroTrainerConfig()
        self.logger = logger
        self.feature_extractor = FeatureExtractor(max_actions=self.config.max_actions)
        self.action_catalog = ActionCatalog(max_actions=self.config.max_actions, feature_extractor=self.feature_extractor)
        self.model = self._build_or_resume_model()
        self.search = PriorGuidedMCTS(
            model=self.model,
            search_config=self._build_search_config(exploration=True),
            feature_extractor=self.feature_extractor,
            action_catalog=self.action_catalog,
        )
        self.eval_search = PriorGuidedMCTS(
            model=self.model,
            search_config=self._build_search_config(exploration=False),
            feature_extractor=self.feature_extractor,
            action_catalog=self.action_catalog,
        )
        self.heuristic_search = PriorGuidedMCTS(
            model=None,
            search_config=self._build_search_config(exploration=False),
            feature_extractor=self.feature_extractor,
            action_catalog=self.action_catalog,
        )

    def _build_search_config(self, exploration: bool) -> SearchConfig:
        return SearchConfig(
            iterations=self.config.search_iterations,
            max_depth=self.config.max_depth,
            c_puct=self.config.c_puct,
            root_action_limit=self.config.root_action_limit,
            child_action_limit=self.config.child_action_limit,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon if exploration else 0.0,
            prior_mix=self.config.prior_mix,
            value_mix=self.config.value_mix,
            value_scale=self.config.value_scale,
            seed=self.config.seed,
        )

    def _build_or_resume_model(self) -> PolicyValueNet:
        resume_path = Path(self.config.resume_from) if self.config.resume_from else None
        if resume_path is not None and resume_path.exists():
            model = PolicyValueNet.from_checkpoint(resume_path)
            if model.action_dim != self.config.max_actions:
                raise ValueError(
                    f"checkpoint action_dim={model.action_dim} does not match max_actions={self.config.max_actions}"
                )
            return model
        return build_policy_value_net(
            feature_extractor=self.feature_extractor,
            action_dim=self.config.max_actions,
            config=PolicyValueNetConfig(
                hidden_dim=self.config.hidden_dim,
                hidden_dim2=self.config.hidden_dim2,
                seed=self.config.seed,
            ),
        )

    def _temperature_for_round(self, round_index: int) -> float:
        if round_index >= self.config.temperature_drop_round:
            return 1e-6
        return self.config.root_temperature

    def _value_target(self, env: AntWarParallelEnv, player: int) -> float:
        if env.state.terminal:
            if env.state.winner is None:
                return 0.0
            return 1.0 if env.state.winner == player else -1.0
        raw = self.feature_extractor.evaluate(env.state, player)
        return float(np.tanh(raw / self.config.value_scale))

    def collect_episode(self, seed: int) -> tuple[SelfPlayBatch, EpisodeSummary]:
        env = self.env_factory(seed=seed)
        try:
            observations, infos = env.reset(seed=seed)
            traces = {agent: [] for agent in env.possible_agents}
            total_reward = {agent: 0.0 for agent in env.possible_agents}
            rounds = 0
            while env.agents and rounds < self.config.max_rounds:
                actions = {}
                for player, agent_name in enumerate(env.possible_agents):
                    bundles = infos[agent_name]["bundles"]
                    result = self.search.search(
                        env.state,
                        player,
                        bundles=bundles,
                        temperature=self._temperature_for_round(env.state.round_index),
                        add_root_noise=True,
                    )
                    current = observations[agent_name]
                    traces[agent_name].append(
                        SelfPlaySample(
                            observation=self.feature_extractor.flatten_observation(current),
                            mask=current["action_mask"].astype(np.float32),
                            policy=result.policy.copy(),
                            value=0.0,
                        )
                    )
                    actions[agent_name] = result.action_index

                observations, rewards, terminations, truncations, infos = env.step(actions)
                for agent_name in env.possible_agents:
                    total_reward[agent_name] += float(rewards[agent_name])
                rounds += 1
                if all(terminations.values()) or all(truncations.values()):
                    break

            player_targets = {
                "player_0": self._value_target(env, 0),
                "player_1": self._value_target(env, 1),
            }
            observation_rows = []
            mask_rows = []
            policy_rows = []
            value_rows = []
            for agent_name in env.possible_agents:
                target_value = player_targets[agent_name]
                for sample in traces[agent_name]:
                    observation_rows.append(sample.observation)
                    mask_rows.append(sample.mask)
                    policy_rows.append(sample.policy)
                    value_rows.append(target_value)

            batch = SelfPlayBatch(
                observations=np.asarray(observation_rows, dtype=np.float32),
                masks=np.asarray(mask_rows, dtype=np.float32),
                policies=np.asarray(policy_rows, dtype=np.float32),
                values=np.asarray(value_rows, dtype=np.float32),
            )
            summary = EpisodeSummary(
                seed=seed,
                rounds=rounds,
                winner=env.state.winner,
                reward_player_0=round(total_reward["player_0"], 4),
                reward_player_1=round(total_reward["player_1"], 4),
                outcome_player_0=round(player_targets["player_0"], 4),
                outcome_player_1=round(player_targets["player_1"], 4),
            )
            return batch, summary
        finally:
            env.close()

    def _merge_batches(self, batches: list[SelfPlayBatch]) -> SelfPlayBatch:
        return SelfPlayBatch(
            observations=np.concatenate([batch.observations for batch in batches], axis=0),
            masks=np.concatenate([batch.masks for batch in batches], axis=0),
            policies=np.concatenate([batch.policies for batch in batches], axis=0),
            values=np.concatenate([batch.values for batch in batches], axis=0),
        )

    def update_from_batch(self, batch: SelfPlayBatch) -> dict[str, float]:
        metrics = self.model.update(
            observations=batch.observations,
            masks=batch.masks,
            policy_targets=batch.policies,
            value_targets=batch.values,
            learning_rate=self.config.learning_rate,
            value_weight=self.config.value_weight,
            l2_weight=self.config.l2_weight,
        )
        metrics["samples"] = float(len(batch.values))
        metrics["mean_target_value"] = float(np.mean(batch.values))
        return metrics

    def _play_evaluation_episode(self, seed: int, trained_side: int) -> tuple[int | None, int]:
        env = self.env_factory(seed=seed)
        try:
            _, infos = env.reset(seed=seed)
            rounds = 0
            while env.agents and rounds < self.config.max_rounds:
                actions = {}
                for player, agent_name in enumerate(env.possible_agents):
                    bundles = infos[agent_name]["bundles"]
                    if player == trained_side:
                        result = self.eval_search.search(env.state, player, bundles=bundles, temperature=1e-6)
                    else:
                        result = self.heuristic_search.search(env.state, player, bundles=bundles, temperature=1e-6)
                    actions[agent_name] = result.action_index
                _, _, terminations, truncations, infos = env.step(actions)
                rounds += 1
                if all(terminations.values()) or all(truncations.values()):
                    break
            return env.state.winner, rounds
        finally:
            env.close()

    def evaluate_against_heuristic(self, num_episodes: int | None = None) -> dict[str, float]:
        games = num_episodes if num_episodes is not None else self.config.evaluation_episodes
        if games <= 0:
            return {"eval_episodes": 0.0, "eval_win_rate": 0.0, "eval_draw_rate": 0.0}
        trained_wins = 0
        draws = 0
        total_rounds = 0
        for episode_index in range(games):
            trained_side = episode_index % 2
            winner, rounds = self._play_evaluation_episode(self.config.seed + 10_000 + episode_index, trained_side)
            total_rounds += rounds
            if winner is None:
                draws += 1
            elif winner == trained_side:
                trained_wins += 1
        return {
            "eval_episodes": float(games),
            "eval_win_rate": float(trained_wins / games),
            "eval_draw_rate": float(draws / games),
            "eval_avg_rounds": float(total_rounds / games),
        }

    def save_checkpoint(self, batch_index: int = -1) -> str:
        self.model.save(self.config.checkpoint_path)
        
        # Save a numbered checkpoint every save_interval batches
        if batch_index >= 0 and self.config.save_interval > 0 and batch_index % self.config.save_interval == 0:
            numbered_path = self.config.checkpoint_path.replace(".npz", f"_{batch_index:04d}.npz")
            self.model.save(numbered_path)
            
        return str(Path(self.config.checkpoint_path))

    def train(self, num_batches: int | None = None) -> tuple[list[dict[str, float]], list[EpisodeSummary]]:
        updates = num_batches if num_batches is not None else self.config.batches
        history: list[dict[str, float]] = []
        samples: list[EpisodeSummary] = []
        for batch_index in range(updates):
            episode_batches = []
            episode_summaries = []
            for episode_offset in range(self.config.episodes):
                seed = self.config.seed + batch_index * 1_000 + episode_offset
                batch, summary = self.collect_episode(seed=seed)
                episode_batches.append(batch)
                episode_summaries.append(summary)
                if self.logger is not None:
                    self.logger.log_episode(batch_index=batch_index, episode_index=episode_offset, payload=asdict(summary))
            merged = self._merge_batches(episode_batches)
            metrics = self.update_from_batch(merged)
            metrics["batch"] = float(batch_index)
            metrics["episodes"] = float(self.config.episodes)
            metrics["checkpoint_saved"] = 1.0
            checkpoint_path = self.save_checkpoint(batch_index=batch_index)
            metrics.update(self.evaluate_against_heuristic())
            history.append(metrics)
            samples.extend(episode_summaries)
            if self.logger is not None:
                self.logger.log_batch_metrics(batch_index=batch_index, payload=metrics)
                self.logger.log_checkpoint(batch_index=batch_index, checkpoint_path=checkpoint_path)
        return history, samples
