from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import random
import sys

import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError:
    pass

from SDK.backend import create_python_backend_state
from SDK.backend.state import BackendState
from SDK.utils.actions import ActionBundle, ActionCatalog
from SDK.utils.features import FeatureExtractor


def _relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0).astype(np.float32, copy=False)


def _softmax(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return logits.astype(np.float32, copy=False)
    shifted = logits - np.max(logits)
    exp = np.exp(shifted).astype(np.float32, copy=False)
    total = float(np.sum(exp))
    if total <= 0.0:
        result = np.zeros_like(logits, dtype=np.float32)
        result[0] = 1.0
        return result
    return (exp / total).astype(np.float32, copy=False)


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = logits.astype(np.float32, copy=True)
    masked[mask <= 0] = -1e9
    if np.all(mask <= 0):
        result = np.zeros_like(masked, dtype=np.float32)
        result[0] = 1.0
        return result
    return _softmax(masked)


def _normalize_policy(policy: np.ndarray, fallback_index: int = 0) -> np.ndarray:
    total = float(np.sum(policy))
    if total <= 0.0:
        fallback = np.zeros_like(policy, dtype=np.float32)
        if fallback.size:
            fallback[min(max(fallback_index, 0), fallback.size - 1)] = 1.0
        return fallback
    return (policy / total).astype(np.float32, copy=False)


def _heuristic_bundle_policy(bundles: list[ActionBundle]) -> np.ndarray:
    if not bundles:
        return np.zeros(0, dtype=np.float32)
    scores = np.asarray([bundle.score for bundle in bundles], dtype=np.float32)
    centered = (scores - np.max(scores)) / 8.0
    return _softmax(centered)


def _terminal_value(state: BackendState, player: int) -> float | None:
    if not state.terminal:
        return None
    if state.winner is None:
        return 0.0
    return 1.0 if state.winner == player else -1.0


import sys
def compat_dataclass(**kwargs):
    if sys.version_info < (3, 10) and 'slots' in kwargs:
        del kwargs['slots']
    return dataclass(**kwargs)

@compat_dataclass(slots=True)
class PolicyValueNetConfig:
    hidden_dim: int = 128
    hidden_dim2: int = 64
    seed: int = 0


import sys
def compat_dataclass(**kwargs):
    if sys.version_info < (3, 10) and 'slots' in kwargs:
        del kwargs['slots']
    return dataclass(**kwargs)

@compat_dataclass(slots=True)
class PolicyValueInference:
    priors: np.ndarray
    value: float
    observation: np.ndarray
    mask: np.ndarray


import sys
def compat_dataclass(**kwargs):
    if sys.version_info < (3, 10) and 'slots' in kwargs:
        del kwargs['slots']
    return dataclass(**kwargs)

@compat_dataclass(slots=True)
class SearchConfig:
    iterations: int = 64
    max_depth: int = 4
    c_puct: float = 1.25
    root_action_limit: int = 16
    child_action_limit: int = 10
    dirichlet_alpha: float = 0.35
    dirichlet_epsilon: float = 0.25
    prior_mix: float = 0.7
    value_mix: float = 0.7
    value_scale: float = 350.0
    seed: int = 0


import sys
def compat_dataclass(**kwargs):
    if sys.version_info < (3, 10) and 'slots' in kwargs:
        del kwargs['slots']
    return dataclass(**kwargs)

@compat_dataclass(slots=True)
class SearchResult:
    action_index: int
    bundle: ActionBundle
    policy: np.ndarray
    root_value: float
    visit_count: int
    priors: np.ndarray


import sys
def compat_dataclass(**kwargs):
    if sys.version_info < (3, 10) and 'slots' in kwargs:
        del kwargs['slots']
    return dataclass(**kwargs)

@compat_dataclass(slots=True)
class SearchNode:
    state: BackendState
    player: int
    prior: float = 0.0
    bundle: ActionBundle | None = None
    action_index: int = 0
    depth: int = 0
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    bundles: list[ActionBundle] = field(default_factory=list)
    priors: np.ndarray | None = None
    children: list[SearchNode] = field(default_factory=list)

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class TorchPolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, hidden_dim2: int):
        super().__init__()
        self.board_channels = 35
        self.map_size = 19
        self.board_flat_size = self.board_channels * self.map_size * self.map_size
        self.stats_size = obs_dim - self.board_flat_size
        
        # Conv Body
        conv_channels = 64
        self.conv_in = nn.Conv2d(self.board_channels, conv_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(conv_channels)
        
        # 4 layers of ResBlocks
        self.res_blocks = nn.Sequential(
            ResBlock(conv_channels),
            ResBlock(conv_channels),
            ResBlock(conv_channels),
            ResBlock(conv_channels)
        )
        
        # Policy Head (2 channels -> flat -> Linear)
        self.policy_conv = nn.Conv2d(conv_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_flat_size = 2 * self.map_size * self.map_size
        self.policy_fc = nn.Linear(self.policy_flat_size + self.stats_size, action_dim)
        
        # Value Head (1 channel -> flat -> Linear -> Linear)
        self.value_conv = nn.Conv2d(conv_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_flat_size = 1 * self.map_size * self.map_size
        self.value_fc1 = nn.Linear(self.value_flat_size + self.stats_size, hidden_dim2)
        self.value_fc2 = nn.Linear(hidden_dim2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Unpack
        board_flat = x[:, :self.board_flat_size]
        stats = x[:, self.board_flat_size:]
        
        # Reshape board to 3D
        board_3d = board_flat.view(-1, self.board_channels, self.map_size, self.map_size)
        
        # CNN Feature Extraction
        c = F.relu(self.bn_in(self.conv_in(board_3d)))
        c = self.res_blocks(c)
        
        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(c)))
        p_flat = p.view(p.size(0), -1)
        p_fused = torch.cat([p_flat, stats], dim=1)
        logits = self.policy_fc(p_fused)
        
        # Value Head
        v = F.relu(self.value_bn(self.value_conv(c)))
        v_flat = v.view(v.size(0), -1)
        v_fused = torch.cat([v_flat, stats], dim=1)
        v = F.relu(self.value_fc1(v_fused))
        value = torch.tanh(self.value_fc2(v))
        
        return logits, value


class PolicyValueNet:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: PolicyValueNetConfig | None = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config or PolicyValueNetConfig()
        
        if "torch" not in sys.modules:
            raise ImportError("PyTorch is required for the GPU version of AlphaZero. Please install PyTorch first.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = TorchPolicyValueNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=self.config.hidden_dim,
            hidden_dim2=self.config.hidden_dim2,
        )
        
        # GPU / Multi-GPU 支持
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.loaded_from: str | None = None
        
        import threading
        self._predict_lock = threading.Lock()
        
        # 预分配推理专用的 Tensor Buffer，避免 MCTS 中频繁的内存分配与释放
        self._obs_buffer = torch.zeros((1, obs_dim), dtype=torch.float32, device=self.device)
        self._mask_buffer = torch.zeros((action_dim,), dtype=torch.float32, device=self.device)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> PolicyValueNet:
        checkpoint = np.load(Path(path), allow_pickle=False)
        config = PolicyValueNetConfig(
            hidden_dim=int(checkpoint["hidden_dim"]),
            hidden_dim2=int(checkpoint["hidden_dim2"]),
            seed=int(checkpoint["seed"]),
        )
        network = cls(obs_dim=int(checkpoint["obs_dim"]), action_dim=int(checkpoint["action_dim"]), config=config)
        
        base_model = network.model.module if isinstance(network.model, nn.DataParallel) else network.model
        state_dict = base_model.state_dict()
        
        with torch.no_grad():
            for k in state_dict.keys():
                if k in checkpoint:
                    state_dict[k].copy_(torch.from_numpy(checkpoint[k]))
                elif k == "conv_in.weight" and "w1" in checkpoint:
                    print(f"Warning: Checkpoint {path} appears to be an old MLP format. Random weights will be used for ResNet.")
                    break
            
        network.loaded_from = str(Path(path))
        return network

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        base_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        state_dict = base_model.state_dict()
        np_dict = {k: v.detach().cpu().numpy() for k, v in state_dict.items()}
        
        np.savez(
            target,
            obs_dim=np.int64(self.obs_dim),
            action_dim=np.int64(self.action_dim),
            hidden_dim=np.int64(self.config.hidden_dim),
            hidden_dim2=np.int64(self.config.hidden_dim2),
            seed=np.int64(self.config.seed),
            **np_dict
        )
        self.loaded_from = str(target)

    def predict(self, observation: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
        self.model.eval()
        # 使用 inference_mode 替代 no_grad，进一步降低底层开销
        with torch.inference_mode(), self._predict_lock:
            # 使用 non_blocking 拷贝到预分配的 Buffer 中，避免新建 Tensor
            self._obs_buffer.copy_(torch.from_numpy(observation.astype(np.float32)), non_blocking=True)
            self._mask_buffer.copy_(torch.from_numpy(mask.astype(np.float32)), non_blocking=True)
            
            logits, values = self.model(self._obs_buffer)
            logits = logits.squeeze(0)
            
            masked_logits = logits.clone()
            masked_logits[self._mask_buffer <= 0] = -1e9
            
            masked_logits = masked_logits - torch.max(masked_logits)
            exp_logits = torch.exp(masked_logits) * self._mask_buffer
            denom = torch.sum(exp_logits)
            
            if denom <= 0:
                priors = torch.zeros_like(logits)
                priors[0] = 1.0
            else:
                priors = exp_logits / denom
                
            return priors.cpu().numpy(), float(values.item())

    def update(
        self,
        observations: np.ndarray,
        masks: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
        learning_rate: float = 1e-3,
        value_weight: float = 1.0,
        l2_weight: float = 1e-5,
    ) -> dict[str, float]:
        self.model.train()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
            param_group['weight_decay'] = l2_weight

        obs_tensor = torch.from_numpy(observations.astype(np.float32)).to(self.device)
        mask_tensor = torch.from_numpy(masks.astype(np.float32)).to(self.device)
        policy_targets_tensor = torch.from_numpy(policy_targets.astype(np.float32)).to(self.device)
        value_targets_tensor = torch.from_numpy(value_targets.astype(np.float32)).unsqueeze(1).to(self.device)

        self.optimizer.zero_grad()
        logits, values = self.model(obs_tensor)
        
        value_loss = F.mse_loss(values, value_targets_tensor)
        
        masked_logits = logits.clone()
        masked_logits[mask_tensor <= 0] = -1e9
        
        policy_targets_tensor = policy_targets_tensor * mask_tensor
        denom = torch.sum(policy_targets_tensor, dim=1, keepdim=True)
        denom[denom <= 0] = 1.0
        policy_targets_tensor = policy_targets_tensor / denom
        
        log_probs = F.log_softmax(masked_logits, dim=1)
        policy_loss = -torch.sum(policy_targets_tensor * log_probs, dim=1).mean()
        
        loss = policy_loss + value_weight * value_loss
        loss.backward()
        self.optimizer.step()
        
        with torch.no_grad():
            probs = torch.exp(log_probs) * mask_tensor
            entropy = -torch.sum(probs * log_probs, dim=1).mean().item()
        
        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy),
            "mean_value_target": float(value_targets_tensor.mean().item()),
            "mean_prediction": float(values.mean().item()),
        }


class PriorGuidedMCTS:
    def __init__(
        self,
        model: PolicyValueNet | None = None,
        search_config: SearchConfig | None = None,
        feature_extractor: FeatureExtractor | None = None,
        action_catalog: ActionCatalog | None = None,
    ) -> None:
        self.model = model
        self.search_config = search_config or SearchConfig()
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.action_catalog = action_catalog or ActionCatalog(feature_extractor=self.feature_extractor)
        self.rng = random.Random(self.search_config.seed)

    @property
    def action_dim(self) -> int:
        return self.action_catalog.max_actions

    def _heuristic_value(self, state: BackendState, player: int) -> float:
        terminal = _terminal_value(state, player)
        if terminal is not None:
            return terminal
        raw = self.feature_extractor.evaluate(state, player)
        return float(np.tanh(raw / self.search_config.value_scale))

    def _blend_policy_value(
        self,
        state: BackendState,
        player: int,
        bundles: list[ActionBundle],
    ) -> PolicyValueInference:
        action_mask = self.action_catalog.action_mask(bundles).astype(np.float32)
        observation = self.feature_extractor.encode_observation(state, player, action_mask)
        flat = self.feature_extractor.flatten_observation(observation)
        heuristic_priors = _heuristic_bundle_policy(bundles)
        heuristic_value = self._heuristic_value(state, player)
        if self.model is None:
            blended_priors = heuristic_priors
            blended_value = heuristic_value
        else:
            model_priors, model_value = self.model.predict(flat, action_mask)
            mixed_policy = self.search_config.prior_mix * model_priors[: len(bundles)]
            mixed_policy += (1.0 - self.search_config.prior_mix) * heuristic_priors
            blended_priors = _normalize_policy(mixed_policy)
            blended_value = float(
                self.search_config.value_mix * model_value
                + (1.0 - self.search_config.value_mix) * heuristic_value
            )
        full_priors = np.zeros(self.action_dim, dtype=np.float32)
        full_priors[: len(bundles)] = blended_priors
        return PolicyValueInference(
            priors=full_priors,
            value=float(blended_value),
            observation=flat,
            mask=action_mask,
        )

    def _predict_policy_only(self, state: BackendState, player: int, bundles: list[ActionBundle]) -> np.ndarray:
        if not bundles:
            return np.zeros(self.action_dim, dtype=np.float32)
        return self._blend_policy_value(state, player, bundles).priors

    def _predict_enemy_bundle(self, state: BackendState, player: int) -> ActionBundle:
        enemy = 1 - player
        enemy_bundles = self.action_catalog.build(state, enemy)
        if not enemy_bundles:
            fallback = self.action_catalog.build(state, player)
            return fallback[0]
        priors = self._predict_policy_only(state, enemy, enemy_bundles)[: len(enemy_bundles)]
        best_index = int(np.argmax(priors))
        return enemy_bundles[best_index]

    def _branch_indices(self, priors: np.ndarray, bundles: list[ActionBundle], limit: int) -> list[int]:
        if not bundles:
            return []
        branch_limit = min(limit, len(bundles))
        order = list(np.argsort(priors[: len(bundles)])[::-1])
        selected = order[:branch_limit]
        if 0 not in selected:
            selected.append(0)
        return sorted(set(int(index) for index in selected))

    def _expand(
        self,
        node: SearchNode,
        bundles: list[ActionBundle] | None = None,
        add_root_noise: bool = False,
    ) -> float:
        if node.expanded:
            return node.mean_value

        terminal = _terminal_value(node.state, node.player)
        if terminal is not None:
            node.expanded = True
            return terminal

        action_bundles = bundles or self.action_catalog.build(node.state, node.player)
        node.bundles = action_bundles
        inference = self._blend_policy_value(node.state, node.player, action_bundles)
        node.priors = inference.priors
        node.expanded = True

        if node.depth >= self.search_config.max_depth or not action_bundles:
            return inference.value

        prior_slice = inference.priors[: len(action_bundles)].astype(np.float32, copy=True)
        if add_root_noise and len(action_bundles) > 1 and self.search_config.dirichlet_epsilon > 0.0:
            noise = np.random.default_rng(self.rng.randrange(1 << 30)).dirichlet(
                [self.search_config.dirichlet_alpha] * len(action_bundles)
            ).astype(np.float32)
            prior_slice = (
                (1.0 - self.search_config.dirichlet_epsilon) * prior_slice
                + self.search_config.dirichlet_epsilon * noise
            )
            prior_slice = _normalize_policy(prior_slice)
            node.priors[: len(action_bundles)] = prior_slice

        limit = self.search_config.root_action_limit if node.depth == 0 else self.search_config.child_action_limit

        # Predict enemy action using fast heuristic instead of assuming hold
        enemy_player = 1 - node.player
        try:
            import sys
            import os
            # Add AI folder to path so we can import custom_utils
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            ai_path = os.path.join(repo_root, "AI")
            if ai_path not in sys.path:
                sys.path.insert(0, ai_path)
            from custom_utils import get_fast_heuristic_enemy_action
            enemy_action_name, enemy_ops = get_fast_heuristic_enemy_action(node.state, enemy_player)
            enemy_bundle = ActionBundle(name=enemy_action_name, score=0.0, tags=("predicted",), operations=enemy_ops)
        except ImportError:
            enemy_bundle = ActionBundle(name="hold", score=0.0, tags=("noop",))

        for action_index in self._branch_indices(node.priors, action_bundles, limit):
            child_state = node.state.clone()
            bundle = action_bundles[action_index]
            
            if node.player == 0:
                child_state.resolve_turn(bundle.operations, enemy_bundle.operations)
            else:
                child_state.resolve_turn(enemy_bundle.operations, bundle.operations)
                
            node.children.append(
                SearchNode(
                    state=child_state,
                    player=node.player,
                    prior=float(node.priors[action_index]),
                    bundle=bundle,
                    action_index=action_index,
                    depth=node.depth + 1,
                )
            )
        return inference.value

    def _puct(self, parent: SearchNode, child: SearchNode) -> float:
        explore = self.search_config.c_puct * child.prior * math.sqrt(parent.visits + 1.0) / (child.visits + 1.0)
        return child.mean_value + explore

    def _backpropagate(self, path: list[SearchNode], value: float) -> None:
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value

    def _sample_from_policy(self, policy: np.ndarray) -> int:
        threshold = self.rng.random()
        cumulative = 0.0
        for index, probability in enumerate(policy.tolist()):
            cumulative += probability
            if threshold <= cumulative:
                return index
        return int(np.argmax(policy))

    def _policy_from_visits(self, visits: np.ndarray, temperature: float) -> np.ndarray:
        if visits.size == 0:
            return visits.astype(np.float32, copy=False)
        if temperature <= 1e-6:
            policy = np.zeros_like(visits, dtype=np.float32)
            policy[int(np.argmax(visits))] = 1.0
            return policy
        scaled = np.power(np.maximum(visits, 1e-6), 1.0 / max(temperature, 1e-6)).astype(np.float32, copy=False)
        return _normalize_policy(scaled)

    def search(
        self,
        state: BackendState,
        player: int,
        bundles: list[ActionBundle] | None = None,
        temperature: float = 0.0,
        add_root_noise: bool = False,
    ) -> SearchResult:
        root = SearchNode(state=state.clone(), player=player)
        root_value = self._expand(root, bundles=bundles, add_root_noise=add_root_noise)
        if not root.bundles:
            fallback = ActionBundle(name="hold", score=0.0, tags=("noop",))
            return SearchResult(
                action_index=0,
                bundle=fallback,
                policy=np.zeros(self.action_dim, dtype=np.float32),
                root_value=float(root_value),
                visit_count=0,
                priors=np.zeros(self.action_dim, dtype=np.float32),
            )

        for _ in range(self.search_config.iterations):
            node = root
            path = [root]
            while node.expanded and node.children and node.depth < self.search_config.max_depth and not node.state.terminal:
                node = max(node.children, key=lambda child: self._puct(path[-1], child))
                path.append(node)
            if node.state.terminal or node.depth >= self.search_config.max_depth:
                value = self._heuristic_value(node.state, node.player)
            else:
                value = self._expand(node)
            self._backpropagate(path, value)

        visit_counts = np.zeros(len(root.bundles), dtype=np.float32)
        for child in root.children:
            visit_counts[child.action_index] = float(child.visits)
        if float(np.sum(visit_counts)) <= 0.0:
            visit_counts = root.priors[: len(root.bundles)] if root.priors is not None else np.ones(len(root.bundles), dtype=np.float32)

        root_policy = self._policy_from_visits(visit_counts, temperature=temperature)
        if temperature <= 1e-6:
            action_index = int(np.argmax(visit_counts))
        else:
            action_index = self._sample_from_policy(root_policy)
        selected_bundle = root.bundles[action_index]
        full_policy = np.zeros(self.action_dim, dtype=np.float32)
        full_policy[: len(root.bundles)] = root_policy
        full_priors = np.zeros(self.action_dim, dtype=np.float32)
        if root.priors is not None:
            full_priors[: len(root.bundles)] = root.priors[: len(root.bundles)]
        return SearchResult(
            action_index=action_index,
            bundle=selected_bundle,
            policy=full_policy,
            root_value=float(root.mean_value if root.visits else root_value),
            visit_count=int(visit_counts[action_index]),
            priors=full_priors,
        )


def build_policy_value_net(
    feature_extractor: FeatureExtractor,
    action_dim: int,
    config: PolicyValueNetConfig | None = None,
) -> PolicyValueNet:
    state = create_python_backend_state()
    mask = np.zeros(action_dim, dtype=np.float32)
    observation = feature_extractor.encode_observation(state, 0, mask)
    obs_dim = len(feature_extractor.flatten_observation(observation))
    return PolicyValueNet(obs_dim=obs_dim, action_dim=action_dim, config=config)


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
