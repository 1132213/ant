from __future__ import annotations

from dataclasses import dataclass, field
import math

try:
    from ai_greedy import GreedyHeuristicFallbackAgent
except ModuleNotFoundError as exc:
    if exc.name != "ai_greedy":
        raise
    from AI.ai_greedy import GreedyHeuristicFallbackAgent

try:
    from common import BaseAgent
except ModuleNotFoundError as exc:
    if exc.name != "common":
        raise
    from AI.common import BaseAgent

from SDK.actions import ActionBundle
from SDK.engine import GameState


@dataclass(slots=True)
class SearchNode:
    state: GameState
    player: int
    bundle: ActionBundle | None = None
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    depth: int = 0
    children: list[SearchNode] = field(default_factory=list)
    unexplored: list[ActionBundle] = field(default_factory=list)

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class MCTSAgent(BaseAgent):
    def __init__(self, iterations: int = 64, max_depth: int = 4, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.iterations = iterations
        self.max_depth = max_depth
        self.opponent_model = GreedyHeuristicFallbackAgent(seed=seed)

    def _expand(self, node: SearchNode) -> None:
        if node.unexplored:
            bundle = node.unexplored.pop(0)
            child_state = node.state.clone()
            enemy_bundle = self.opponent_model.choose_bundle(child_state, 1 - node.player)
            if node.player == 0:
                child_state.resolve_turn(bundle.operations, enemy_bundle.operations)
            else:
                child_state.resolve_turn(enemy_bundle.operations, bundle.operations)
            child = SearchNode(
                state=child_state,
                player=node.player,
                bundle=bundle,
                prior=max(bundle.score, 0.0) + 1.0,
                depth=node.depth + 1,
                unexplored=self.catalog.build(child_state, node.player)[:10],
            )
            node.children.append(child)

    def _uct(self, parent: SearchNode, child: SearchNode) -> float:
        exploration = 1.25 * child.prior * math.sqrt(parent.visits + 1) / (child.visits + 1)
        return child.mean_value + exploration

    def _rollout(self, state: GameState, player: int, depth: int) -> float:
        rollout = state.clone()
        for _ in range(depth, self.max_depth):
            if rollout.terminal:
                break
            my_bundle = self.opponent_model.choose_bundle(rollout, player)
            enemy_bundle = self.opponent_model.choose_bundle(rollout, 1 - player)
            if player == 0:
                rollout.resolve_turn(my_bundle.operations, enemy_bundle.operations)
            else:
                rollout.resolve_turn(enemy_bundle.operations, my_bundle.operations)
        return self.feature_extractor.evaluate(rollout, player)

    def _simulate(self, root: SearchNode) -> None:
        path = [root]
        node = root
        while node.children and node.depth < self.max_depth and not node.state.terminal:
            node = max(node.children, key=lambda child: self._uct(path[-1], child))
            path.append(node)
        if node.depth < self.max_depth and not node.state.terminal:
            self._expand(node)
            if node.children:
                node = node.children[-1]
                path.append(node)
        value = self._rollout(node.state, node.player, node.depth)
        for current in reversed(path):
            current.visits += 1
            current.value_sum += value

    def choose_bundle(self, state: GameState, player: int, bundles: list[ActionBundle] | None = None) -> ActionBundle:
        bundles = bundles or self.list_bundles(state, player)
        root = SearchNode(
            state=state.clone(),
            player=player,
            unexplored=bundles[: min(12, len(bundles))],
        )
        if not root.unexplored:
            return bundles[0]
        for _ in range(self.iterations):
            self._simulate(root)
        if not root.children:
            return root.unexplored[0]
        best = max(root.children, key=lambda child: (child.visits, child.mean_value))
        return best.bundle if best.bundle is not None else bundles[0]


class AI(MCTSAgent):
    pass
