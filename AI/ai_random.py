from __future__ import annotations

try:
    from common import BaseAgent
except ImportError:
    from AI.common import BaseAgent

from SDK.actions import ActionBundle
from SDK.engine import GameState


class RandomAgent(BaseAgent):
    def choose_bundle(self, state: GameState, player: int, bundles: list[ActionBundle] | None = None) -> ActionBundle:
        bundles = bundles or self.list_bundles(state, player)
        if len(bundles) <= 1:
            return bundles[0]
        # Bias very slightly away from the mandatory no-op so the agent explores the action space.
        pool = bundles[1:] if len(bundles) > 1 else bundles
        return self.rng.choice(pool)


class AI(RandomAgent):
    pass
