from __future__ import annotations

import os
from typing import List
from pathlib import Path
import time
import numpy as np

try:
    from common import BaseAgent
except ModuleNotFoundError as exc:
    if exc.name != "common":
        raise
    from AI.common import BaseAgent

from SDK.alphazero import PolicyValueNet, PriorGuidedMCTS, SearchConfig, SearchNode, _terminal_value
from SDK.utils.actions import ActionBundle
from SDK.utils.constants import MAX_ACTIONS, TowerType, SuperWeaponType
from SDK.backend.state import BackendState

# 导入启发式接口
from custom_utils import (
    get_best_super_weapon_target,
    generate_build_operation,
    generate_upgrade_operation,
    generate_downgrade_operation,
    generate_super_weapon_operation,
    generate_base_upgrade,
    get_enemy_frontline_distance,
    evaluate_frontline_status,
    get_frontline_strategic_slots,
    get_towers_can_upgrade,
    get_fast_heuristic_enemy_action
)

class AlphaZeroAgent(BaseAgent):
    """
    AlphaZero AI: 
    使用预训练的神经网络评估局面，并结合启发式动作剪枝。
    这是混合模型与深度强化学习的结合产物，极大提高了搜索效率与容错率。
    """
    def __init__(
        self,
        iterations: int = 20000,  # 极大提高迭代上限，让它在评测时只受时间约束 (Time-Bound)
        max_depth: int = 10,      # 增加搜索深度，让它能看透更多回合
        seed: int | None = None,
        max_actions: int = MAX_ACTIONS,
        model_path: str | os.PathLike[str] | None = None,
        c_puct: float = 1.25,
        prior_mix: float = 0.5, # 开启神经网络策略先验
        value_mix: float = 0.5, # 开启神经网络价值评估
    ) -> None:
        super().__init__(seed=seed, max_actions=max_actions)
        self.model = self._load_model(model_path)
        self.search = PriorGuidedMCTS(
            model=self.model,
            search_config=SearchConfig(
                iterations=iterations,
                max_depth=max_depth,
                c_puct=c_puct,
                prior_mix=prior_mix,
                value_mix=value_mix,
                seed=seed or 0,
            ),
            feature_extractor=self.feature_extractor,
            action_catalog=self.catalog,
        )

    def _candidate_model_paths(self, override: str | os.PathLike[str] | None) -> list[Path]:
        candidates: list[Path] = []
        if override is not None:
            candidates.append(Path(override))
            return candidates
        module_root = Path(__file__).resolve().parent
        repo_root = module_root.parent
        candidates.extend(
            [
                module_root / "ai_alphazero_latest.npz",
                repo_root / "checkpoints" / "ai_alphazero_latest.npz",
            ]
        )
        return candidates

    def _load_model(self, model_path: str | os.PathLike[str] | None) -> PolicyValueNet | None:
        for candidate in self._candidate_model_paths(model_path):
            if not candidate.exists():
                continue
            try:
                model = PolicyValueNet.from_checkpoint(candidate)
            except (OSError, ValueError, KeyError):
                continue
            if model.action_dim != self.catalog.max_actions:
                continue
            return model
        return None

    def _get_pruned_bundles(self, state: BackendState, player: int, all_bundles: List[ActionBundle]) -> List[ActionBundle]:
        pruned_bundles = []
        op_to_bundle = {}
        for b in all_bundles:
            for o in b.operations:
                key = (o.op_type, o.arg0, o.arg1)
                if key not in op_to_bundle:
                    op_to_bundle[key] = b
                    
        def add_bundle_by_op(op):
            key = (op.op_type, op.arg0, op.arg1)
            if key in op_to_bundle:
                pruned_bundles.append(op_to_bundle[key])

        noop_bundle = next((b for b in all_bundles if b.name == "hold"), all_bundles[0])
        pruned_bundles.append(noop_bundle)

        # 尝试闪电风暴（基于热力图，不局限于守家）
        storm_target = get_best_super_weapon_target(state, player, SuperWeaponType.LIGHTNING_STORM)
        if storm_target:
            op = generate_super_weapon_operation(SuperWeaponType.LIGHTNING_STORM, storm_target[0], storm_target[1])
            add_bundle_by_op(op)
        
        # 尝试 EMP 轰炸（瘫痪敌方塔阵，主要用于进攻）
        emp_target = get_best_super_weapon_target(state, player, SuperWeaponType.EMP_BLASTER)
        if emp_target:
            op = generate_super_weapon_operation(SuperWeaponType.EMP_BLASTER, emp_target[0], emp_target[1])
            add_bundle_by_op(op)
            
        # 尝试引力护盾 / 紧急回避（保护我方冲锋兵线）
        if state.coins[player] >= 100:
            deflector_target = get_best_super_weapon_target(state, player, SuperWeaponType.DEFLECTOR)
            if deflector_target:
                op = generate_super_weapon_operation(SuperWeaponType.DEFLECTOR, deflector_target[0], deflector_target[1])
                add_bundle_by_op(op)
                
            evasion_target = get_best_super_weapon_target(state, player, SuperWeaponType.EMERGENCY_EVASION)
            if evasion_target:
                op = generate_super_weapon_operation(SuperWeaponType.EMERGENCY_EVASION, evasion_target[0], evasion_target[1])
                add_bundle_by_op(op)

        from SDK.utils.constants import OperationType
        for op_type in [OperationType.UPGRADE_GENERATION_SPEED, OperationType.UPGRADE_GENERATED_ANT]:
            op = generate_base_upgrade(op_type)
            if state.can_apply_operation(player, op):
                add_bundle_by_op(op)

        status = evaluate_frontline_status(state, player)
        strategic_slots = get_frontline_strategic_slots(state, player, status)
        
        for i in range(min(5, len(strategic_slots))):
            best_slot = strategic_slots[i]
            op = generate_build_operation(best_slot[0], best_slot[1])
            add_bundle_by_op(op)

        if state.coins[player] >= 60:
            upgradable_towers = get_towers_can_upgrade(state, player)
            if upgradable_towers:
                for target_tower in upgradable_towers[:2]:
                    possible_targets = [
                        TowerType.HEAVY, TowerType.QUICK, TowerType.MORTAR, TowerType.PRODUCER,
                        TowerType.HEAVY_PLUS, TowerType.ICE, TowerType.BEWITCH,
                        TowerType.QUICK_PLUS, TowerType.DOUBLE, TowerType.SNIPER,
                        TowerType.MORTAR_PLUS, TowerType.PULSE, TowerType.MISSILE,
                        TowerType.PRODUCER_FAST, TowerType.PRODUCER_SIEGE, TowerType.PRODUCER_MEDIC
                    ]
                    for target_type in possible_targets:
                        op = generate_upgrade_operation(target_tower.tower_id, target_type)
                        add_bundle_by_op(op)

            if state.coins[player] < 100:
                for t in state.towers_of(player):
                    if t.x < 10 and player == 0 and status != "DEFEND":
                        op = generate_downgrade_operation(t.tower_id)
                        if state.can_apply_operation(player, op):
                            add_bundle_by_op(op)
                    elif t.x > 20 and player == 1 and status != "DEFEND":
                        op = generate_downgrade_operation(t.tower_id)
                        if state.can_apply_operation(player, op):
                            add_bundle_by_op(op)
                            
        seen = set()
        unique_bundles = []
        for b in pruned_bundles:
            if b.name not in seen:
                seen.add(b.name)
                unique_bundles.append(b)
                
        return unique_bundles

    def choose_bundle(
        self,
        state: BackendState,
        player: int,
        bundles: list[ActionBundle] | None = None,
    ) -> ActionBundle:
        start_time = time.time()
        time_limit = 8.0 
        
        all_bundles = bundles or self.list_bundles(state, player)
        if not all_bundles:
            return ActionBundle(name="hold", score=0.0, tags=("noop",))
            
        pruned_bundles = self._get_pruned_bundles(state, player, all_bundles)
        
        if len(pruned_bundles) == 1 or time.time() - start_time > time_limit:
            return pruned_bundles[0]
            
        # 自定义启发式价值评估函数
        def custom_heuristic_value(node_state: BackendState, node_player: int) -> float:
            terminal = _terminal_value(node_state, node_player)
            if terminal is not None:
                return float(terminal) # 严格返回 1.0 或 -1.0，防止 MCTS Q值爆炸
                
            enemy_player = 1 - node_player
            
            my_danger = node_state.nearest_ant_distance(node_player)
            enemy_danger = node_state.nearest_ant_distance(enemy_player)
            if my_danger == float('inf'): my_danger = 50.0
            if enemy_danger == float('inf'): enemy_danger = 50.0
            
            line_score = (my_danger - enemy_danger) / 20.0
            coin_score = (node_state.coins[node_player] - node_state.coins[enemy_player]) / 300.0
            
            my_towers = node_state.towers_of(node_player)
            enemy_towers = node_state.towers_of(enemy_player)
            
            def eval_tower(t):
                front_bonus = t.x / 10.0 if node_player == 0 else (30 - t.x) / 10.0
                hp_ratio = t.hp / float(t.max_hp if hasattr(t, 'max_hp') else 15.0)
                return ((t.level + 1) * 3.0 + front_bonus) * (0.5 + 0.5 * hp_ratio)
                
            my_tower_score = sum(eval_tower(t) for t in my_towers)
            enemy_tower_score = sum(eval_tower(t) for t in enemy_towers)
            tower_score = (my_tower_score - enemy_tower_score) / 15.0
            
            total_score = line_score + coin_score + tower_score
            
            if not hasattr(custom_heuristic_value, "_score_history"):
                custom_heuristic_value._score_history = []
            custom_heuristic_value._score_history.append(abs(total_score))
            if len(custom_heuristic_value._score_history) > 1000:
                custom_heuristic_value._score_history.pop(0)
                
            dynamic_scale = 3.0
            if len(custom_heuristic_value._score_history) > 100:
                dynamic_scale = max(0.5, float(np.percentile(custom_heuristic_value._score_history, 90)))
                
            return float(np.tanh(total_score / dynamic_scale)) * 0.95

        # 结合神经网络进行展开（覆写 _expand 的行为以支持 pruning）
        def custom_expand(node: SearchNode, bundles: list[ActionBundle] | None = None, add_root_noise: bool = False) -> float:
            if node.expanded:
                return node.mean_value
            terminal = _terminal_value(node.state, node.player)
            if terminal is not None:
                node.expanded = True
                return terminal
                
            # 使用传入的剪枝动作，或是对当前状态重新剪枝
            action_bundles = bundles or self._get_pruned_bundles(node.state, node.player, self.catalog.build(node.state, node.player))
            node.bundles = action_bundles
            
            # 使用网络获取 Priors 与 Value
            inference = self.search._blend_policy_value(node.state, node.player, action_bundles)
            node.priors = inference.priors
            node.expanded = True
            
            if node.depth >= self.search.search_config.max_depth or not action_bundles:
                return inference.value
                
            enemy_player = 1 - node.player
            enemy_action_name, enemy_ops = get_fast_heuristic_enemy_action(node.state, enemy_player)
            enemy_bundle = ActionBundle(name=enemy_action_name, score=0.0, tags=("predicted",), operations=enemy_ops)
                
            limit = self.search.search_config.root_action_limit if node.depth == 0 else self.search.search_config.child_action_limit
            for action_index in self.search._branch_indices(node.priors, action_bundles, limit):
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

        # 设置自定义方法
        self.search._heuristic_value = custom_heuristic_value
        self.search._expand = custom_expand
        self.catalog._rerank_with_one_step_rollout = lambda s, p, b: b
        
        root = SearchNode(state=state.clone(), player=player)
        # 第一层展开，传入剪枝后的动作
        root_value = self.search._expand(root, bundles=pruned_bundles, add_root_noise=False)
        
        if not root.bundles:
            return ActionBundle(name="hold", score=0.0, tags=("noop",))
            
        iterations_done = 0
        while time.time() - start_time < time_limit and iterations_done < self.search.search_config.iterations:
            node = root
            path = [root]
            while node.expanded and node.children and node.depth < self.search.search_config.max_depth and not node.state.terminal:
                node = max(node.children, key=lambda child: self.search._puct(path[-1], child))
                path.append(node)
            if node.state.terminal or node.depth >= self.search.search_config.max_depth:
                value = self.search._heuristic_value(node.state, node.player)
            else:
                value = self.search._expand(node, bundles=None, add_root_noise=False)
            self.search._backpropagate(path, value)
            iterations_done += 1
            
        visit_counts = np.zeros(len(root.bundles), dtype=np.float32)
        for child in root.children:
            visit_counts[child.action_index] = float(child.visits)
            
        import sys
        print(f"[Turn {getattr(state, 'current_turn', getattr(state, 'turn', 0))}] Time: {time.time()-start_time:.3f}s, Iterations: {iterations_done}, Candidates: {len(pruned_bundles)}", file=sys.stderr)
        
        if float(np.sum(visit_counts)) <= 0.0:
            action_index = 0
        else:
            action_index = int(np.argmax(visit_counts))
            
        return root.bundles[action_index]

class AI(AlphaZeroAgent):
    pass
