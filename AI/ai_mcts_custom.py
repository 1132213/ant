from __future__ import annotations

import os
from typing import List
from pathlib import Path

try:
    from common import BaseAgent
except ModuleNotFoundError as exc:
    if exc.name != "common":
        raise
    from AI.common import BaseAgent

from SDK.alphazero import PolicyValueNet, PriorGuidedMCTS, SearchConfig
from SDK.utils.actions import ActionBundle
from SDK.utils.constants import MAX_ACTIONS, TowerType, SuperWeaponType
from SDK.backend.state import BackendState

# 导入我们写的启发式接口
from custom_utils import (
    get_affordable_strategic_slots,
    get_towers_can_upgrade,
    get_best_super_weapon_target,
    generate_build_operation,
    generate_upgrade_operation,
    generate_super_weapon_operation,
    generate_base_upgrade,
    get_enemy_frontline_distance,
    evaluate_frontline_status,
    get_frontline_strategic_slots
)

class CustomMCTSAgent(BaseAgent):
    """
    Phase 3: MCTS + 启发式剪枝
    利用 Phase 2 的规则筛选出高质量的动作，然后喂给 MCTS 进行未来推演。
    """
    def __init__(
        self,
        iterations: int = 1000,  # 提高迭代次数上限，依靠 time_limit 动态截断
        max_depth: int = 5,    # 提高推演深度，让 MCTS 能看到更远的未来
        seed: int | None = None,
        max_actions: int = MAX_ACTIONS,
        model_path: str | os.PathLike[str] | None = None,
        c_puct: float = 1.25,
        prior_mix: float = 0.0, 
        value_mix: float = 0.0, 
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
        env_path = os.getenv("AGENT_TRADITION_MCTS_MODEL")
        if env_path:
            candidates.append(Path(env_path))
        module_root = Path(__file__).resolve().parent
        repo_root = module_root.parent
        candidates.extend(
            [
                module_root / "ai_mcts_model.npz",
                repo_root / "checkpoints" / "ai_mcts_latest.npz",
                repo_root / "SDK" / "checkpoints" / "ai_mcts_latest.npz",
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
        """
        启发式动作裁剪：
        从成千上万个合法的 bundle 中，挑选出符合我们“均衡推进流”战术的若干个动作，
        大幅缩小 MCTS 的搜索宽度。并使用 O(N) 的哈希匹配解决预处理 TLE 问题。
        """
        pruned_bundles = []
        
        # O(N) 一次性构建映射表，极大提升动作匹配速度
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

        # 必须保留空操作（因为有可能我们要攒钱）
        noop_bundle = next((b for b in all_bundles if b.name == "hold"), all_bundles[0])
        pruned_bundles.append(noop_bundle)

        # 1. 危机处理候选动作：闪电风暴
        frontline_dist = get_enemy_frontline_distance(state, player)
        if frontline_dist <= 4:
            target = get_best_super_weapon_target(state, player, SuperWeaponType.LIGHTNING_STORM)
            if target:
                op = generate_super_weapon_operation(SuperWeaponType.LIGHTNING_STORM, target[0], target[1])
                add_bundle_by_op(op)

        # 2. 基地升级候选动作：如果有足够金币，优先考虑基地升级（提升长远胜率）
        from SDK.utils.constants import OperationType
        for op_type in [OperationType.UPGRADE_GENERATION_SPEED, OperationType.UPGRADE_GENERATED_ANT]:
            op = generate_base_upgrade(op_type)
            if state.can_apply_operation(player, op):
                add_bundle_by_op(op)

        # 3. 建塔候选动作：根据攻防状态保留更多最佳位置的建塔动作
        status = evaluate_frontline_status(state, player)
        strategic_slots = get_frontline_strategic_slots(state, player, status)
        
        valid_slots = []
        if state.coins[player] >= 30:  # BASIC 塔的花费是 30
            valid_slots = strategic_slots
            
        for i in range(min(5, len(valid_slots))):
            best_slot = valid_slots[i]
            op = generate_build_operation(best_slot[0], best_slot[1])
            add_bundle_by_op(op)

        # 4. 升级塔候选动作：如果有钱，提供升级到多个战术分支的动作
        if state.coins[player] >= 60:
            upgradable_towers = get_towers_can_upgrade(state, player)
            if upgradable_towers:
                # 挑选离前线最近的两个塔尝试升级
                for target_tower in upgradable_towers[:2]:
                    # 拓展升级选项：HEAVY(单体), QUICK(速射), MORTAR(AOE), PRODUCER(产兵), ICE(减速), HEAVY_PLUS等
                    # 具体能否升在 get_towers_can_upgrade 里做了粗筛，再靠 add_bundle_by_op(op) 匹配合法性
                    possible_targets = [
                        TowerType.HEAVY, TowerType.QUICK, TowerType.MORTAR, TowerType.PRODUCER,
                        TowerType.HEAVY_PLUS, TowerType.ICE, TowerType.MORTAR_PLUS, TowerType.QUICK_PLUS
                    ]
                    for target_type in possible_targets:
                        op = generate_upgrade_operation(target_tower.tower_id, target_type)
                        add_bundle_by_op(op)
                            
        # 去重并返回
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
        import time
        start_time = time.time()
        time_limit = 8.0  # 限制最多搜索 8 秒（原为9.0），给平台评测留出更充足的安全余量
        
        all_bundles = bundles or self.list_bundles(state, player)
        if not all_bundles:
            return ActionBundle(name="hold", score=0.0, tags=("noop",))
            
        # 1. 使用我们的启发式规则大幅裁剪动作空间
        pruned_bundles = self._get_pruned_bundles(state, player, all_bundles)
        
        # 如果只剩 1 个动作（比如 hold），直接返回，避免初始化 MCTS 消耗额外时间
        if len(pruned_bundles) == 1:
            return pruned_bundles[0]
            
        # 检查时间：如果预处理本身耗时太多（比如热力图初次计算），立即退回保守动作，防止第一回合直接 TLE
        if time.time() - start_time > time_limit:
            return pruned_bundles[0]
            
        # 动态计算可以做多少次迭代，利用 time_limit 控制
        # 借用官方 MCTS 的底层逻辑，但在外层自己实现一个简易的 Time-bound MCTS
        
        # 初始化根节点
        from SDK.alphazero import SearchNode, _terminal_value, _normalize_policy
        import numpy as np
        
        # 自定义启发式价值评估函数（取代官方仅根据胜负或简单 feature 提取的逻辑）
        def custom_heuristic_value(node_state: BackendState, node_player: int) -> float:
            terminal = _terminal_value(node_state, node_player)
            if terminal is not None:
                return float(terminal) # 严格返回 1.0 或 -1.0，防止 MCTS Q值爆炸
                
            enemy_player = 1 - node_player
            
            # 1. 基地血量差 (最核心，权重最高)
            hp_diff = node_state.bases[node_player].hp - node_state.bases[enemy_player].hp
            hp_score = hp_diff / 25.0  # 50血上限，差一半就占极大优势
            
            # 2. 科技与经济差
            my_tech = node_state.bases[node_player].generation_level + node_state.bases[node_player].ant_level
            enemy_tech = node_state.bases[enemy_player].generation_level + node_state.bases[enemy_player].ant_level
            tech_score = (my_tech - enemy_tech) * 0.3
            coin_score = (node_state.coins[node_player] - node_state.coins[enemy_player]) / 200.0
            
            # 3. 兵线差 (弱化单只蚂蚁的影响)
            my_danger = min(node_state.nearest_ant_distance(node_player), 30)
            enemy_danger = min(node_state.nearest_ant_distance(enemy_player), 30)
            line_score = (my_danger - enemy_danger) / 30.0 
            
            # 4. 防御塔质量与前压 (乘法加成)
            def eval_towers(player_id):
                score = 0.0
                for t in node_state.towers_of(player_id):
                    # 归一化前线推进度 0~1 (地图大小为 28)
                    progress = t.x / 28.0 if player_id == 0 else (28.0 - t.x) / 28.0
                    # 基础分 * (1 + 前压红利)
                    score += ((t.level + 1) * 3.0) * (1.0 + progress)
                return score
                
            tower_score = (eval_towers(node_player) - eval_towers(enemy_player)) / 20.0
            
            # 汇总进入 tanh
            total_score = hp_score * 1.5 + tech_score + coin_score + line_score * 0.5 + tower_score
            return float(np.tanh(total_score))

        # 将官方的 heuristic_value 替换为我们的 custom_heuristic_value
        self.search._heuristic_value = custom_heuristic_value
        
        # 为了避免我们手动构造 child 时遗漏官方 MCTS 内部需要的其他属性导致报错，
        # 且为了极大地加速 MCTS 内层的 _expand 操作，我们禁用 ActionCatalog 的耗时操作
        # 1. 禁用 _rerank_with_one_step_rollout (节省 1-step 模拟的时间)
        self.catalog._rerank_with_one_step_rollout = lambda s, p, b: b
        
        # 2. 禁用预测敌方动作的耗时搜索，假设敌方在 MCTS 推演中只做保守操作或 hold
        # 这能将搜索树的分支复杂度和耗时再降低一倍以上
        self.search._predict_enemy_bundle = lambda s, p: ActionBundle(name="hold", score=0.0, tags=("noop",))
        
        root = SearchNode(state=state.clone(), player=player)
        root_value = self.search._expand(root, bundles=pruned_bundles, add_root_noise=False)
        
        if not root.bundles:
            return ActionBundle(name="hold", score=0.0, tags=("noop",))
            
        # Time-bound MCTS 迭代
        iterations_done = 0
        while time.time() - start_time < time_limit and iterations_done < self.search.search_config.iterations:
            node = root
            path = [root]
            # Selection
            while node.expanded and node.children and node.depth < self.search.search_config.max_depth and not node.state.terminal:
                node = max(node.children, key=lambda child: self.search._puct(path[-1], child))
                path.append(node)
            # Evaluation / Expansion
            if node.state.terminal or node.depth >= self.search.search_config.max_depth:
                value = self.search._heuristic_value(node.state, node.player)
            else:
                value = self.search._expand(node, bundles=None, add_root_noise=False)
            # Backpropagation
            self.search._backpropagate(path, value)
            iterations_done += 1
            
        # 根据访问次数选择最佳动作
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

class AI(CustomMCTSAgent):
    pass