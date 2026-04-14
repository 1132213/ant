from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SDK.training.alphazero import AlphaZeroSelfPlayTrainer, AlphaZeroTrainerConfig
from SDK.training.env import AntWarParallelEnv
from SDK.training.logging_utils import TrainingLogger
from SDK.alphazero import PriorGuidedMCTS, SearchNode, _terminal_value, ActionBundle
from SDK.backend.state import BackendState

# 导入用户在混合模型中实现的启发式工具
try:
    from AI.custom_utils import (
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
    )
    from SDK.utils.constants import OperationType, TowerType, SuperWeaponType
except ImportError:
    pass

class CustomPriorGuidedMCTS(PriorGuidedMCTS):
    """
    注入启发式剪枝和价值函数的 MCTS，用于加速 Self-Play 和减小动作空间。
    """
    def _get_pruned_bundles(self, state: BackendState, player: int, all_bundles: list[ActionBundle]) -> list[ActionBundle]:
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

        # 必须保留空操作
        noop_bundle = next((b for b in all_bundles if b.name == "hold"), all_bundles[0])
        pruned_bundles.append(noop_bundle)

        try:
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

            # 允许降级/拆除后方没用的塔来回血/刷钱
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

        except NameError:
            # 如果 custom_utils 导入失败，回退到全部动作
            return all_bundles

        seen = set()
        unique_bundles = []
        for b in pruned_bundles:
            if b.name not in seen:
                seen.add(b.name)
                unique_bundles.append(b)
                
        return unique_bundles

    def _heuristic_value(self, state: BackendState, player: int) -> float:
        terminal = _terminal_value(state, player)
        if terminal is not None:
            return terminal
        
        # 融入混合模型中的稠密奖励函数，提供更平滑的学习信号
        my_danger = state.nearest_ant_distance(player)
        enemy_danger = state.nearest_ant_distance(1 - player)
        if my_danger == float('inf'): my_danger = 50.0
        if enemy_danger == float('inf'): enemy_danger = 50.0
        
        line_score = (my_danger - enemy_danger) / 20.0
        # 因为击杀战斗蚂蚁赏金提高到 18，金币容易通胀，调整分母为 300
        coin_score = (state.coins[player] - state.coins[1 - player]) / 300.0
        
        my_towers = state.towers_of(player)
        enemy_towers = state.towers_of(1 - player)
        
        def eval_tower(t):
            front_bonus = t.x / 10.0 if player == 0 else (30 - t.x) / 10.0
            # 加入残血考量，闪电风暴有细碎伤害，满血和残血的塔价值不同
            hp_ratio = t.hp / float(t.max_hp if hasattr(t, 'max_hp') else 15.0)
            return ((t.level + 1) * 3.0 + front_bonus) * (0.5 + 0.5 * hp_ratio)
            
        my_tower_score = sum(eval_tower(t) for t in my_towers)
        enemy_tower_score = sum(eval_tower(t) for t in enemy_towers)
        tower_score = (my_tower_score - enemy_tower_score) / 15.0
        
        total_score = line_score + coin_score + tower_score
        
        # 记录近期的 total_score 用于动态缩放计算
        if not hasattr(self, "_score_history"):
            self._score_history = []
        self._score_history.append(abs(total_score))
        if len(self._score_history) > 1000:
            self._score_history.pop(0)
            
        # 动态计算缩放因子（使用 90% 分位数，避免极端值影响）
        # 如果样本太少，默认使用 3.0 作为保守初始值
        dynamic_scale = 3.0
        if len(self._score_history) > 100:
            dynamic_scale = max(0.5, float(np.percentile(self._score_history, 90)))
            
        # 调试用：每隔一定概率打印一次 total_score 的分布，检查是否过早饱和
        # import random
        # if random.random() < 0.001:  # 大约每 1000 次调用打印一次，避免刷屏
        #     print(f"[Heuristic Value] line: {line_score:.2f}, coin: {coin_score:.2f}, tower: {tower_score:.2f} | total: {total_score:.2f} | tanh(total/{dynamic_scale:.2f}): {float(np.tanh(total_score / dynamic_scale)):.2f}")
        
        # 将总分压缩到 (-0.95, 0.95) 的区间内，留出绝对的空间给真实的胜负信号 (+1.0 / -1.0)
        return float(np.tanh(total_score / dynamic_scale)) * 0.95

    def _expand(self, node: SearchNode, bundles: list[ActionBundle] | None = None, add_root_noise: bool = False) -> float:
        if node.expanded:
            return node.mean_value

        terminal = _terminal_value(node.state, node.player)
        if terminal is not None:
            node.expanded = True
            return terminal

        action_bundles = bundles or self.action_catalog.build(node.state, node.player)
        # 在扩展节点时进行动作剪枝
        action_bundles = self._get_pruned_bundles(node.state, node.player, action_bundles)
        
        node.bundles = action_bundles
            
        # 拦截：如果是纯启发式，直接跳过 GPU 推理
        if getattr(self.search_config, "prior_mix", 0.5) >= 1.0 and getattr(self.search_config, "value_mix", 0.5) >= 1.0:
            from SDK.alphazero import PolicyValueInference, _heuristic_bundle_policy
            heur_priors = _heuristic_bundle_policy(action_bundles)
            heur_val = self._heuristic_value(node.state, node.player)
            # 由于没有真实的网络观测和 action_mask，这里使用伪造的 dummy 数组
            # 这个 dummy 不会影响当前 MCTS 搜索，因为纯启发式不需要训练这些 observation
            action_mask = np.ones(len(action_bundles), dtype=np.float32)
            flat_obs = np.zeros(1, dtype=np.float32)
            full_priors = np.zeros(self.action_catalog.max_actions, dtype=np.float32)
            full_priors[: len(action_bundles)] = heur_priors
            inference = PolicyValueInference(priors=full_priors, value=heur_val, observation=flat_obs, mask=action_mask)
        else:
            inference = self._blend_policy_value(node.state, node.player, action_bundles)
        
        node.priors = inference.priors
        node.expanded = True

        if node.depth >= self.search_config.max_depth or not action_bundles:
            return inference.value

        prior_slice = inference.priors[: len(action_bundles)].astype(np.float32, copy=True)
        if add_root_noise and len(action_bundles) > 1 and self.search_config.dirichlet_epsilon > 0.0:
            from SDK.alphazero import _normalize_policy
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
        for action_index in self._branch_indices(node.priors, action_bundles, limit):
            child_state = node.state.clone()
            bundle = action_bundles[action_index]
            # 加速：使用快速启发式预测敌方动作，取代默认的 hold
            enemy_player = 1 - node.player
            try:
                from AI.custom_utils import get_fast_heuristic_enemy_action
                enemy_action_name, enemy_ops = get_fast_heuristic_enemy_action(node.state, enemy_player)
                enemy_bundle = ActionBundle(name=enemy_action_name, score=0.0, tags=("predicted",), operations=enemy_ops)
            except ImportError:
                enemy_bundle = ActionBundle(name="hold", score=0.0, tags=("noop",))
                
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


import multiprocessing
import os

# 将单局收集逻辑提取为顶层函数，避免传参时 Pickle 序列化主进程庞大对象的报错
def _collect_episode_worker(
    seed: int,
    batch_index: int,
    config_dict: dict,
    checkpoint_path: str,
    worker_id: int,
    prefer_native_backend: bool,
) -> tuple[SelfPlayBatch, EpisodeSummary]:
    import torch
    
    try:
        import setproctitle
        setproctitle.setproctitle("python")
    except ImportError:
        pass

    from SDK.alphazero import PolicyValueNet
    from SDK.training.env import AntWarParallelEnv
    from SDK.training.alphazero import AlphaZeroTrainerConfig
    
    # 1. 多卡负载均衡：获取当前进程分配的 GPU
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        gpu_id = worker_id % num_gpus
        torch.cuda.set_device(gpu_id)
        
    # 2. 重建配置和环境
    config = AlphaZeroTrainerConfig(**config_dict)
    env = AntWarParallelEnv(seed=seed, max_actions=config.max_actions, prefer_native_backend=prefer_native_backend)
    
    # 3. 独立加载模型副本（避免主进程污染）
    if os.path.exists(checkpoint_path):
        model = PolicyValueNet.from_checkpoint(checkpoint_path)
    else:
        # 如果是第一个 batch，模型还未保存，回退到启发式或者随机初始化
        model = None
        
    # 4. 初始化 MCTS 搜索器
    from SDK.utils.features import FeatureExtractor
    from SDK.utils.actions import ActionCatalog
    from SDK.alphazero import SearchConfig
    feature_extractor = FeatureExtractor(max_actions=config.max_actions)
    action_catalog = ActionCatalog(max_actions=config.max_actions, feature_extractor=feature_extractor)
    action_catalog._rerank_with_one_step_rollout = lambda s, p, b: b # 禁用耗时操作
    
    search_config = SearchConfig(
        iterations=config.search_iterations,
        max_depth=config.max_depth,
        c_puct=config.c_puct,
        root_action_limit=config.root_action_limit,
        child_action_limit=config.child_action_limit,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_epsilon=config.dirichlet_epsilon,
        prior_mix=config.prior_mix,
        value_mix=config.value_mix,
        value_scale=config.value_scale,
        seed=config.seed,
    )
    
    search = CustomPriorGuidedMCTS(
        model=model,
        search_config=search_config,
        feature_extractor=feature_extractor,
        action_catalog=action_catalog,
    )
    search.current_batch = batch_index
    
    # 5. 实例化临时 Trainer 只负责这一局的执行逻辑
    # 因为原有的 collect_episode 需要引用 _temperature_for_round, _value_target 等
    temp_trainer = CustomAlphaZeroSelfPlayTrainer(
        env_factory=lambda seed=0: env,
        config=config,
        logger=None
    )
    temp_trainer.model = model
    temp_trainer.search = search
    temp_trainer.feature_extractor = feature_extractor
    temp_trainer.action_catalog = action_catalog
    
    try:
        batch, summary = temp_trainer.collect_episode(seed=seed)
        return batch, summary
    finally:
        env.close()

class CustomAlphaZeroSelfPlayTrainer(AlphaZeroSelfPlayTrainer):
    def __init__(self, env_factory, config, logger):
        super().__init__(env_factory, config, logger)
        # 替换 MCTS 实例为我们自定义的加速并剪枝的版本
        
        # 禁用 ActionCatalog 的耗时操作
        self.action_catalog._rerank_with_one_step_rollout = lambda s, p, b: b

        self.search = CustomPriorGuidedMCTS(
            model=self.model,
            search_config=self._build_search_config(exploration=True),
            feature_extractor=self.feature_extractor,
            action_catalog=self.action_catalog,
        )
        self.eval_search = CustomPriorGuidedMCTS(
            model=self.model,
            search_config=self._build_search_config(exploration=False),
            feature_extractor=self.feature_extractor,
            action_catalog=self.action_catalog,
        )
        self.heuristic_search = CustomPriorGuidedMCTS(
            model=None,
            search_config=self._build_search_config(exploration=False),
            feature_extractor=self.feature_extractor,
            action_catalog=self.action_catalog,
        )

    def _value_target(self, env: AntWarParallelEnv, player: int) -> float:
        # 如果游戏真实分出了胜负，直接返回真实胜负作为最强信号 (+1.0 或 -1.0)
        if env.state.terminal:
            if env.state.winner is None:
                return 0.0
            return 1.0 if env.state.winner == player else -1.0
            
        # 如果是平局截断或过程中评估，使用受约束的稠密奖励 (-0.95 ~ 0.95)
        return self.search._heuristic_value(env.state, player)

    def train(self, num_batches: int | None = None) -> tuple[list[dict[str, float]], list[EpisodeSummary]]:
        import concurrent.futures
        from dataclasses import asdict
        
        updates = num_batches if num_batches is not None else self.config.batches
        history: list[dict[str, float]] = []
        samples: list[EpisodeSummary] = []
        
        # Adaptive scheduling states
        policy_loss_history = []
        value_loss_history = []
        target_value_history = []
        lr_cooldown = 0
        current_lr = self.config.learning_rate
        current_prior_mix = self.config.prior_mix
        current_value_mix = self.config.value_mix
        
        # Hard limits for adaptive parameters
        MIN_LR = 1e-5
        MAX_PRIOR_MIX = 0.95
        MIN_PRIOR_MIX = 0.5
        MAX_VALUE_MIX = 0.95
        MIN_VALUE_MIX = 0.5
        
        # 在启动多进程之前，强制先保存一次初始化的模型（供子进程加载）
        initial_checkpoint = self.save_checkpoint()
        
        # 将 Config 序列化为字典供进程间传递
        config_dict = asdict(self.config)
        
        # 准备参数获取原生后端的偏好设置
        prefer_native = getattr(self.env_factory(seed=0), "backend").name != "PythonEngineBackend"
        
        for i in range(updates):
            # 支持通过 --start-batch 控制训练起点，以接上之前的 Curriculum 混合比例
            batch_index = getattr(self.config, "start_batch", 0) + i
            episode_batches = []
            episode_summaries = []
            
            # Ensure the config sent to workers has the latest adaptive mixes
            config_dict["prior_mix"] = current_prior_mix
            config_dict["value_mix"] = current_value_mix
            
            # 使用 ProcessPoolExecutor (多进程) 替代 ThreadPoolExecutor
            max_workers = getattr(self.config, "workers", 4)
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for episode_offset in range(self.config.episodes):
                    seed = self.config.seed + batch_index * 1_000 + episode_offset
                    # 将收集任务分发给独立子进程，传递序列化后的参数
                    futures.append(
                        executor.submit(
                            _collect_episode_worker,
                            seed=seed,
                            batch_index=batch_index,
                            config_dict=config_dict,
                            checkpoint_path=self.config.checkpoint_path,
                            worker_id=episode_offset,
                            prefer_native_backend=prefer_native
                        )
                    )
                    
                for future in concurrent.futures.as_completed(futures):
                    try:
                        batch, summary = future.result()
                        episode_batches.append(batch)
                        episode_summaries.append(summary)
                        if self.logger is not None:
                            self.logger.log_episode(batch_index=batch_index, episode_index=summary.seed % 1000, payload=asdict(summary))
                    except Exception as exc:
                        print(f"Episode collection failed in subprocess: {exc}")
                        raise
                        
            merged = self._merge_batches(episode_batches)
            
            # 主进程负责汇总数据并更新（单卡快速更新即可）
            # 动态更新 config 中的 lr 以让 update_from_batch 使用最新的 lr
            self.config.learning_rate = current_lr
            metrics = self.update_from_batch(merged)
            # 使用真实对战的 Value 作为 mean_target_value 并记录
            metrics["mean_target_value"] = float(np.mean(merged.values))
            
            # ---------------------------------------------------------
            # 自适应调度 (Adaptive Scheduling)
            # ---------------------------------------------------------
            policy_loss = metrics.get("policy_loss", 0.0)
            value_loss = metrics.get("value_loss", 0.0)
            mean_target = metrics.get("mean_target_value", 0.0)
            
            policy_loss_history.append(policy_loss)
            value_loss_history.append(value_loss)
            target_value_history.append(mean_target)
            
            # 保持窗口大小为 5
            if len(policy_loss_history) > 5:
                policy_loss_history.pop(0)
                value_loss_history.pop(0)
                target_value_history.pop(0)
                
            # 冷却计数器
            if lr_cooldown > 0:
                lr_cooldown -= 1
                
            # 当窗口填满时，开始判断自适应规则
            if len(policy_loss_history) == 5:
                # 1. LR 衰减：检测 Policy Loss 平台期
                recent_policy_avg = np.mean(policy_loss_history)
                policy_variance = np.std(policy_loss_history)
                # 如果方差极小（平台期）且没在冷却期
                if policy_variance < 0.01 and lr_cooldown == 0:
                    current_lr = max(MIN_LR, current_lr * 0.5)
                    lr_cooldown = 10 # 冷却 10 个 Batch
                    print(f"[*] LR Plateau detected! Decay LR to {current_lr}")
                    
                # 2. Prior Mix 自适应：基于 Value Loss 的稳定性
                max_recent_value_loss = np.max(value_loss_history)
                # 如果连续 5 个 Batch 的 Value Loss 都很健康，提升信任度
                if max_recent_value_loss < 0.6:
                    current_prior_mix = min(MAX_PRIOR_MIX, current_prior_mix + 0.05)
                # 如果 Value Loss 突然飙升（模型遇到盲区，胜率崩溃），立即回退先验，让启发式接管
                elif value_loss > 1.0:
                    current_prior_mix = max(MIN_PRIOR_MIX, current_prior_mix - 0.1)
                    print(f"[*] Value Loss Spike! Fallback prior_mix to {current_prior_mix:.2f}")
                    
                # 3. Value Mix 自适应：基于自我博弈的局势胶着度 (Mean Target Value)
                recent_target_avg = np.mean(np.abs(target_value_history))
                # 如果 Value Loss 健康，且自我博弈很胶着（平均 target 接近 0），提升对价值评估的信任
                if max_recent_value_loss < 0.6 and recent_target_avg < 0.4:
                    current_value_mix = min(MAX_VALUE_MIX, current_value_mix + 0.05)
                # 如果一边倒（可能是一方碾压），降低信任，多用启发式稳定局面
                elif recent_target_avg > 0.8:
                    current_value_mix = max(MIN_VALUE_MIX, current_value_mix - 0.1)
            
            # 更新 metrics 以供记录
            metrics["batch"] = float(batch_index)
            metrics["episodes"] = float(self.config.episodes)
            metrics["checkpoint_saved"] = 1.0
            metrics["lr"] = float(current_lr)
            metrics["prior_mix"] = float(current_prior_mix)
            metrics["value_mix"] = float(current_value_mix)
            
            checkpoint_path = self.save_checkpoint(batch_index=batch_index)
            metrics.update(self.evaluate_against_heuristic())
            history.append(metrics)
            samples.extend(episode_summaries)
            
            # --- 增加控制台日志输出 ---
            print(f"[Batch {batch_index:03d}] P-Loss: {policy_loss:.4f} | V-Loss: {value_loss:.4f} | Target: {mean_target:.4f} | LR: {current_lr:.2e} | P-Mix: {current_prior_mix:.2f} | V-Mix: {current_value_mix:.2f}")
            
            # 聚合动作统计并输出
            total_actions = 0
            aggregated_action_stats = {}
            for summary in episode_summaries:
                if summary.action_stats:
                    for action_name, count in summary.action_stats.items():
                        aggregated_action_stats[action_name] = aggregated_action_stats.get(action_name, 0) + count
                        total_actions += count
            
            if total_actions > 0:
                # 对动作出现次数降序排序
                sorted_actions = sorted(aggregated_action_stats.items(), key=lambda item: item[1], reverse=True)
                stats_str_list = [f"{name}({count/total_actions*100:.1f}%)" for name, count in sorted_actions[:10]] # 只打印前 10 高频动作
                action_stats_str = ', '.join(stats_str_list)
                print(f"  └─ Action Stats: {action_stats_str}")
                # 把字符串放入 metrics，交给 logger 记录到 train.log
                metrics["action_stats_str"] = action_stats_str
                # 也可以把前五动作记入 metrics 供 Tensorboard / json logger 记录
                for idx, (name, count) in enumerate(sorted_actions[:5]):
                    metrics[f"top_{idx+1}_action_ratio"] = count / total_actions
            
            if self.logger is not None:
                self.logger.log_batch_metrics(batch_index=batch_index, payload=metrics)
                self.logger.log_checkpoint(batch_index=batch_index, checkpoint_path=checkpoint_path)
                
        return history, samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AlphaZero with Hybrid Heuristic Pruning.")
    parser.add_argument("--batches", type=int, default=1, help="Number of train/update cycles to run.")
    parser.add_argument("--episodes", type=int, default=2, help="Self-play episodes collected per update.")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent threads for self-play collection.")
    parser.add_argument("--iterations", type=int, default=50, help="MCTS iterations per decision.")
    parser.add_argument("--max-depth", type=int, default=5, help="Search depth in whole-turn plies.")
    parser.add_argument("--max-rounds", type=int, default=512, help="Hard cap for each self-play match.")
    parser.add_argument("--temp-drop", type=int, default=100, help="Round to drop temperature.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for search and environment resets.")
    parser.add_argument("--max-actions", type=int, default=96, help="Candidate action budget exposed by the env.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Shared SGD step size for the network.")
    parser.add_argument("--prior-mix", type=float, default=0.5, help="Blend ratio of learned priors against heuristics.")
    parser.add_argument("--value-mix", type=float, default=0.5, help="Blend ratio of learned value against heuristics.")
    # 增加 Dirichlet 噪声以加强网络前期的探索
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3, help="Dirichlet noise alpha.")
    parser.add_argument("--evaluation-episodes", type=int, default=2, help="Number of games to evaluate against heuristic per batch (0 to skip).")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ai_alphazero_latest.npz", help="Path to save the latest checkpoint.")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to a checkpoint (.npz) to resume training from.")
    parser.add_argument("--start-batch", type=int, default=0, help="The batch index to resume from (affects curriculum mix).")
    parser.add_argument("--save-interval", type=int, default=10, help="Save a numbered checkpoint every N batches.")
    parser.add_argument("--log-dir", type=str, default="logs/train_alphazero", help="Base directory for training logs.")
    parser.add_argument("--prefer-native-backend", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        import setproctitle
        setproctitle.setproctitle("python")
    except ImportError:
        pass
        
    import multiprocessing
    # 强制设置多进程启动方式为 spawn，解决 CUDA fork 重初始化报错
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
        
    args = parse_args()
    # 强制将项目根目录加入 sys.path，保证 AI.custom_utils 能够被正确导入
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    config = AlphaZeroTrainerConfig(
        batches=args.batches,
        episodes=args.episodes,
        workers=args.workers,
        learning_rate=args.learning_rate,
        search_iterations=args.iterations,
        max_depth=args.max_depth,
        prior_mix=args.prior_mix,
        value_mix=args.value_mix,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=args.seed,
        max_rounds=args.max_rounds,
        temperature_drop_round=args.temp_drop,
        max_actions=args.max_actions,
        checkpoint_path=args.checkpoint,
        resume_from=args.resume_from,
        start_batch=args.start_batch,
        evaluation_episodes=args.evaluation_episodes,
        save_interval=args.save_interval,
    )
    logger = TrainingLogger(base_dir=args.log_dir)
    logger.log_config({"argv": vars(args), "trainer_config": asdict(config)})
    try:
        trainer = CustomAlphaZeroSelfPlayTrainer(
            env_factory=lambda seed=0: AntWarParallelEnv(
                seed=seed,
                max_actions=args.max_actions,
                prefer_native_backend=args.prefer_native_backend,
            ),
            config=config,
            logger=logger,
        )
        
        # Check backend of the first environment to verify native backend usage
        dummy_env = AntWarParallelEnv(seed=0, prefer_native_backend=args.prefer_native_backend)
        dummy_env.reset() # 必须先 reset 才能获取到状态
        backend_type = type(dummy_env.state).__name__
        print(f"==================================================")
        print(f"Backend Initialization Check:")
        print(f"Requested Native Backend: {args.prefer_native_backend}")
        print(f"Actual Backend Loaded: {backend_type}")
        if backend_type == "NativeGameStateAdapter":
            print(f"SUCCESS: C++ Native Backend is active! Enjoy the speedup.")
        else:
            print(f"WARNING: Fallback to Python Backend! Native module might not be compiled correctly.")
        print(f"==================================================")
        
        print("Starting AlphaZero training with heuristic pruning...")
        history, samples = trainer.train()
        print(f"Training completed. Saved to {args.checkpoint}")
    except Exception as exc:
        logger.log_error(f"training failed: {exc}")
        raise
    finally:
        logger.close()

if __name__ == "__main__":
    main()
