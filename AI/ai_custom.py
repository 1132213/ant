from __future__ import annotations

from typing import List

try:
    from common import BaseAgent
except ModuleNotFoundError:
    from AI.common import BaseAgent
from SDK.backend.state import BackendState
from SDK.utils.actions import ActionBundle
from SDK.utils.constants import TowerType, SuperWeaponType

# 导入刚才写的高层状态查询接口
from custom_utils import (
    get_my_ants,
    get_enemy_ants,
    get_affordable_strategic_slots,
    get_towers_can_upgrade,
    get_best_super_weapon_target,
    generate_build_operation,
    generate_upgrade_operation,
    generate_super_weapon_operation,
    get_enemy_frontline_distance,
    evaluate_frontline_status,
    get_frontline_strategic_slots
)

class CustomAI(BaseAgent):
    """
    Phase 2: 均衡推进流 Baseline (选项 C)
    主打逻辑：根据攻防状态动态建塔（优势往前压，劣势往后缩），危急时刻放技能，有闲钱升 Heavy。
    """
    
    def choose_bundle(
        self,
        state: BackendState,
        player: int,
        bundles: List[ActionBundle] | None = None,
    ) -> ActionBundle:
        
        # 1. 危机处理：如果基地极其危险（敌方兵线 <= 4 且有钱），释放超级武器
        frontline_dist = get_enemy_frontline_distance(state, player)
        if frontline_dist <= 4:
            # 尝试寻找放闪电风暴的好位置
            target = get_best_super_weapon_target(state, player, SuperWeaponType.LIGHTNING_STORM)
            if target:
                op = generate_super_weapon_operation(SuperWeaponType.LIGHTNING_STORM, target[0], target[1])
                bundles = bundles or self.list_bundles(state, player)
                for b in bundles:
                    if any(o.op_type == op.op_type and o.arg0 == op.arg0 and o.arg1 == op.arg1 for o in b.operations):
                        return b
                        
        # 2. 评估当前攻防状态
        status = evaluate_frontline_status(state, player)
        
        # 3. 防御建设：根据攻防状态在不同位置建塔
        strategic_slots = get_frontline_strategic_slots(state, player, status)
        if strategic_slots:
            best_slot = strategic_slots[0]
            op = generate_build_operation(best_slot[0], best_slot[1])
            bundles = bundles or self.list_bundles(state, player)
            for b in bundles:
                if any(o.op_type == op.op_type and o.arg0 == op.arg0 and o.arg1 == op.arg1 for o in b.operations):
                    return b
        
        # 4. 升级已有塔：如果经济健康（比如钱 > 100）且有塔可以升级，就升级为 HEAVY
        if state.coins[player] > 100:
            upgradable_towers = get_towers_can_upgrade(state, player)
            if upgradable_towers:
                target_tower = upgradable_towers[0]
                op = generate_upgrade_operation(target_tower.tower_id, TowerType.HEAVY)
                bundles = bundles or self.list_bundles(state, player)
                for b in bundles:
                    if any(o.op_type == op.op_type and o.arg0 == op.arg0 for o in b.operations):
                        return b
        
        # 5. 空操作（攒钱或无事可做）
        bundles = bundles or self.list_bundles(state, player)
        return bundles[0]

# 统一入口暴露给 main.py 或 run_local_match.py
class AI(CustomAI):
    pass
