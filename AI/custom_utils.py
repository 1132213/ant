import math
from typing import List, Tuple, Optional

from SDK.backend.state import BackendState
from SDK.backend.model import Ant, Tower, Operation
from SDK.utils.constants import (
    TowerType, 
    SuperWeaponType, 
    OperationType,
    PLAYER_BASES,
    MAP_SIZE
)
from SDK.utils.geometry import hex_distance

def get_my_ants(state: BackendState, player: int) -> List[Ant]:
    """获取己方所有存活的蚂蚁"""
    return [ant for ant in state.ants_of(player) if ant.is_alive()]

def get_enemy_ants(state: BackendState, player: int) -> List[Ant]:
    """获取敌方所有存活的蚂蚁"""
    return [ant for ant in state.ants_of(1 - player) if ant.is_alive()]

def get_my_towers(state: BackendState, player: int) -> List[Tower]:
    """获取己方所有防御塔"""
    return state.towers_of(player)

def get_enemy_towers(state: BackendState, player: int) -> List[Tower]:
    """获取敌方所有防御塔"""
    return state.towers_of(1 - player)

def get_base_pos(player: int) -> Tuple[int, int]:
    """获取基地坐标"""
    return PLAYER_BASES[player]

def distance_to_my_base(player: int, x: int, y: int) -> int:
    """计算某个坐标到己方基地的距离"""
    bx, by = get_base_pos(player)
    return hex_distance(bx, by, x, y)

def distance_to_enemy_base(player: int, x: int, y: int) -> int:
    """计算某个坐标到敌方基地的距离"""
    bx, by = get_base_pos(1 - player)
    return hex_distance(bx, by, x, y)

def get_enemy_frontline_distance(state: BackendState, player: int) -> int:
    """
    获取敌方兵线距离己方基地的最短距离（数值越小越危险）
    """
    return state.nearest_ant_distance(player)

def evaluate_frontline_status(state: BackendState, player: int) -> str:
    """
    评估当前局面的攻守状态。
    返回: 'DEFEND' (危险), 'BALANCED' (僵持), 'ATTACK' (优势)
    """
    my_danger_dist = state.nearest_ant_distance(player)
    enemy_danger_dist = state.nearest_ant_distance(1 - player)
    
    if my_danger_dist <= 5:
        return 'DEFEND'
    elif enemy_danger_dist <= 5:
        return 'ATTACK'
    else:
        return 'BALANCED'

def get_frontline_strategic_slots(state: BackendState, player: int, status: str) -> List[Tuple[int, int]]:
    """
    根据攻防状态，筛选战略位置。
    DEFEND: 倾向于在己方半场、离基地近的地方建塔。
    ATTACK: 倾向于在敌方半场、离敌方基地近的地方建塔（封门）。
    BALANCED: 倾向于在敌方威胁热度高的区域附近建塔。
    """
    slots = get_affordable_strategic_slots(state, player)
    if not slots:
        return []
        
    my_base_x, my_base_y = get_base_pos(player)
    enemy_base_x, enemy_base_y = get_base_pos(1 - player)
    
    if status == 'ATTACK':
        # 按照离敌方基地由近到远排序（往前线压）
        slots.sort(key=lambda pos: hex_distance(pos[0], pos[1], enemy_base_x, enemy_base_y))
    else:
        # DEFEND 和 BALANCED 状态，利用威胁热力图找到最需要防守的前线交火区域
        # 移除原先后场建塔逻辑，确保在交战前线建塔
        heatmap = calculate_threat_heatmap(state, player, radius=3)
        slots.sort(key=lambda pos: heatmap.get(pos, 0.0), reverse=True)
        
    return slots

def get_affordable_strategic_slots(state: BackendState, player: int) -> List[Tuple[int, int]]:
    """
    获取当前买得起、可以建塔的战略位置（已按官方启发式优先级排序）。
    排除了被 EMP 干扰的区域和周围拥挤的区域。
    """
    cost = state.build_tower_cost()
    if state.coins[player] < cost:
        return []
        
    valid_slots = []
    for x, y in state.strategic_slots(player):
        # 确保该位置及周围为空，且没有被EMP干扰
        if state.current_and_neighbors_empty(x, y) and not state.is_shielded_by_emp(player, x, y):
            valid_slots.append((x, y))
    return valid_slots

def get_towers_can_upgrade(state: BackendState, player: int) -> List[Tower]:
    """
    获取己方当前可能可以升级的塔（简单判断金币和等级，后续由 can_apply_operation 精确验证）。
    按该塔所在位置的威胁热度从高到低排序，优先升级前线或受压力的塔。
    """
    towers = get_my_towers(state, player)
    upgradable = []
    for t in towers:
        if state.is_shielded_by_emp(player, t.x, t.y):
            continue
        # 放宽判断：只要不是满级且有基础升级费用即可
        if t.level < 2 and state.coins[player] >= 60:
            upgradable.append(t)
            
    # 利用热力图进行优先级排序，优先升级高威胁区域的塔
    heatmap = calculate_threat_heatmap(state, player, radius=3)
    upgradable.sort(key=lambda t: heatmap.get((t.x, t.y), 0.0), reverse=True)
    
    return upgradable

# 缓存机制，避免同一回合内重复计算热力图
_heatmap_cache = {}
_heatmap_cache_turn = -1

def calculate_threat_heatmap(state: BackendState, player: int, radius: int = 3) -> dict[Tuple[int, int], float]:
    """
    计算全图每个格子的敌方威胁热力图（用于超级武器释放及高级防御策略）
    权重考虑：
    1. 距离己方基地的远近（越近威胁越大）
    2. 蚂蚁的当前血量（血量越高，越难处理，或者反向认为血少的更容易被闪电风暴清掉，这里取综合考虑）
    3. 蚂蚁的移动速度
    返回: {(x, y): 综合威胁值}
    """
    global _heatmap_cache, _heatmap_cache_turn
    current_turn = getattr(state, "current_turn", getattr(state, "turn", 0))
    if _heatmap_cache_turn == current_turn and player in _heatmap_cache:
        return _heatmap_cache[player]

    enemy_ants = get_enemy_ants(state, player)
    enemy_towers = get_enemy_towers(state, player)
    heatmap = {}
    
    my_base_x, my_base_y = get_base_pos(player)
    
    for ant in enemy_ants:
        # 1. 距离因子：距离基地越近，威胁呈指数级上升
        dist_to_base = hex_distance(ant.x, ant.y, my_base_x, my_base_y)
        # 防止除以 0，加一个小常数。距离越小，权重越大。
        distance_weight = 10.0 / (dist_to_base + 1.0)
        
        # 2. 血量因子：闪电风暴伤害是固定范围的，但如果全是大血量蚂蚁，不一定能秒掉；
        # 这里把血量当做“需要集火”的指标，血量越高威胁越高。
        hp_weight = ant.hp / 100.0  # 假设初始兵大概几十到一百血，这里做个缩放
        
        # 3. 速度因子：跑得快的威胁更大
        speed_weight = 1.0 + (ant.speed / 10.0)
        
        # 战斗蚂蚁高额赏金：给予战斗蚂蚁极高的威胁权重，吸引闪电风暴
        combat_bonus = 3.0 if getattr(ant, 'kind', 0) == 1 else 1.0
        
        # 单只蚂蚁的综合威胁分
        ant_threat = distance_weight * speed_weight * (1.0 + hp_weight) * combat_bonus
        
        # 使用预先计算好的六边形偏移量可以略微加速
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = ant.x + dx, ant.y + dy
                if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
                    if hex_distance(ant.x, ant.y, nx, ny) <= radius:
                        heatmap[(nx, ny)] = heatmap.get((nx, ny), 0.0) + ant_threat

    # 将敌方防御塔加入热力图评估（用于闪电风暴等拆塔战略）
    for tower in enemy_towers:
        # 塔的基础威胁：等级越高，威胁越大。闪电风暴可拆塔，提高塔的权重
        base_threat = (tower.level + 1) * 3.0
        
        # 拆塔残血诱惑：如果塔血量较低（特别是 <= 3 或 <= 6，容易被闪电风暴收割），增加威胁分
        hp_ratio = tower.hp / float(tower.max_hp if hasattr(tower, 'max_hp') else 15.0)
        vulnerability_bonus = 5.0 * (1.0 - hp_ratio)
        if tower.hp <= 6:
            vulnerability_bonus += 5.0
            
        tower_threat = base_threat + vulnerability_bonus
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = tower.x + dx, tower.y + dy
                if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
                    if hex_distance(tower.x, tower.y, nx, ny) <= radius:
                        heatmap[(nx, ny)] = heatmap.get((nx, ny), 0.0) + tower_threat
                        
    if _heatmap_cache_turn != current_turn:
        _heatmap_cache.clear()
        _heatmap_cache_turn = current_turn
    _heatmap_cache[player] = heatmap

    return heatmap

def get_fast_heuristic_enemy_action(state: BackendState, player: int) -> Tuple[str, Tuple[Operation, ...]]:
    """
    Fast heuristic to predict the enemy's most likely action (Scheme A).
    Returns a tuple of (action_name, operations_tuple) for the enemy.
    
    Logic:
    1. Super Weapons: Check Lightning Storm (>=150), EMP (>=150), Deflector (>=100).
    2. Base Tech Upgrades: If economy is good and no immediate danger.
    3. Tower Upgrades: If there are existing towers under threat.
    4. Downgrade Towers: If poor (<100 coins) and have useless backline towers.
    5. Build Towers: If under pressure and has >= 50 coins.
    6. Hold: Default.
    """
    from SDK.utils.constants import SuperWeaponType, OperationType
    
    status = evaluate_frontline_status(state, player)
    
    # 1. Check Super Weapons (High priority if affordable and targets exist)
    if state.coins[player] >= 150:
        target = get_best_super_weapon_target(state, player, SuperWeaponType.LIGHTNING_STORM)
        if target:
            op = generate_super_weapon_operation(SuperWeaponType.LIGHTNING_STORM, target[0], target[1])
            return "storm", (op,)
            
    if state.coins[player] >= 150:
        target = get_best_super_weapon_target(state, player, SuperWeaponType.EMP_BLASTER)
        if target:
            op = generate_super_weapon_operation(SuperWeaponType.EMP_BLASTER, target[0], target[1])
            return "emp", (op,)
            
    if state.coins[player] >= 100:
        target = get_best_super_weapon_target(state, player, SuperWeaponType.DEFLECTOR)
        if target:
            op = generate_super_weapon_operation(SuperWeaponType.DEFLECTOR, target[0], target[1])
            return "deflector", (op,)
            
    # 2. Check Base Tech Upgrades (If safe and rich)
    if status == "ATTACK" or (status == "BALANCED" and state.coins[player] >= 300):
        # Prefer upgrading ant quality or generation speed
        for op_type in [OperationType.UPGRADE_GENERATION_SPEED, OperationType.UPGRADE_GENERATED_ANT]:
            op = generate_base_upgrade(op_type)
            if state.can_apply_operation(player, op):
                return f"tech_{op_type.name}", (op,)
                
    # 3. Check Tower Upgrades
    if state.coins[player] >= 60:
        upgradable_towers = get_towers_can_upgrade(state, player)
        if upgradable_towers:
            target_tower = upgradable_towers[0]
            from SDK.utils.constants import TowerType, TOWER_UPGRADE_TREE
            
            # Dynamic heuristic: Choose the first available upgrade path for the current tower type
            if target_tower.tower_type in TOWER_UPGRADE_TREE:
                possible_upgrades = TOWER_UPGRADE_TREE[target_tower.tower_type]
                if possible_upgrades:
                    # Default to the first path (usually the most direct upgrade, e.g., HEAVY or HEAVY_PLUS)
                    target_type = possible_upgrades[0]
                    op = generate_upgrade_operation(target_tower.tower_id, target_type)
                    if state.can_apply_operation(player, op):
                        return f"upgrade_{target_tower.tower_id}_{target_type.name}", (op,)
                
    # 4. Check Downgrading Towers (If poor or towers are useless in the backline)
    # 拆除/降级防御塔：当经济较低，或者后方防御塔已经没有战略价值时
    if state.coins[player] < 100:
        my_towers = get_my_towers(state, player)
        if my_towers:
            # 使用热力图评估塔的当前战略价值
            heatmap = calculate_threat_heatmap(state, player, radius=3)
            for t in my_towers:
                # 如果这个塔周围没有任何威胁，且不是在前线，可以考虑卖掉
                if heatmap.get((t.x, t.y), 0.0) < 0.5:
                    # 距离己方基地很近但没威胁的塔，说明战线已经推远了，或者这是一个废塔
                    my_base_x, my_base_y = get_base_pos(player)
                    enemy_base_x, enemy_base_y = get_base_pos(1 - player)
                    dist_to_enemy = hex_distance(t.x, t.y, enemy_base_x, enemy_base_y)
                    # 如果离敌人基地远（比如距离大于15），且周围没威胁，则降级换钱
                    if dist_to_enemy > 15:
                        op = generate_downgrade_operation(t.tower_id)
                        if state.can_apply_operation(player, op):
                            return f"downgrade_{t.tower_id}", (op,)
                            
    # 5. Check Building Towers (Under pressure)
    if status in ["DEFEND", "BALANCED"] and state.coins[player] >= 50:
        slots = get_frontline_strategic_slots(state, player, status)
        if slots:
            best_slot = slots[0]
            op = generate_build_operation(best_slot[0], best_slot[1])
            return f"build_{best_slot[0]}_{best_slot[1]}", (op,)
            
    # 6. Default to hold
    return "hold", ()

def get_best_super_weapon_target(state: BackendState, player: int, weapon_type: SuperWeaponType) -> Optional[Tuple[int, int]]:
    """
    根据超级武器类型，寻找最佳释放位置
    """
    cost = state.weapon_cost(weapon_type)
    if state.coins[player] < cost:
        return None
        
    # 检查冷却
    if state.weapon_cooldowns[player][weapon_type.value - 1] > 0:
        return None

    if weapon_type == SuperWeaponType.LIGHTNING_STORM:
        # 闪电风暴：寻找敌方威胁（战斗蚂蚁+高价值防御塔）最集中的区域
        heatmap = calculate_threat_heatmap(state, player, radius=3)
        if not heatmap:
            return None
        best_pos = max(heatmap.items(), key=lambda item: item[1])
        # 提高释放阈值，确保砸在真正有价值的目标上（高等级塔或战斗蚂蚁群）
        if best_pos[1] < 3.0:  
            return None
        return best_pos[0]

    elif weapon_type == SuperWeaponType.EMP_BLASTER:
        # EMP轰炸：寻找敌方防御塔最密集的区域（用来瘫痪塔阵）
        enemy_towers = get_enemy_towers(state, player)
        if not enemy_towers:
            return None
        
        best_pos = None
        max_towers_hit = 0
        
        # 遍历敌方每个塔，假设以它为中心放 EMP
        for center_tower in enemy_towers:
            hit_count = 0
            for other_tower in enemy_towers:
                if hex_distance(center_tower.x, center_tower.y, other_tower.x, other_tower.y) <= 3:
                    # 优先瘫痪高级塔
                    hit_count += (other_tower.level + 1)
            
            if hit_count > max_towers_hit:
                max_towers_hit = hit_count
                best_pos = (center_tower.x, center_tower.y)
                
        # 至少能瘫痪 2 级以上的塔才值得放
        if max_towers_hit >= 2 and best_pos is not None:
            return best_pos

    elif weapon_type == SuperWeaponType.DEFLECTOR or weapon_type == SuperWeaponType.EMERGENCY_EVASION:
        # 引力护盾 / 紧急回避：保护我方冲锋在最前线的蚂蚁群（主要保护高价值战斗蚂蚁）
        my_ants = get_my_ants(state, player)
        if not my_ants:
            return None
            
        enemy_base_x, enemy_base_y = get_base_pos(1 - player)
        best_pos = None
        max_ants_saved = 0
        
        for center_ant in my_ants:
            saved_count = 0
            for other_ant in my_ants:
                if hex_distance(center_ant.x, center_ant.y, other_ant.x, other_ant.y) <= 3:
                    # 战斗蚂蚁造价昂贵且赏金高，保护优先级极大提升
                    saved_count += 5 if getattr(other_ant, 'kind', 0) == 1 else 1
                    
            if saved_count > max_ants_saved:
                max_ants_saved = saved_count
                best_pos = (center_ant.x, center_ant.y)
                
        # 至少能保护 1 只战斗蚂蚁或多只工蚁才值得放
        if max_ants_saved >= 5 and best_pos is not None:
            return best_pos
        
    return None

def generate_build_operation(x: int, y: int) -> Operation:
    return Operation(OperationType.BUILD_TOWER, x, y)

def generate_upgrade_operation(tower_id: int, target_type: TowerType) -> Operation:
    return Operation(OperationType.UPGRADE_TOWER, tower_id, int(target_type))

def generate_downgrade_operation(tower_id: int) -> Operation:
    return Operation(OperationType.DOWNGRADE_TOWER, tower_id)

def generate_base_upgrade(op_type: OperationType) -> Operation:
    return Operation(op_type)

def generate_super_weapon_operation(weapon_type: SuperWeaponType, x: int, y: int) -> Operation:
    if weapon_type == SuperWeaponType.LIGHTNING_STORM:
        op = OperationType.USE_LIGHTNING_STORM
    elif weapon_type == SuperWeaponType.EMP_BLASTER:
        op = OperationType.USE_EMP_BLASTER
    elif weapon_type == SuperWeaponType.DEFLECTOR:
        op = OperationType.USE_DEFLECTOR
    else:
        op = OperationType.USE_EMERGENCY_EVASION
    return Operation(op, x, y)
