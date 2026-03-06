from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

try:
    from common import BaseAgent
except ImportError:
    from AI.common import BaseAgent

from SDK.actions import ActionBundle

try:
    from greedy_runtime import (
        GreedyController,
        GameInfo,
        MAX_ROUND,
        Operation,
        OperationType,
        Simulator,
        SuperWeaponType,
        TowerType,
        hex_distance,
        info_from_state,
        is_valid_pos,
    )
except ImportError:
    from AI.greedy_runtime import (
        GreedyController,
        GameInfo,
        MAX_ROUND,
        Operation,
        OperationType,
        Simulator,
        SuperWeaponType,
        TowerType,
        hex_distance,
        info_from_state,
        is_valid_pos,
    )

MAX_NODE_COUNT = 20000
SEARCH_SELECT_BUDGET = 256

SLOT_LAYOUT = {
    0: (
        (2, 9),
        (4, 9),
        (5, 9),
        (5, 7),
        (6, 9),
        (5, 11),
        (5, 6),
        (6, 7),
        (6, 11),
        (5, 12),
        (4, 3),
        (5, 3),
        (7, 8),
        (7, 10),
        (4, 15),
        (5, 15),
        (4, 2),
        (6, 4),
        (7, 5),
        (8, 7),
        (8, 11),
        (7, 13),
        (6, 14),
        (4, 16),
        (6, 1),
        (6, 2),
        (6, 16),
        (6, 17),
        (7, 1),
        (8, 4),
        (8, 14),
        (7, 17),
        (8, 2),
        (8, 16),
        (3, 9),
    ),
    1: (
        (16, 9),
        (14, 9),
        (13, 9),
        (13, 7),
        (12, 9),
        (13, 11),
        (12, 6),
        (12, 7),
        (12, 11),
        (12, 12),
        (14, 3),
        (13, 3),
        (10, 8),
        (10, 10),
        (14, 15),
        (13, 15),
        (13, 2),
        (11, 4),
        (11, 5),
        (10, 7),
        (10, 11),
        (11, 13),
        (11, 14),
        (13, 16),
        (12, 1),
        (11, 2),
        (11, 16),
        (12, 17),
        (11, 1),
        (9, 4),
        (9, 14),
        (11, 17),
        (9, 2),
        (9, 16),
        (15, 9),
    ),
}

BASE_SLOT = 0
STORM_SLOT = 34
TACTICAL_SITES = (
    1,
    2,
    4,
    10,
    16,
    11,
    14,
    23,
    15,
    17,
    18,
    22,
    21,
    3,
    6,
    7,
    5,
    9,
    8,
    19,
    12,
    13,
    20,
    24,
    25,
    28,
    27,
    26,
    31,
)

SITE_GROUPS = {}
for cluster in (
    (1, 2, 4),
    (3, 6, 7),
    (5, 9, 8),
    (10, 16, 11),
    (14, 23, 15),
    (19, 12, 13, 20),
    (17, 18),
    (22, 21),
    (24, 25, 28),
    (27, 26, 31),
    (32, 29),
    (30, 33),
):
    for site in cluster:
        SITE_GROUPS[site] = cluster

UPGRADE_PATHS = {
    TowerType.BASIC: (TowerType.HEAVY, TowerType.MORTAR, TowerType.QUICK),
    TowerType.HEAVY: (TowerType.HEAVY_PLUS, TowerType.CANNON, TowerType.ICE),
    TowerType.MORTAR: (TowerType.MORTAR_PLUS, TowerType.MISSILE, TowerType.PULSE),
    TowerType.QUICK: (TowerType.QUICK_PLUS, TowerType.DOUBLE, TowerType.SNIPER),
}


@dataclass(slots=True)
class GreedyLedger:
    side: int = 0
    current_round: int = 0
    mode: int = 0
    current_hp: int = 0
    enemy_old_count: int = 0
    enemy_die_count: int = 0
    attack_stance: bool = False
    last_superweapon_type: int = int(SuperWeaponType.LIGHTNING_STORM)
    last_superweapon_round: int = -1
    reserved_bias: int = 0


@dataclass(slots=True)
class SearchNode:
    sim: Simulator
    node_id: int = -1
    parent: int = -1
    children: list[int] = field(default_factory=list)
    actions: list[Operation] = field(default_factory=list)
    node_val: float = 0.0
    max_val: float = 0.0
    turn: int = 0
    loss: int = 0
    max_expand: int = 0
    expand_count: int = 0
    fail_round: int = 0
    danger: bool = False
    safe: bool = True
    dis_vals: list[int] = field(default_factory=lambda: [0] * 60)

    def __post_init__(self) -> None:
        self.turn = _as_int(self.sim.info.round)


@dataclass(slots=True)
class ActionResult:
    operation: Operation | None
    coins: int
    towers: int


@dataclass(slots=True)
class BestChoice:
    score: float
    index: int


def _as_int(value) -> int:
    if isinstance(value, bytes):
        return value[0]
    if isinstance(value, str):
        return ord(value)
    return int(value)


def _sdk_operation(op_type: OperationType, arg0: int = -1, arg1: int = -1) -> Operation:
    return Operation(op_type, arg0, arg1)


def _tower_kind(tower) -> TowerType:
    return TowerType(_as_int(tower.type))


def _slot(side: int, code: int) -> tuple[int, int]:
    return SLOT_LAYOUT[side][code]


def _site_group(code: int) -> tuple[int, ...]:
    return SITE_GROUPS[code]


def _find_tower_by_pos(info, x: int, y: int):
    for tower in info.towers:
        if _as_int(tower.x) == x and _as_int(tower.y) == y:
            return tower
    return None


def _find_tower_by_id(info, tower_id: int):
    for tower in info.towers:
        if _as_int(tower.id) == tower_id:
            return tower
    return None


def _enemy_front_distance(side: int, info) -> int:
    target_x, target_y = _slot(side ^ 1, BASE_SLOT)
    best = 100
    for ant in info.ants:
        if _as_int(ant.player) == side:
            best = min(best, hex_distance(_as_int(ant.x), _as_int(ant.y), target_x, target_y))
    return best


def _safe_coin(side: int, info) -> int:
    enemy = side ^ 1
    emp_cd = _as_int(info.super_weapon_cd[enemy][int(SuperWeaponType.EMP_BLASTER)])
    enemy_coins = _as_int(info.coins[enemy])
    if emp_cd >= 90:
        return 0
    if emp_cd > 0:
        return max(int(min(enemy_coins, 149) - emp_cd * 1.66), 0)
    return min(enemy_coins, 149)


def _safe_value(side: int, info) -> int:
    return min(0, _as_int(info.coins[side]) - _safe_coin(side, info))


def _nearest_ant_to_base(side: int, info) -> int:
    base_x, base_y = _slot(side, BASE_SLOT)
    best = 32
    for ant in info.ants:
        if _as_int(ant.player) != side ^ 1:
            continue
        distance = hex_distance(_as_int(ant.x), _as_int(ant.y), base_x, base_y)
        if distance < best:
            best = distance
    return best


def _cxx_div(numerator: int | float, denominator: int) -> int:
    return math.trunc(numerator / denominator)


class GreedyPlanner:
    def __init__(self) -> None:
        self.memory = GreedyLedger()
        self.nodes: list[SearchNode] = []

    @property
    def side(self) -> int:
        return self.memory.side

    @property
    def enemy(self) -> int:
        return self.memory.side ^ 1

    def reset(self) -> None:
        self.memory = GreedyLedger()
        self.nodes = []

    def _prepare_new_match(self, player: int) -> None:
        self.reset()
        self.memory.side = player

    def _best_value(self, candidates: Iterable[tuple], value_index: int) -> tuple | None:
        best = None
        best_value = -1e18
        for candidate in candidates:
            value = candidate[value_index]
            if value > best_value:
                best = candidate
                best_value = value
        return best

    def _can_build_or_upgrade(
        self,
        code: int,
        action_kind: int,
        info,
        coins: int,
        towers: int,
        branch: int = 0,
        ignored_site: int = -1,
        blocked_sites: set[int] | None = None,
    ) -> ActionResult:
        blocked_sites = blocked_sites or set()
        x, y = _slot(self.side, code)
        building_tag = _as_int(info.building_tag[x][y])
        if action_kind == 1:
            cost = _as_int(info.build_tower_cost(towers))
            if coins < cost:
                return ActionResult(None, coins, towers)
            for site in _site_group(code):
                if site == ignored_site:
                    continue
                if site in blocked_sites:
                    return ActionResult(None, coins, towers)
                sx, sy = _slot(self.side, site)
                if _as_int(info.building_tag[sx][sy]) != 0:
                    return ActionResult(None, coins, towers)
            return ActionResult(_sdk_operation(OperationType.BUILD_TOWER, x, y), coins - cost, towers + 1)

        if action_kind == 2:
            if building_tag == 0:
                return ActionResult(None, coins, towers)
            tower = _find_tower_by_pos(info, x, y)
            if tower is None:
                return ActionResult(None, coins, towers)
            tower_type = _tower_kind(tower)
            if tower_type.value // 10 > 0:
                return ActionResult(None, coins, towers)
            target_type = UPGRADE_PATHS[tower_type][branch]
            cost = _as_int(info.upgrade_tower_cost(int(target_type)))
            if coins < cost:
                return ActionResult(None, coins, towers)
            return ActionResult(
                _sdk_operation(OperationType.UPGRADE_TOWER, _as_int(tower.id), int(target_type)),
                coins - cost,
                towers,
            )

        if action_kind == 3:
            if building_tag == 0:
                return ActionResult(None, coins, towers)
            tower = _find_tower_by_pos(info, x, y)
            if tower is None or _tower_kind(tower) != TowerType.BASIC:
                return ActionResult(None, coins, towers)
            refund = _as_int(info.destroy_tower_income(towers))
            return ActionResult(_sdk_operation(OperationType.DOWNGRADE_TOWER, _as_int(tower.id)), coins + refund, towers - 1)

        if action_kind == 4:
            if building_tag == 0:
                return ActionResult(None, coins, towers)
            tower = _find_tower_by_pos(info, x, y)
            if tower is None or _tower_kind(tower) == TowerType.BASIC:
                return ActionResult(None, coins, towers)
            refund = _as_int(info.downgrade_tower_income(int(_tower_kind(tower))))
            return ActionResult(_sdk_operation(OperationType.DOWNGRADE_TOWER, _as_int(tower.id)), coins + refund, towers)

        return ActionResult(None, coins, towers)

    def _series_actions(self, tactic: int, info, blocked_sites: set[int]) -> list[list[Operation]]:
        operations: list[list[Operation]] = []
        if tactic == 0:
            for code in TACTICAL_SITES:
                if code in blocked_sites:
                    continue
                result = self._can_build_or_upgrade(code, 1, info, _as_int(info.coins[self.side]), _as_int(info.tower_num_of_player(self.side)), blocked_sites=blocked_sites)
                if result.operation is not None:
                    operations.append([result.operation])
            return operations

        if tactic == 1:
            for code in TACTICAL_SITES:
                if code in blocked_sites:
                    continue
                for branch in range(3):
                    result = self._can_build_or_upgrade(code, 2, info, _as_int(info.coins[self.side]), _as_int(info.tower_num_of_player(self.side)), branch=branch, blocked_sites=blocked_sites)
                    if result.operation is not None:
                        operations.append([result.operation])
            return operations

        if tactic == 2:
            for code in TACTICAL_SITES:
                if code in blocked_sites:
                    continue
                first = self._can_build_or_upgrade(code, 4, info, _as_int(info.coins[self.side]), _as_int(info.tower_num_of_player(self.side)), blocked_sites=blocked_sites)
                if first.operation is None:
                    continue
                for code2 in TACTICAL_SITES:
                    if code2 in blocked_sites or code2 == code:
                        continue
                    second = self._can_build_or_upgrade(code2, 1, info, first.coins, first.towers, blocked_sites=blocked_sites)
                    if second.operation is not None:
                        operations.append([first.operation, second.operation])
            return operations

        if tactic == 3:
            for code in TACTICAL_SITES:
                if code in blocked_sites:
                    continue
                result = self._can_build_or_upgrade(code, 3, info, _as_int(info.coins[self.side]), _as_int(info.tower_num_of_player(self.side)), blocked_sites=blocked_sites)
                if result.operation is not None:
                    operations.append([result.operation])
            return operations

        if tactic == 4:
            for code in TACTICAL_SITES:
                if code in blocked_sites:
                    continue
                first = self._can_build_or_upgrade(code, 3, info, _as_int(info.coins[self.side]), _as_int(info.tower_num_of_player(self.side)), blocked_sites=blocked_sites)
                if first.operation is None:
                    continue
                for code2 in TACTICAL_SITES:
                    if code2 in blocked_sites or code2 == code:
                        continue
                    for branch in range(3):
                        second = self._can_build_or_upgrade(code2, 2, info, first.coins, first.towers, branch=branch, blocked_sites=blocked_sites)
                        if second.operation is not None:
                            operations.append([first.operation, second.operation])
            return operations

        if tactic == 5:
            for code in TACTICAL_SITES:
                if code in blocked_sites:
                    continue
                result = self._can_build_or_upgrade(code, 4, info, _as_int(info.coins[self.side]), _as_int(info.tower_num_of_player(self.side)), blocked_sites=blocked_sites)
                if result.operation is not None:
                    operations.append([result.operation])
            return operations

        if tactic == 6:
            for code in TACTICAL_SITES:
                if code in blocked_sites:
                    continue
                first = self._can_build_or_upgrade(code, 3, info, _as_int(info.coins[self.side]), _as_int(info.tower_num_of_player(self.side)), blocked_sites=blocked_sites)
                if first.operation is None:
                    continue
                for code2 in TACTICAL_SITES:
                    if code2 in blocked_sites or code2 == code:
                        continue
                    second = self._can_build_or_upgrade(code2, 1, info, first.coins, first.towers, ignored_site=code, blocked_sites=blocked_sites)
                    if second.operation is not None:
                        operations.append([first.operation, second.operation])
            return operations

        if tactic == 7:
            for code in TACTICAL_SITES:
                if code in blocked_sites:
                    continue
                first = self._can_build_or_upgrade(code, 4, info, _as_int(info.coins[self.side]), _as_int(info.tower_num_of_player(self.side)), blocked_sites=blocked_sites)
                if first.operation is None:
                    continue
                for code2 in TACTICAL_SITES:
                    if code2 in blocked_sites or code2 == code:
                        continue
                    for branch in range(3):
                        second = self._can_build_or_upgrade(code2, 2, info, first.coins, first.towers, branch=branch, blocked_sites=blocked_sites)
                        if second.operation is not None:
                            operations.append([first.operation, second.operation])
        return operations

    def _evaluate(self, node: SearchNode) -> float:
        sim = node.sim.clone()
        info = sim.info
        offset = _as_int(info.round) - self.memory.current_round
        if offset < 60:
            node.dis_vals[offset] = _nearest_ant_to_base(self.side, info)

        safe_val = 0
        if self.memory.current_round > 60:
            safe_val = _safe_value(self.side, info)
            node.safe = safe_val == 0

        ruin_round = self.memory.current_round + 60
        fail_flag = False
        if _as_int(info.bases[self.side].hp) <= self.memory.current_hp - 1:
            fail_flag = True
            ruin_round = node.fail_round

        if not fail_flag:
            node.fail_round = self.memory.current_round + 60
            for turn in range(_as_int(info.round), self.memory.current_round + 60):
                if not sim.fast_next_round(self.side):
                    break
                node.dis_vals[turn - self.memory.current_round] = _nearest_ant_to_base(self.side, info)
                if _as_int(info.bases[self.side].hp) <= self.memory.current_hp - 1:
                    node.fail_round = _as_int(info.round)
                    if _as_int(info.bases[self.side].hp) <= self.memory.current_hp - 2:
                        ruin_round = node.fail_round
                    break

        if _as_int(info.bases[self.side].hp) > self.memory.current_hp - 2:
            for turn in range(_as_int(info.round), self.memory.current_round + 60):
                if not sim.fast_next_round(self.side):
                    break
                node.dis_vals[turn - self.memory.current_round] = _nearest_ant_to_base(self.side, info)
                if _as_int(info.bases[self.side].hp) <= self.memory.current_hp - 2:
                    ruin_round = _as_int(info.round)
                    break

        enemy_ant_level = _as_int(info.bases[self.enemy].ant_level)
        ant_ratio = (3.0, 5.0, 7.0)[enemy_ant_level]
        node.node_val = (
            (_as_int(info.bases[self.side].hp) - self.memory.current_hp)
            + (node.fail_round - self.memory.current_round) * 0.8
            + (ruin_round - node.fail_round) * 0.1
            - node.loss * 1.5
            + 20
        )
        if self.memory.mode == 0:
            node.node_val += (
                -(_as_int(info.old_count[self.enemy]) - self.memory.enemy_old_count) * ant_ratio * 2
                + (_as_int(info.die_count[self.enemy]) - self.memory.enemy_die_count) * ant_ratio * 1.5
            )

        if node.fail_round - self.memory.current_round <= 16:
            node.danger = True
            node.node_val -= 500
            if ruin_round - node.fail_round <= 8:
                node.node_val -= 300

        if not node.safe and not node.danger and self.memory.mode >= 0:
            node.node_val += (-40 + _cxx_div(safe_val, 5)) * min(_cxx_div(self.memory.current_round - 60, 30), 1)

        tower_positions: list[tuple[int, int]] = []
        corrected_tower_num = _as_int(info.tower_num_of_player(self.side))
        node.node_val -= (2 ** corrected_tower_num - 1) * 15 * 0.2 * 0.75
        distanced = False
        for tower in info.towers:
            if _as_int(tower.player) != self.side:
                continue
            tower_type = _as_int(tower.type)
            if tower_type > 0 and tower_type // 10 == 0:
                node.node_val -= 60 * 0.2 * 0.75
            elif tower_type // 10 > 0:
                node.node_val -= 260 * 0.2 * 0.75
            tower_positions.append((_as_int(tower.x), _as_int(tower.y)))

        for index, position in enumerate(tower_positions[:-1]):
            for other in tower_positions[index + 1 :]:
                distance = hex_distance(position[0], position[1], other[0], other[1])
                if distance <= 3:
                    node.node_val -= 5
                elif distance <= 6:
                    node.node_val -= 2
                else:
                    distanced = True
        if corrected_tower_num >= 3 and not distanced:
            node.node_val -= 20

        base_x = _as_int(info.bases[self.side].x)
        base_y = _as_int(info.bases[self.side].y)
        for x, y in tower_positions:
            node.node_val += hex_distance(x, y, base_x, base_y) * 0.4

        if self.memory.mode >= 0:
            close_flag = False
            horizon = min(60, _as_int(info.round) - self.memory.current_round - 4)
            for index in range(max(horizon, 0)):
                distance = node.dis_vals[index]
                if distance <= 3:
                    close_flag = True
                if distance == 5:
                    node.node_val -= 0.2
                elif distance == 4:
                    node.node_val -= 0.5
                    node.node_val -= 2
                elif distance in (1, 2, 3):
                    node.node_val -= 2
            if close_flag:
                node.node_val -= 20
            ant_count = 0
            mis_val = 0.0
            for ant in info.ants:
                if _as_int(ant.player) != self.enemy:
                    continue
                mis_val += 32 - _as_int(ant.age) - hex_distance(_as_int(ant.x), _as_int(ant.y), base_x, base_y) * 1.5
                ant_count += 1
            if ant_count > 0 and self.memory.current_round >= 20:
                node.node_val += mis_val / ant_count * 0.5

        node.max_val = node.node_val
        return node.node_val

    def _blocked_sites(self, info) -> set[int]:
        blocked: set[int] = set()
        for effect in info.super_weapons:
            if _as_int(effect.player) == self.enemy and _as_int(effect.type) == int(SuperWeaponType.EMP_BLASTER):
                for code in range(34):
                    x, y = _slot(self.side, code)
                    if hex_distance(_as_int(effect.x), _as_int(effect.y), x, y) <= 3:
                        blocked.add(code)
                break
        return blocked

    def _expand(self, node: SearchNode, is_root: bool = False) -> None:
        info = node.sim.info
        if _as_int(info.round) >= MAX_ROUND or _as_int(info.bases[self.side].hp) <= 0 or _as_int(info.bases[self.enemy].hp) <= 0:
            return
        if not is_root:
            offset = _as_int(info.round) - self.memory.current_round
            if offset < 60:
                node.dis_vals[offset] = _nearest_ant_to_base(self.side, info)
            if not node.sim.fast_next_round(self.side):
                return
            info = node.sim.info

        blocked_sites = self._blocked_sites(info)
        action_sets: list[list[Operation]] = []
        for tactic in range(8):
            if node.actions and tactic in (3, 5):
                continue
            if len(node.actions) == 1 and node.actions[0].op_type == OperationType.BUILD_TOWER and node.expand_count < 2 and tactic in (3, 4, 6):
                continue
            if len(node.actions) == 1 and node.actions[0].op_type == OperationType.UPGRADE_TOWER and node.expand_count < 2 and tactic == 2:
                continue
            if len(node.actions) == 2 and node.actions[1].op_type == OperationType.BUILD_TOWER and node.expand_count < 2 and tactic in (3, 4, 6):
                continue
            if _as_int(info.tower_num_of_player(self.side)) >= 4 and tactic in (0, 2):
                continue
            action_sets.extend(self._series_actions(tactic, info, blocked_sites))

        if is_root:
            empty_child = SearchNode(node.sim.clone())
            empty_child.node_id = len(self.nodes)
            empty_child.parent = node.node_id
            self._evaluate(empty_child)
            self.nodes.append(empty_child)
            node.children.append(empty_child.node_id)

        for op_sequence in action_sets:
            if len(self.nodes) >= MAX_NODE_COUNT - 10:
                break
            child = SearchNode(Simulator(info.clone()))
            child.node_id = len(self.nodes)
            child.parent = node.node_id
            child.loss = node.loss
            child.fail_round = node.fail_round
            child.node_val = -1e9
            child.max_val = -1e9
            prefix = min(60, max(_as_int(info.round) - self.memory.current_round, 0))
            if prefix > 0:
                child.dis_vals[:prefix] = node.dis_vals[:prefix]
            child.actions = list(op_sequence)
            child_info = child.sim.info
            for operation in op_sequence:
                if operation.op_type == OperationType.DOWNGRADE_TOWER:
                    tower = _find_tower_by_id(child_info, operation.arg0)
                    if tower is not None:
                        if _tower_kind(tower) == TowerType.BASIC:
                            child.loss += int(_as_int(child_info.build_tower_cost(_as_int(child_info.tower_num_of_player(self.side)))) * 0.2)
                        else:
                            child.loss += int(_as_int(child_info.upgrade_tower_cost(int(_tower_kind(tower)))) * 0.2)
                child.sim.add_operation_of_player(self.side, operation)
            child.sim.apply_operations_of_player(self.side)
            child_value = self._evaluate(child)
            if child_value > node.max_val:
                node.max_val = child_value
                node.max_expand = node.expand_count + 1
            self.nodes.append(child)
            node.children.append(child.node_id)

        if is_root:
            node.sim.fast_next_round(self.side)
        node.expand_count += 1

    def _select_expand(self) -> bool:
        root = self.nodes[0]
        if not root.children:
            return False
        target_id = -1
        best_value = -1e18
        for child_id in root.children:
            child = self.nodes[child_id]
            value = -child.expand_count
            if child_id == 0:
                value += self.memory.reserved_bias
            if not child.children:
                value += 1000
            if child.danger:
                value += 20
            if not child.safe:
                value -= 20
            if value > best_value:
                best_value = value
                target_id = child_id
        if target_id < 0:
            return False
        self._expand(self.nodes[target_id])
        return True

    def _try_sell_all(self, coins: int, towers: int, coin_need: int, info) -> tuple[int, int, list[Operation]]:
        operations: list[Operation] = []
        max_coins = coins
        valid_tower_num = 0
        for tower in info.towers:
            if _as_int(tower.player) != self.side or info.is_shielded_by_emp(self.side, _as_int(tower.x), _as_int(tower.y)):
                continue
            tower_type = _tower_kind(tower)
            if tower_type == TowerType.BASIC:
                coins += _as_int(info.destroy_tower_income(towers))
                towers -= 1
            else:
                coins += _as_int(info.downgrade_tower_income(int(tower_type)))
                if tower_type.value // 10 == 0:
                    max_coins += 48
                else:
                    max_coins += 48 + 160
            operations.append(_sdk_operation(OperationType.DOWNGRADE_TOWER, _as_int(tower.id)))
            valid_tower_num += 1
            if coins >= coin_need:
                return coins, towers, operations
        max_coins += (2 ** _as_int(info.tower_num_of_player(self.side)) - 2 ** (_as_int(info.tower_num_of_player(self.side)) - valid_tower_num)) * 12
        if max_coins >= coin_need:
            return coins, towers, operations
        return coins, towers, []

    def _try_sell(self, coins: int, towers: int, coin_need: int, info) -> tuple[int, int, list[Operation]]:
        tower_ids = [
            _as_int(tower.id)
            for tower in info.towers
            if _as_int(tower.player) == self.side and not info.is_shielded_by_emp(self.side, _as_int(tower.x), _as_int(tower.y))
        ]
        valid_count = len(tower_ids)
        if valid_count == 0:
            return coins, towers, []

        sim = Simulator(info)
        fail_round = 48
        for turn in range(1, 49):
            if not sim.fast_next_round(self.side):
                break
            if _as_int(sim.info.bases[self.side].hp) < _as_int(info.bases[self.side].hp):
                fail_round = turn
                break

        max_round = -1
        max_coins = coins
        best_operations: list[Operation] = []
        from itertools import permutations

        for ordering in permutations(range(valid_count)):
            operations: list[Operation] = []
            new_sim = Simulator(info)
            new_info = new_sim.info
            new_coins = coins
            new_towers = towers
            valid = False
            for position in ordering:
                tower_id = tower_ids[position]
                tower = _find_tower_by_id(new_info, tower_id)
                if tower is None:
                    continue
                if _tower_kind(tower) == TowerType.BASIC:
                    new_coins += _as_int(new_info.destroy_tower_income(new_towers))
                    new_towers -= 1
                else:
                    new_coins += _as_int(new_info.downgrade_tower_income(int(_tower_kind(tower))))
                operations.append(_sdk_operation(OperationType.DOWNGRADE_TOWER, tower_id))
                if new_coins >= coin_need:
                    valid = True
                    break
            if not valid:
                continue
            for operation in operations:
                new_sim.add_operation_of_player(self.side, operation)
            new_sim.apply_operations_of_player(self.side)
            value = 48
            base_hp = _as_int(new_info.bases[self.side].hp)
            for turn in range(1, 49):
                if not new_sim.fast_next_round(self.side):
                    break
                if _as_int(new_info.bases[self.side].hp) < base_hp:
                    value = turn
                    break
            if value > max_round:
                max_round = value
                max_coins = new_coins
                best_operations = operations

        if max_round < min(24, fail_round):
            return coins, towers, []
        return max_coins, towers, best_operations

    def _try_use_storm(self, info, all_in: bool) -> list[Operation]:
        operations: list[Operation] = []
        if _as_int(info.super_weapon_cd[self.side][int(SuperWeaponType.LIGHTNING_STORM)]) > 0:
            return []
        storm_cost = _as_int(info.use_super_weapon_cost(int(SuperWeaponType.LIGHTNING_STORM)))
        use = _as_int(info.coins[self.side]) >= storm_cost
        coins = _as_int(info.coins[self.side])
        towers = _as_int(info.tower_num_of_player(self.side))
        if not use:
            if all_in:
                coins, towers, operations = self._try_sell_all(coins, towers, storm_cost, info)
            else:
                coins, towers, operations = self._try_sell(coins, towers, storm_cost, info)
            use = bool(operations) and coins >= storm_cost
        if not use:
            return []

        best_score = -1
        best_target: tuple[int, int] | None = None
        for x in range(19):
            for y in range(19):
                if not is_valid_pos(x, y):
                    continue
                sim = Simulator(info)
                for operation in operations:
                    sim.add_operation_of_player(self.side, operation)
                sim.add_operation_of_player(self.side, _sdk_operation(OperationType.USE_LIGHTNING_STORM, x, y))
                sim.apply_operations_of_player(self.side)
                fail_round = 32
                for turn in range(32):
                    if not sim.fast_next_round(self.side):
                        break
                    if _as_int(sim.info.bases[self.side].hp) < _as_int(info.bases[self.side].hp):
                        fail_round = turn
                        break
                if fail_round < 24:
                    continue
                score = _as_int(sim.info.die_count[self.enemy]) + fail_round
                if score > best_score:
                    best_score = score
                    best_target = (x, y)
        if best_target is None:
            return []
        return operations + [_sdk_operation(OperationType.USE_LIGHTNING_STORM, best_target[0], best_target[1])]

    def _try_end_storm(self, info) -> list[Operation]:
        if _as_int(info.super_weapon_cd[self.side][int(SuperWeaponType.LIGHTNING_STORM)]) > 0:
            return []
        storm_cost = _as_int(info.use_super_weapon_cost(int(SuperWeaponType.LIGHTNING_STORM)))
        coins = _as_int(info.coins[self.side])
        towers = _as_int(info.tower_num_of_player(self.side))
        operations: list[Operation] = []
        use = coins >= storm_cost
        if not use:
            coins, towers, operations = self._try_sell_all(coins, towers, storm_cost, info)
            use = bool(operations) and coins >= storm_cost
        if not use:
            return []
        storm_x, storm_y = _slot(self.side, STORM_SLOT)
        return operations + [_sdk_operation(OperationType.USE_LIGHTNING_STORM, storm_x, storm_y)]

    def _choose_superweapon(self, info) -> list[Operation]:
        operations: list[Operation] = []
        coins = _as_int(info.coins[self.side])
        towers = _as_int(info.tower_num_of_player(self.side))
        can_emp = _as_int(info.super_weapon_cd[self.side][int(SuperWeaponType.EMP_BLASTER)]) == 0 and coins >= _as_int(info.use_super_weapon_cost(int(SuperWeaponType.EMP_BLASTER)))
        can_deflect = _as_int(info.super_weapon_cd[self.side][int(SuperWeaponType.DEFLECTOR)]) == 0 and coins >= _as_int(info.use_super_weapon_cost(int(SuperWeaponType.DEFLECTOR)))
        can_evasion = _as_int(info.super_weapon_cd[self.side][int(SuperWeaponType.EMERGENCY_EVASION)]) == 0 and coins >= _as_int(info.use_super_weapon_cost(int(SuperWeaponType.EMERGENCY_EVASION)))
        enemy_storm = _as_int(info.super_weapon_cd[self.enemy][int(SuperWeaponType.LIGHTNING_STORM)]) == 0 and _as_int(info.coins[self.enemy]) >= _as_int(info.use_super_weapon_cost(int(SuperWeaponType.LIGHTNING_STORM)))

        if not can_emp and _as_int(info.super_weapon_cd[self.side][int(SuperWeaponType.EMP_BLASTER)]) == 0:
            coins, towers, operations = self._try_sell(coins, towers, 150, info)
        if not operations and (
            (_as_int(info.super_weapon_cd[self.side][int(SuperWeaponType.DEFLECTOR)]) == 0 and not can_deflect)
            or (_as_int(info.super_weapon_cd[self.side][int(SuperWeaponType.EMERGENCY_EVASION)]) == 0 and not can_evasion)
        ):
            coins, towers, operations = self._try_sell(coins, towers, 100, info)

        can_emp = _as_int(info.super_weapon_cd[self.side][int(SuperWeaponType.EMP_BLASTER)]) == 0 and coins >= _as_int(info.use_super_weapon_cost(int(SuperWeaponType.EMP_BLASTER)))
        can_deflect = _as_int(info.super_weapon_cd[self.side][int(SuperWeaponType.DEFLECTOR)]) == 0 and coins >= _as_int(info.use_super_weapon_cost(int(SuperWeaponType.DEFLECTOR)))
        can_evasion = _as_int(info.super_weapon_cd[self.side][int(SuperWeaponType.EMERGENCY_EVASION)]) == 0 and coins >= _as_int(info.use_super_weapon_cost(int(SuperWeaponType.EMERGENCY_EVASION)))

        sim = Simulator(info)
        for _ in range(24):
            if not sim.fast_next_round(self.enemy):
                break
        base_enemy_hp = _as_int(sim.info.bases[self.enemy].hp)
        base_die_count = _as_int(sim.info.die_count[self.side])
        deferred_emp: list[tuple[int, int, float]] = []

        if can_emp:
            results: list[tuple[int, int, float]] = []
            for x in range(19):
                for y in range(19):
                    if not is_valid_pos(x, y):
                        continue
                    value = 0.0
                    for tower in info.towers:
                        if _as_int(tower.player) == self.enemy and hex_distance(_as_int(tower.x), _as_int(tower.y), x, y) <= 3:
                            if _tower_kind(tower) == TowerType.BASIC:
                                value += 50
                            elif _tower_kind(tower).value // 10 < 0:
                                value += 60
                            else:
                                value += 80
                    if value < 100:
                        continue
                    new_sim = Simulator(info)
                    for operation in operations:
                        new_sim.add_operation_of_player(self.side, operation)
                    new_sim.add_operation_of_player(self.side, _sdk_operation(OperationType.USE_EMP_BLASTER, x, y))
                    new_sim.apply_operations_of_player(self.side)
                    for _ in range(24):
                        if not new_sim.fast_next_round(self.enemy):
                            break
                    enemy_hp = _as_int(new_sim.info.bases[self.enemy].hp)
                    if self.memory.current_round > 495:
                        if enemy_hp >= base_enemy_hp:
                            continue
                    elif self.memory.current_round > 460:
                        if enemy_hp >= base_enemy_hp - 2:
                            continue
                    elif enemy_hp >= base_enemy_hp - 4:
                        continue
                    value += 100 * (base_enemy_hp - enemy_hp)
                    enemy_base_x, enemy_base_y = _slot(self.enemy, BASE_SLOT)
                    for code in range(1, 34):
                        sx, sy = _slot(self.enemy, code)
                        if hex_distance(sx, sy, x, y) <= 3:
                            value += 3 - hex_distance(sx, sy, enemy_base_x, enemy_base_y) * 0.01
                    results.append((x, y, value))
            if results and not enemy_storm:
                best = self._best_value(results, 2)
                assert best is not None
                self.memory.last_superweapon_round = self.memory.current_round
                self.memory.last_superweapon_type = int(SuperWeaponType.EMP_BLASTER)
                return operations + [_sdk_operation(OperationType.USE_EMP_BLASTER, best[0], best[1])]
            deferred_emp = results

        if can_deflect or can_evasion:
            results: list[tuple[int, int, float, bool]] = []
            if can_evasion:
                for x in range(19):
                    for y in range(19):
                        if not is_valid_pos(x, y):
                            continue
                        value = 0.0
                        evasions = 0
                        min_distance = 100
                        for ant in info.ants:
                            if _as_int(ant.player) == self.side and hex_distance(_as_int(ant.x), _as_int(ant.y), x, y) <= 3 and ant.is_alive():
                                value += _as_int(ant.level) + 1
                                evasions += 1
                                distance = hex_distance(_as_int(ant.x), _as_int(ant.y), *_slot(self.enemy, BASE_SLOT))
                                if distance < min_distance:
                                    min_distance = distance
                        if self.memory.current_round <= 506 and min_distance > 5:
                            continue
                        if evasions < 3 or (self.memory.current_round > 460 and evasions < 2):
                            continue
                        new_sim = Simulator(info)
                        for operation in operations:
                            new_sim.add_operation_of_player(self.side, operation)
                        new_sim.add_operation_of_player(self.side, _sdk_operation(OperationType.USE_EMERGENCY_EVASION, x, y))
                        new_sim.apply_operations_of_player(self.side)
                        for _ in range(24):
                            if not new_sim.fast_next_round(self.enemy):
                                break
                        enemy_hp = _as_int(new_sim.info.bases[self.enemy].hp)
                        if self.memory.current_round > 506:
                            if enemy_hp >= base_enemy_hp and _as_int(new_sim.info.die_count[self.side]) >= base_die_count - 2:
                                continue
                        elif self.memory.current_round > 460:
                            if enemy_hp >= base_enemy_hp - 2:
                                continue
                        elif enemy_hp >= base_enemy_hp - 3:
                            continue
                        value += 100 * (base_enemy_hp - enemy_hp)
                        results.append((x, y, value, True))
            if can_deflect and not results:
                storm_x, storm_y = _slot(self.enemy, STORM_SLOT)
                for x in range(19):
                    for y in range(19):
                        if not is_valid_pos(x, y):
                            continue
                        if hex_distance(x, y, *_slot(self.enemy, BASE_SLOT)) > 4:
                            continue
                        new_sim = Simulator(info)
                        for operation in operations:
                            new_sim.add_operation_of_player(self.side, operation)
                        new_sim.add_operation_of_player(self.side, _sdk_operation(OperationType.USE_DEFLECTOR, x, y))
                        new_sim.apply_operations_of_player(self.side)
                        for _ in range(24):
                            if not new_sim.fast_next_round(self.enemy):
                                break
                        enemy_hp = _as_int(new_sim.info.bases[self.enemy].hp)
                        if (self.memory.current_round > 460 and enemy_hp >= base_enemy_hp - 2) or enemy_hp >= base_enemy_hp - 3:
                            continue
                        value = 100 * (base_enemy_hp - enemy_hp) - hex_distance(x, y, storm_x, storm_y)
                        results.append((x, y, value, False))
            if results:
                best = self._best_value(results, 2)
                assert best is not None
                self.memory.last_superweapon_round = self.memory.current_round
                if best[3]:
                    self.memory.last_superweapon_type = int(SuperWeaponType.EMERGENCY_EVASION)
                    return operations + [_sdk_operation(OperationType.USE_EMERGENCY_EVASION, best[0], best[1])]
                self.memory.last_superweapon_type = int(SuperWeaponType.DEFLECTOR)
                return operations + [_sdk_operation(OperationType.USE_DEFLECTOR, best[0], best[1])]

        if can_emp and deferred_emp:
            best = self._best_value(deferred_emp, 2)
            assert best is not None
            self.memory.last_superweapon_round = self.memory.current_round
            self.memory.last_superweapon_type = int(SuperWeaponType.EMP_BLASTER)
            return operations + [_sdk_operation(OperationType.USE_EMP_BLASTER, best[0], best[1])]
        return []

    def _try_emp(self, info) -> list[Operation]:
        if _enemy_front_distance(self.side, info) > 5:
            return []
        if _as_int(info.super_weapon_cd[self.side][int(SuperWeaponType.EMP_BLASTER)]) > 0:
            return []
        coins = _as_int(info.coins[self.side])
        towers = _as_int(info.tower_num_of_player(self.side))
        enemy_coins = _as_int(info.coins[self.enemy])
        operations: list[Operation] = []
        if coins - enemy_coins < 100 or coins < 150:
            coins, towers, operations = self._try_sell(coins, towers, max(enemy_coins + 100, 150), info)
            if not operations:
                return []

        my_sim = Simulator(info)
        for _ in range(24):
            if not my_sim.fast_next_round(self.side):
                break
            if _as_int(my_sim.info.bases[self.side].hp) < _as_int(info.bases[self.side].hp):
                return []

        enemy_sim = Simulator(info)
        for _ in range(24):
            if not enemy_sim.fast_next_round(self.enemy):
                break
        base_enemy_hp = _as_int(enemy_sim.info.bases[self.enemy].hp)

        results: list[tuple[int, int, float]] = []
        for x in range(19):
            for y in range(19):
                if not is_valid_pos(x, y):
                    continue
                value = 0.0
                for tower in info.towers:
                    if _as_int(tower.player) == self.enemy and hex_distance(_as_int(tower.x), _as_int(tower.y), x, y) <= 3:
                        if _tower_kind(tower) == TowerType.BASIC:
                            value += 50
                        elif _tower_kind(tower).value // 10 < 0:
                            value += 60
                        else:
                            value += 80
                if value < 100:
                    continue
                new_sim = Simulator(info)
                for operation in operations:
                    new_sim.add_operation_of_player(self.side, operation)
                new_sim.add_operation_of_player(self.side, _sdk_operation(OperationType.USE_EMP_BLASTER, x, y))
                new_sim.apply_operations_of_player(self.side)
                for _ in range(24):
                    if not new_sim.fast_next_round(self.enemy):
                        break
                enemy_hp = _as_int(new_sim.info.bases[self.enemy].hp)
                if enemy_hp >= base_enemy_hp - 4:
                    continue
                value += 100 * (base_enemy_hp - enemy_hp)
                enemy_base_x, enemy_base_y = _slot(self.enemy, BASE_SLOT)
                for code in range(1, 34):
                    sx, sy = _slot(self.enemy, code)
                    if hex_distance(sx, sy, x, y) <= 3:
                        value += 3 - hex_distance(sx, sy, enemy_base_x, enemy_base_y) * 0.01
                results.append((x, y, value))
        if results:
            best = self._best_value(results, 2)
            assert best is not None
            self.memory.last_superweapon_round = self.memory.current_round
            self.memory.last_superweapon_type = int(SuperWeaponType.EMP_BLASTER)
            return operations + [_sdk_operation(OperationType.USE_EMP_BLASTER, best[0], best[1])]
        return []

    def _try_attack(self, info) -> list[Operation]:
        if self.memory.mode == 0:
            return self._choose_superweapon(info)

        if self.memory.current_round <= 460:
            if _as_int(info.bases[self.side].ant_level) == 0:
                if _as_int(info.coins[self.side]) >= 200:
                    return [_sdk_operation(OperationType.UPGRADE_GENERATED_ANT)]
            elif _as_int(info.bases[self.side].ant_level) == 1:
                coins = _as_int(info.coins[self.side])
                towers = _as_int(info.tower_num_of_player(self.side))
                if coins >= 250:
                    return [_sdk_operation(OperationType.UPGRADE_GENERATED_ANT)]
                coins, towers, sold = self._try_sell(coins, towers, 250, info)
                if sold:
                    return sold + [_sdk_operation(OperationType.UPGRADE_GENERATED_ANT)]
            elif _as_int(info.bases[self.side].gen_speed_level) == 0:
                coins = _as_int(info.coins[self.side])
                towers = _as_int(info.tower_num_of_player(self.side))
                if coins >= 200:
                    return [_sdk_operation(OperationType.UPGRADE_GENERATION_SPEED)]
                coins, towers, sold = self._try_sell_all(coins, towers, 200, info)
                if sold:
                    return sold + [_sdk_operation(OperationType.UPGRADE_GENERATION_SPEED)]
            return self._choose_superweapon(info)

        if self.memory.current_round <= 470 and _as_int(info.bases[self.side].ant_level) == 0:
            if _as_int(info.coins[self.side]) >= 200:
                return [_sdk_operation(OperationType.UPGRADE_GENERATED_ANT)]
            return []
        return self._choose_superweapon(info)

    def _run_search(self, info) -> list[Operation]:
        modified = info.clone()
        modified.set_base_hp(self.enemy, 1000)
        root = SearchNode(Simulator(modified))
        root.node_id = 0
        root.parent = -1
        self.nodes = [root]
        self._evaluate(root)
        self._expand(root, is_root=True)
        for _ in range(SEARCH_SELECT_BUDGET):
            if len(self.nodes) >= MAX_NODE_COUNT - 10:
                break
            if not self._select_expand():
                break

        max_id = -1
        max_val = -1e18
        for child_id in root.children:
            child = self.nodes[child_id]
            if child.max_val > max_val:
                max_val = child.max_val
                max_id = child_id

        if len(self.nodes) > 1 and max_id > 1 and max_val - self.nodes[1].max_val < 2:
            max_id = 1
            max_val = self.nodes[1].max_val

        if max_id < 0:
            return []

        enemy_emp = -1
        for effect in info.super_weapons:
            if _as_int(effect.player) == self.enemy and _as_int(effect.type) == int(SuperWeaponType.EMP_BLASTER):
                enemy_emp = _as_int(effect.left_time)
                break

        chosen = self.nodes[max_id]
        critical = (
            (self.memory.mode >= 0 and ((enemy_emp > 0 and chosen.fail_round - self.memory.current_round < min(8, enemy_emp) and max_val < -400) or (max_val < -700 and chosen.fail_round - self.memory.current_round <= 2)))
            or (self.memory.mode == 0 and _as_int(info.die_count[self.enemy]) - _as_int(info.die_count[self.side]) >= 8 and chosen.fail_round - self.memory.current_round <= 1)
        )
        if critical:
            emergency = self._try_use_storm(info, all_in=self.memory.current_round >= 480)
            if emergency:
                return emergency

        self.memory.reserved_bias = chosen.max_expand
        return list(chosen.actions)

    def choose(self, player: int, info) -> list[Operation]:
        self.memory.current_round = _as_int(info.round)
        self.memory.enemy_old_count = _as_int(info.old_count[self.enemy])
        self.memory.enemy_die_count = _as_int(info.die_count[self.enemy])
        if self.memory.current_round == 0:
            self.memory.side = player
        self.memory.current_hp = _as_int(info.bases[self.side].hp)

        self.memory.mode = 0
        if _as_int(info.bases[self.side].hp) > _as_int(info.bases[self.enemy].hp):
            self.memory.mode = 1
            self.memory.attack_stance = False
        elif _as_int(info.bases[self.side].hp) < _as_int(info.bases[self.enemy].hp):
            self.memory.mode = -1

        attack = self.memory.mode == -1
        scale = min(1.0, (512 - self.memory.current_round) / 20.0)
        enemy_strength = float(_as_int(info.die_count[self.enemy]))
        my_strength = float(_as_int(info.die_count[self.side]))
        for ant in info.ants:
            if _as_int(ant.player) == self.enemy and ant.is_alive():
                enemy_strength += scale
            elif _as_int(ant.player) == self.side and ant.is_alive():
                my_strength += scale

        if not attack and self.memory.mode == 0:
            diff = enemy_strength - my_strength
            if diff >= 4:
                self.memory.attack_stance = False
            elif diff <= -3 - max(_cxx_div(450 - self.memory.current_round, 50), 0):
                attack = True
            elif self.memory.attack_stance:
                attack = True
            elif self.memory.current_round >= 450 and diff <= 1:
                attack = True

        if self.memory.mode <= 0 and not self.memory.reserved_bias:
            opportunistic_emp = self._try_emp(info)
            if opportunistic_emp:
                return opportunistic_emp
        if attack and not self.memory.reserved_bias:
            self.memory.attack_stance = True
            aggressive = self._try_attack(info)
            if aggressive:
                return aggressive

        if self.memory.mode == 1 and self.memory.current_round >= 488:
            storm_finish = self._try_end_storm(info)
            if storm_finish:
                return storm_finish
        if self.memory.mode == 0 and self.memory.current_round >= 510:
            final_storm = self._try_use_storm(info, True)
            if final_storm:
                return final_storm

        return self._run_search(info)


class GreedyAgent(BaseAgent):
    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.core = GreedyPlanner()
        self._controller: GreedyController | None = None
        self._seed = 0 if seed is None else seed

    def on_match_start(self, player: int, seed: int) -> None:
        self._seed = seed
        self.core._prepare_new_match(player)
        self._controller = GreedyController.start(player, seed)

    def on_self_operations(self, operations) -> None:
        if self._controller is not None:
            self._controller.apply_self_operations(operations)

    def on_opponent_operations(self, operations) -> None:
        if self._controller is not None:
            self._controller.apply_opponent_operations(operations)

    def on_round_state(self, public_round_state) -> None:
        if self._controller is not None:
            self._controller.sync_public_round_state(public_round_state)

    def _decision_info(self, state, player: int) -> GameInfo:
        if self._controller is not None and self._controller.self_player_id == player:
            return self._controller.info
        if getattr(state, 'round_index', None) == 0 and self.core.memory.current_round != 0:
            self.core._prepare_new_match(player)
        return info_from_state(state, player=player, seed=self._seed)

    def choose_operations(self, state, player: int, bundles: list[ActionBundle] | None = None) -> list[Operation]:
        info = self._decision_info(state, player)
        return self.core.choose(player, info)

    def choose_bundle(self, state, player: int, bundles: list[ActionBundle] | None = None) -> ActionBundle:
        operations = tuple(self.choose_operations(state, player, bundles=bundles))
        bundles = bundles or self.list_bundles(state, player)
        target_signature = [tuple(op.to_protocol_tokens()) for op in operations]
        for bundle in bundles:
            if [tuple(op.to_protocol_tokens()) for op in bundle.operations] == target_signature:
                return bundle
        return ActionBundle(name='greedy', operations=operations, score=0.0, tags=('greedy',))


class AI(GreedyAgent):
    pass


@dataclass(slots=True)
class PhaseWeights:
    hp: float
    safety: float
    tempo: float
    offense: float
    economy: float


class GreedyHeuristicFallbackAgent(BaseAgent):
    def _phase_weights(self, state, player: int) -> PhaseWeights:
        hp_delta = state.bases[player].hp - state.bases[1 - player].hp
        nearest_enemy = state.nearest_ant_distance(player)
        safe_coin = state.coins[player] - state.safe_coin_threshold(player)
        if nearest_enemy <= 4 or hp_delta < 0:
            return PhaseWeights(hp=1.4, safety=1.5, tempo=0.5, offense=0.3, economy=0.4)
        if hp_delta > 0 and state.frontline_distance(player) <= 8:
            return PhaseWeights(hp=0.8, safety=0.4, tempo=1.0, offense=1.5, economy=0.8)
        if safe_coin < 0:
            return PhaseWeights(hp=1.0, safety=1.3, tempo=0.5, offense=0.2, economy=1.4)
        return PhaseWeights(hp=1.1, safety=1.0, tempo=1.0, offense=0.9, economy=0.9)

    def _predict_enemy_bundle(self, state, player: int) -> ActionBundle:
        enemy_bundles = self.list_bundles(state, 1 - player)
        return enemy_bundles[0] if not enemy_bundles else enemy_bundles[0]

    def _score_bundle(self, state, player: int, bundle: ActionBundle, enemy_bundle: ActionBundle) -> float:
        trial = state.clone()
        if player == 0:
            trial.resolve_turn(bundle.operations, enemy_bundle.operations)
        else:
            trial.resolve_turn(enemy_bundle.operations, bundle.operations)
        summary = self.feature_extractor.summarize(trial, player).named
        weights = self._phase_weights(state, player)
        score = bundle.score
        score += summary['hp_delta'] * 12.0 * weights.hp
        score += summary['safe_coin'] * 0.03 * weights.economy
        score += summary['frontline_advantage'] * 1.8 * weights.offense
        score -= summary['enemy_progress'] * 0.7 * weights.safety
        score += summary['my_progress'] * 0.5 * weights.offense
        score += summary['generation_level'] * 4.0 * weights.tempo
        score += summary['ant_level'] * 6.0 * weights.tempo
        score += summary['kill_delta'] * 2.5 * weights.hp
        score += summary['tower_spread'] * 0.8
        if bundle.tags and bundle.tags[0] == 'weapon' and state.coins[player] < state.safe_coin_threshold(player):
            score -= 8.0
        return score

    def choose_bundle(self, state, player: int, bundles: list[ActionBundle] | None = None) -> ActionBundle:
        bundles = bundles or self.list_bundles(state, player)
        enemy_bundle = self._predict_enemy_bundle(state, player)
        shortlist = bundles[: min(18, len(bundles))]
        scored = [(self._score_bundle(state, player, bundle, enemy_bundle), bundle) for bundle in shortlist]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1] if scored else bundles[0]
