from __future__ import annotations

from SDK.constants import OperationType, TowerType
from SDK.engine import GameState
from SDK.model import Ant, Operation, Tower


def test_initial_round_spawns_ants_and_advances_time() -> None:
    state = GameState.initial(seed=7)
    state.resolve_turn([], [])
    assert state.round_index == 1
    assert len(state.ants) == 2
    assert state.coins == [51, 51]


def test_build_and_upgrade_tower_updates_coin_and_state() -> None:
    state = GameState.initial(seed=3)
    build = Operation(OperationType.BUILD_TOWER, 6, 9)
    assert state.can_apply_operation(0, build)
    assert state.apply_operation_list(0, [build]) == []
    assert state.coins[0] == 35
    tower = state.tower_at(6, 9)
    assert tower is not None
    upgrade = Operation(OperationType.UPGRADE_TOWER, tower.tower_id, int(TowerType.HEAVY))
    state.coins[0] = 100
    assert state.can_apply_operation(0, upgrade)
    assert state.apply_operation_list(0, [upgrade]) == []
    assert tower.tower_type == TowerType.HEAVY
    assert state.coins[0] == 40


def test_quick_tower_attacks_enemy_ant() -> None:
    state = GameState.initial(seed=1)
    state.towers.append(Tower(0, 0, 6, 9, TowerType.QUICK, cooldown_clock=1.0))
    state.ants.append(Ant(0, 1, 8, 9, hp=10, level=0))
    state.advance_round()
    assert state.die_count[1] == 1 or any(ant.hp < 10 for ant in state.ants)


def test_emp_prevents_building_inside_field() -> None:
    state = GameState.initial(seed=1)
    state.active_effects.append(__import__('SDK.model', fromlist=['WeaponEffect']).WeaponEffect(__import__('SDK.constants', fromlist=['SuperWeaponType']).SuperWeaponType.EMP_BLASTER, 1, 6, 9, 3))
    blocked = Operation(OperationType.BUILD_TOWER, 6, 9)
    assert not state.can_apply_operation(0, blocked)
