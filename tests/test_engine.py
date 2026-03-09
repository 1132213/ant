from __future__ import annotations

from SDK.constants import ANT_TELEPORT_INTERVAL, AntBehavior, OperationType, SuperWeaponType, TowerType
from SDK.engine import GameState
from SDK.model import Ant, Operation, Tower, WeaponEffect


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


def test_random_ant_degrades_to_default_after_five_rounds() -> None:
    ant = Ant(0, 0, 2, 9, hp=10, level=0, behavior=AntBehavior.RANDOM)
    state = GameState.initial(seed=3)
    state.ants.append(ant)
    for _ in range(5):
        state._increase_ant_age()
    assert ant.behavior == AntBehavior.DEFAULT


def test_ice_freeze_promotes_ant_to_random_after_thaw() -> None:
    state = GameState.initial(seed=2)
    ant = Ant(0, 1, 8, 9, hp=25, level=1, behavior=AntBehavior.CONSERVATIVE)
    tower = Tower(0, 0, 6, 9, TowerType.ICE, cooldown_clock=0.0)
    state.ants.append(ant)
    state._damage_ant_from_tower(tower, ant)
    assert ant.frozen
    state._prepare_ants_for_attack()
    assert ant.behavior == AntBehavior.RANDOM


def test_control_free_ant_ignores_control_and_teleport() -> None:
    state = GameState.initial(seed=9)
    immune = Ant(0, 1, 8, 9, hp=10, level=0, behavior=AntBehavior.CONTROL_FREE)
    target = Ant(1, 1, 9, 9, hp=10, level=0, behavior=AntBehavior.DEFAULT)
    state.ants.extend([immune, target])
    original = (immune.x, immune.y)
    state._control_ant(immune, AntBehavior.RANDOM)
    assert immune.behavior == AntBehavior.CONTROL_FREE
    state.round_index = ANT_TELEPORT_INTERVAL - 1
    state._teleport_ants()
    assert (immune.x, immune.y) == original


def test_lightning_and_emp_effects_drift_each_tick() -> None:
    state = GameState.initial(seed=11)
    state.active_effects = [
        WeaponEffect(SuperWeaponType.LIGHTNING_STORM, 0, 9, 9, 3),
        WeaponEffect(SuperWeaponType.EMP_BLASTER, 1, 10, 9, 3),
    ]
    before = [(effect.x, effect.y) for effect in state.active_effects]
    state._tick_effects()
    after = [(effect.x, effect.y) for effect in state.active_effects]
    assert len(after) == 2
    assert all(state.active_effects[index].remaining_turns == 2 for index in range(2))
    assert all(0 <= x < 19 and 0 <= y < 19 for x, y in after)
    assert before != after
