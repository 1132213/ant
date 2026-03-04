from __future__ import annotations

from typing import List, Tuple

# Simple port of Ant class from ant_game - deploy/src/ant.cpp

_hp_list = [10, 25, 50]

# movement directions (same as map.py)
DIRECTIONS = [
    [(0, 1), (-1, 0), (0, -1), (1, -1), (1, 0), (1, 1)],
    [(-1, 1), (-1, 0), (-1, -1), (0, -1), (1, 0), (0, 1)],
]


class Ant:
    class Status:
        Alive = "alive"
        Success = "success"
        Fail = "fail"
        TooOld = "too_old"
        Frozen = "frozen"

    def __init__(self, player: int, id: int, x: int, y: int, level: int):
        self.player = player
        self.id = id
        self.pos = (x, y)
        self.level = level
        self.hp_limit = _hp_list[level]
        self.hp = _hp_list[level]
        self.path: List[int] = []
        self.age = 0
        self.is_frozen = False
        self.all_frozen = False
        self.shield = 0
        self.defend = False

    def get_status(self, base_camps: Tuple[Tuple[int, int], Tuple[int, int]]) -> str:
        if self.hp <= 0:
            return Ant.Status.Fail
        # success if reached opponent base camp
        target0 = base_camps[0]
        target1 = base_camps[1]
        if self.player and self.pos == target0:
            return Ant.Status.Success
        if not self.player and self.pos == target1:
            return Ant.Status.Success
        # rule: ants die when age > 32
        if self.age > 32:
            return Ant.Status.TooOld
        if self.is_frozen or self.all_frozen:
            return Ant.Status.Frozen
        return Ant.Status.Alive

    def increase_age(self) -> None:
        self.age += 1

    def move(self, direction: int) -> None:
        if direction == -1:
            return
        x, y = self.pos
        dx, dy = DIRECTIONS[y % 2][direction]
        self.pos = (x + dx, y + dy)
        self.path.append(direction)

    def set_hp(self, change: int) -> None:
        # emulate C++ logic with shield/defend
        if self.shield > 0:
            change = 0
            self.shield -= 1
        elif self.defend and change < 0 and (-change) * 2 < self.hp_limit:
            change = 0
        self.hp += change
        if self.hp > self.hp_limit:
            self.hp = self.hp_limit

    def set_hp_true(self, change: int) -> None:
        self.hp += change
