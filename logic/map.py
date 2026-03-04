from __future__ import annotations

from typing import List, Tuple
from logic.ant import Ant

# ported from ant_game - deploy/include/map.h and src/map.cpp

SIDE_LENGTH = 10
MAP_SIZE = 2 * SIDE_LENGTH - 1

# positions of the two players' base camps (used by ant movement and status)
PLAYER_0_BASE_CAMP = (2, SIDE_LENGTH - 1)
PLAYER_1_BASE_CAMP = (MAP_SIZE - 3, SIDE_LENGTH - 1)

# movement directions for hex grid depending on row parity
DIRECTIONS = [
    [(0, 1), (-1, 0), (0, -1), (1, -1), (1, 0), (1, 1)],
    [(-1, 1), (-1, 0), (-1, -1), (0, -1), (1, 0), (0, 1)],
]

# blocks that are invalid in the original map shape (copied from C++ constructor)
_invalid_blocks = {
    (6, 1), (7, 1), (9, 1), (11, 1),
    (12, 1), (4, 2), (6, 2), (8, 2),
    (9, 2), (11, 2), (13, 2), (4, 3),
    (5, 3), (13, 3), (14, 3), (6, 4),
    (8, 4), (9, 4), (11, 4), (3, 5),
    (4, 5), (7, 5), (9, 5), (11, 5),
    (14, 5), (15, 5), (3, 6), (5, 6),
    (12, 6), (14, 6), (2, 7), (5, 7),
    (6, 7), (8, 7), (9, 7), (10, 7),
    (12, 7), (13, 7), (16, 7), (1, 8),
    (2, 8), (7, 8), (10, 8), (15, 8),
    (16, 8), (0, 9), (4, 9), (5, 9),
    (6, 9), (9, 9), (12, 9), (13, 9),
    (14, 9), (18, 9), (1, 10), (2, 10),
    (7, 10), (10, 10), (15, 10), (16, 10),
    (2, 11), (5, 11), (6, 11), (8, 11),
    (9, 11), (10, 11), (12, 11), (13, 11),
    (16, 11), (3, 12), (5, 12), (12, 12),
    (14, 12), (3, 13), (4, 13), (7, 13),
    (9, 13), (11, 13), (14, 13), (15, 13),
    (6, 14), (8, 14), (9, 14), (11, 14),
    (4, 15), (5, 15), (13, 15), (14, 15),
    (4, 16), (6, 16), (8, 16), (9, 16),
    (11, 16), (13, 16), (6, 17), (7, 17),
    (9, 17), (11, 17), (12, 17)
}


class Map:
    """Hexagonal map helper with validity, pheromone and ant movement.

    This is a light port of the C++ Map class from the ant_game deploy
    subproject.  It provides:

    * :func:`is_valid` -- whether a coordinate belongs to the playable
      hex shape.
    * :func:`distance` -- hex distance used by ants.
    * :func:`get_move` -- select a direction for an ant based on
      pheromone values and heuristic.
    * pheromone grid and decay logic used by :func:`next_round`.
    """

    def __init__(self) -> None:
        # map of pheromone values: [player][x][y]
        # initialize to TAU_BASE (10.0) to match C++ code
        self.pheromone = [
            [[10.0 for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)],
            [[10.0 for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)],
        ]

    def is_valid(self, x: int, y: int) -> bool:
        if x < 0 or x >= MAP_SIZE or y < 0 or y >= MAP_SIZE:
            return False
        if (x, y) in _invalid_blocks:
            return False
        return True

    def distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Compute hex distance between two positions."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if dy % 2 == 1:
            if a[0] > b[0]:
                dx = max(0, dx - dy // 2 - (a[1] % 2))
            else:
                dx = max(0, dx - dy // 2 - (1 - (a[1] % 2)))
        else:
            dx = max(0, dx - dy // 2)
        return dx + dy

    def next_round(self) -> None:
        """Decay pheromone globally (lambda factor from C++)."""
        LAMBDA = 0.97
        TAU_BASE = 10.0
        for i in range(MAP_SIZE):
            for j in range(MAP_SIZE):
                for p in range(2):
                    self.pheromone[p][i][j] = (
                        LAMBDA * self.pheromone[p][i][j]
                        + (1.0 - LAMBDA) * TAU_BASE
                    )

    def get_move(self, ant: "Ant", des: Tuple[int, int]) -> int:
        """Choose next direction for *ant* towards *des*.

        This is a direct port of Map::get_move from the C++ version of the
        game.  It avoids backtracking, respects validity and uses the
        pheromone/eta heuristic.
        """
        x, y = ant.pos
        player = ant.player
        pvals = [-1.0] * 6

        for i in range(6):
            nx = x + DIRECTIONS[y % 2][i][0]
            ny = y + DIRECTIONS[y % 2][i][1]
            if ant.path and ant.path[-1] == ((i + 3) % 6):
                pvals[i] = -1.0
            elif not self.is_valid(nx, ny):
                pvals[i] = -1.0
            else:
                pvals[i] = self.pheromone[player][nx][ny]
            # multiply with eta
            if pvals[i] >= 0:
                if self.distance((nx, ny), des) < self.distance((x, y), des):
                    m = 1.25
                elif self.distance((nx, ny), des) == self.distance((x, y), des):
                    m = 1.0
                else:
                    m = 0.75
                pvals[i] *= m
        best = -1
        bestval = -0.1
        for i in range(6):
            if pvals[i] > bestval:
                bestval = pvals[i]
                best = i
        return best

    def update_pheromone(self, ant: "Ant") -> None:
        """Increment pheromone along the path of a finished ant.

        A direct translation of the C++ Map::update_pheromone method.
        """
        # constants from C++ implementation
        Q1 = 10.0
        Q2 = -5.0
        Q3 = -3.0
        status = ant.get_status((PLAYER_0_BASE_CAMP, PLAYER_1_BASE_CAMP))
        if status == Ant.Status.Success:
            Q = Q1
        elif status == Ant.Status.Fail:
            Q = Q2
        elif status == Ant.Status.TooOld:
            Q = Q3
        else:
            return

        x, y = ant.pos
        # update starting cell
        self.pheromone[ant.player][x][y] = max(0.0, self.pheromone[ant.player][x][y] + Q)
        # traverse path backwards (excluding the move that led here)
        for mov in reversed(ant.path):
            if mov == -1:
                continue
            # step backwards using opposite direction
            inv = (mov + 3) % 6
            dx, dy = DIRECTIONS[y % 2][inv]
            x += dx
            y += dy
            if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                self.pheromone[ant.player][x][y] = max(0.0, self.pheromone[ant.player][x][y] + Q)


# expose convenient helpers

def distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return Map().distance(a, b)
