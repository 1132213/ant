from typing import List

from logic.gamedata import Direction
from logic.gamestate import GameState
from logic.constant import row, col
from logic.computation import compute_attack, compute_defence
from logic.map import PLAYER_0_BASE_CAMP, PLAYER_1_BASE_CAMP


def move_army_op(position: list[int], direction: Direction, num: int) -> List[int]:
    return [1, position[0], position[1], int(direction) + 1, num]


def _find_main(state: GameState, player: int):
    for g in state.generals:
        if type(g).__name__ == "MainGenerals" and g.player == player:
            return g
    return None


def policy(round_idx: int, my_seat: int, state: GameState) -> list[list[int]]:
    """Greedy baseline enhanced with ant interception.

    - 如果有敌方蚂蚁靠近自己的基地（距离 ≤2），
      即刻从主将处派出最多 3 军前往拦截。
    - 否则退回到原始的邻格贪心占领逻辑。
    """
    # 优先处理附近的敌蚂蚁
    base = PLAYER_0_BASE_CAMP if my_seat == 0 else PLAYER_1_BASE_CAMP
    for ant in getattr(state, "ants", []):
        if ant.player != my_seat:
            # 使用地图提供的距离函数
            dist = state.map.distance(ant.pos, base)
            if dist <= 2:
                g = _find_main(state, my_seat)
                if g:
                    x, y = g.position
                    if ant.pos[0] < x:
                        d = Direction.UP
                    elif ant.pos[0] > x:
                        d = Direction.DOWN
                    elif ant.pos[1] < y:
                        d = Direction.LEFT
                    else:
                        d = Direction.RIGHT
                    army_here = state.board[x][y].army
                    if army_here > 1:
                        num = min(3, army_here - 1)
                        return [move_army_op([x, y], d, num), [8]]
    # 回退到原始贪心占领逻辑
    ops: list[list[int]] = []
    g = _find_main(state, my_seat)
    if not g:
        return [[8]]
    x, y = g.position
    moves = min(2, state.rest_move_step[my_seat])
    dirs = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
    for _ in range(moves):
        best = None
        best_vs = 0.0
        army_here = state.board[x][y].army
        if army_here <= 1:
            break
        for d in dirs:
            nx = x + (-1 if d == Direction.UP else 1 if d == Direction.DOWN else 0)
            ny = y + (-1 if d == Direction.LEFT else 1 if d == Direction.RIGHT else 0)
            if nx < 0 or nx >= row or ny < 0 or ny >= col:
                continue
            dest = state.board[nx][ny]
            if int(dest.type) == 2:  # avoid mountains (no tech)
                continue
            atk = compute_attack(state.board[x][y], state)
            dfn = compute_defence(dest, state)
            for num in range(1, min(3, army_here - 1) + 1):
                vs = num * atk - dest.army * dfn
                if dest.player != my_seat and vs > best_vs:
                    best_vs = vs
                    best = (d, num)
            if dest.player == -1 and dest.army == 0 and best is None:
                best = (d, 1)
                best_vs = 1.0
        if best is None:
            break
        d, num = best
        ops.append(move_army_op([x, y], d, int(num)))
        break
    ops.append([8])
    return ops

def ai_func(state: GameState) -> list[list[int]]:
    """
    Lightweight adapter for controller-style callers.
    Maps a GameState to a list of commands, ending with [8].
    Assumes acting as player 0 and uses state's current round if available.
    """
    round_idx = getattr(state, "round", 0) + 1
    return policy(round_idx, 0, state)
