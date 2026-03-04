# 本文件定义了游戏状态类，以及负责初始化将军，更新回合的函数
import random
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from logic.call_generals import call_generals
from logic.constant import *
from logic.gamedata import (
    Cell,
    CellType,
    Direction,
    Farmer,
    Generals,
    MainGenerals,
    SkillType,
    SubGenerals,
    SuperWeapon,
    WeaponType,
    init_coin,
)
from logic.ant import Ant
from logic.map import Map, PLAYER_0_BASE_CAMP, PLAYER_1_BASE_CAMP
from logic.general_skills import skill_activate
from logic.generate_round_replay import get_single_round_replay
from logic.movement import army_move, general_move
from logic.super_weapons import *
from logic.upgrade import *


# =========================
# AntWar-style replay writer (JSON array)
# - seed only appears in the first element
# =========================
class AntReplayWriter:
    def __init__(self, path: str, seed: int):
        self.fp = open(path, "w", encoding="utf-8")
        self.first = True
        self.seed = int(seed)
        self.seed_written = False
        self.fp.write("[\n")

    def append(self, frame: dict) -> None:
        if not self.seed_written:
            frame = dict(frame)
            frame["seed"] = self.seed
            self.seed_written = True

        if not self.first:
            self.fp.write(",\n")
        self.fp.write(json.dumps(frame, ensure_ascii=False))
        self.first = False
        self.fp.flush()

    def close(self) -> None:
        self.fp.write("\n]\n")
        self.fp.close()


@dataclass
class GameState:
    replay_file: str = "default_replay.json"
    round: int = 1  # 当前游戏回合数
    generals: list[Generals] = field(default_factory=list)  # 游戏中的将军列表，用于通信
    coin: list[int] = field(default_factory=lambda: [init_coin() for p in range(2)])  # 每个玩家的金币数量列表
    active_super_weapon: list[SuperWeapon] = field(default_factory=list)
    super_weapon_unlocked: list[bool] = field(default_factory=lambda: [False, False])
    super_weapon_cd: list[int] = field(default_factory=lambda: [-1, -1])
    tech_level: list[list[int]] = field(default_factory=lambda: [[2, 0, 0, 0], [2, 0, 0, 0]])
    # 科技等级列表，第一层对应玩家一，玩家二，第二层分别对应行动力，攀岩，免疫沼泽，超级武器
    rest_move_step: list[int] = field(default_factory=lambda: [2, 2])

    board: list[list[Cell]] = field(default_factory=lambda: [[Cell(position=[i, j]) for j in range(col)] for i in range(row)])
    changed_cells: list[list[int]] = field(default_factory=lambda: [])
    next_generals_id: int = 0
    winner: int = -1

    # ant-specific state (port of ant_game logic)
    base_hp: list[int] = field(default_factory=lambda: [50, 50])
    # track number of enemy ants killed by each player (for tiebreaker)
    kill_count: list[int] = field(default_factory=lambda: [0, 0])
    # record how many times each player has used a super weapon
    superweapon_used: list[int] = field(default_factory=lambda: [0, 0])
    ants: list[Ant] = field(default_factory=list)
    _ant_id: int = field(default=0, init=False, repr=False)
    map: Map = field(default_factory=Map)

    # ---- replay support ----
    replay_seed: int = 0
    _ant_replay: Optional[AntReplayWriter] = field(default=None, init=False, repr=False)

    # store last ops of each player for the current full round
    _last_ops: list[list[list[int]]] = field(default_factory=lambda: [[], []], init=False, repr=False)

    # tower delta tracking (so towers list can be "delta-like")
    _prev_towers_snapshot: Dict[int, tuple] = field(default_factory=dict, init=False, repr=False)

    # ----- ant/pheromone helpers (not part of official rules) -----
    # map from destination coordinate to last move direction (int)
    _ant_moves: Dict[tuple[int, int], int] = field(default_factory=dict, init=False, repr=False)
    # 2×row×col pheromone grid used by frontends for pathfinding
    _pheromone: Optional[list] = field(default=None, init=False, repr=False)
    _lcg_seed: int = field(default=0, init=False, repr=False)
    # track last round when per-10-turn army bonus or full-round coin was granted
    _last_bonus_round: int = field(default=0, init=False, repr=False)

    def _lcg(self) -> int:
        # linear congruential generator matching game.md snippet
        self._lcg_seed = (25214903917 * self._lcg_seed) & ((1 << 48) - 1)
        return self._lcg_seed

    def _init_pheromon(self, seed: int) -> None:
        self._lcg_seed = int(seed)
        self._pheromone = [[[0.0 for _ in range(col)] for _ in range(row)] for _ in range(2)]
        if self._pheromone is None:
            return
        for i in range(2):
            for j in range(row):
                for k in range(col):
                    self._pheromone[i][j][k] = self._lcg() * (2 ** -46) + 8

    def replay_open(self, seed: int) -> None:
        self.replay_seed = int(seed)
        self._ant_replay = AntReplayWriter(self.replay_file, self.replay_seed)
        # initialize pheromone map so frontend can show a tourable background
        try:
            self._init_pheromon(seed)
        except Exception:
            pass

    def replay_close(self) -> None:
        if self._ant_replay is not None:
            self._ant_replay.close()
            self._ant_replay = None

    def set_last_ops(self, player: int, ops: list[list[int]]) -> None:
        if player in (0, 1):
            self._last_ops[player] = ops

    def find_general_position_by_id(self, general_id: int):
        for gen in self.generals:
            if gen.id == general_id:
                return gen.position
        return None

    def trans_state_to_init_json(self, player):
        """
        这是你现有 AI/前端使用的 JSON rep。
        保持不变。
        """
        result = get_single_round_replay(
            self, [[int(i / col), i % col] for i in range(row * col)], player, [8]
        )
        cell_type = ""
        for i in range(row * col):
            cell_type += str(int(self.board[int(i / col)][i % col].type))
        result["Cell_type"] = cell_type
        return result

    # =========================
    # AntWar replay: op + round_state mapping
    # =========================
    def _op_to_ant_op(self, op: list[int]) -> dict:
        """
        AntWar replay op schema requires:
          {"args":..., "id":..., "pos":{"x","y"}, "type":...}
        这里做 best-effort 映射：
        - 若 op 至少有 3 个数，则把 op[1],op[2] 当作 pos
        - 若 op 至少有 2 个数，则把 op[1] 当作 id
        """
        t = int(op[0]) if op else -1
        args = -1
        tid = -1
        pos = {"x": -1, "y": -1}

        if len(op) >= 2:
            tid = int(op[1])
        if len(op) >= 3:
            pos["x"], pos["y"] = int(op[1]), int(op[2])

        return {"args": args, "id": tid, "pos": pos, "type": t}

    def _build_pheromone(self) -> list:
        """
        pheromone: [2][H][W]
        Frontend uses this map for pathfinding/rendering. we maintain a seeded
        random baseline (initialized in replay_open) and optionally bump values
        by army strength so that occupied cells attract ants.
        """
        # make sure we have something to return even if init failed
        if self._pheromone is None:
            # fallback to the old army-based occupancy
            target_h = max(row, 19)
            target_w = max(col, 19)
            pheromone = []
            for p in (0, 1):
                grid = []
                for i in range(target_h):
                    line = []
                    for j in range(target_w):
                        if i < row and j < col:
                            c = self.board[i][j]
                            line.append(int(c.army) if c.player == p else 0)
                        else:
                            line.append(0)
                    grid.append(line)
                pheromone.append(grid)
            return pheromone

        # copy the seeded baseline and optionally add a small army bias
        result = []
        for p in (0, 1):
            grid = []
            for i in range(row):
                line = []
                for j in range(col):
                    base = float(self._pheromone[p][i][j])
                    # bias towards stronger armies of that player
                    if self.board[i][j].player == p:
                        base += float(self.board[i][j].army) * 0.1
                    line.append(base)
                grid.append(line)
            # pad to at least 19×19 for frontend convenience
            for _ in range(len(grid), max(row, 19)):
                grid.append([0.0] * max(col, 19))
            for line in grid:
                if len(line) < max(col, 19):
                    line.extend([0.0] * (max(col, 19) - len(line)))
            result.append(grid)
        return result

    def _build_ants(self) -> list:
        """
        Convert the internal :class:`Ant` objects into the JSON-friendly
        structure the frontend expects.  The ``move`` field records the last
        step taken (or -1 if none).
        """
        ants_json = []
        for ant in self.ants:
            status = ant.get_status((PLAYER_0_BASE_CAMP, PLAYER_1_BASE_CAMP))
            mv = ant.path[-1] if ant.path else -1
            ants_json.append({
                "age": int(ant.age),
                "hp": int(ant.hp),
                "id": int(ant.id),
                "level": int(ant.level),
                "move": int(mv),
                "player": int(ant.player),
                "pos": {"x": int(ant.pos[0]), "y": int(ant.pos[1])},
                "status": status,
            })
            # path is only needed internally; clear after snapshot so that
            # replay/UI clients cannot inadvertently render the full trail.
            ant.path.clear()
        # housekeeping: clear movement log to avoid buildup
        if hasattr(self, "_ant_moves"):
            self._ant_moves.clear()
        return ants_json

    def _base_hp_from_main_general(self) -> list[int]:
        """
        ANT GAME CHANGE

        return the explicit base_hp values instead of using the generals'
        army; the old implementation caused the health to grow when
        production happened on the general tile.
        """
        return [int(self.base_hp[0]), int(self.base_hp[1])]

    def _build_towers_full(self) -> list:
        towers = []
        for g in self.generals:
            if g.player == -1:
                continue
            x, y = g.position
            # ignore any generator whose stored position is outside the board
            if x < 0 or y < 0 or x >= row or y >= col:
                # concurrent bug: general moved out of range, skip it
                continue
            ttype = 0 if isinstance(g, MainGenerals) else (1 if isinstance(g, SubGenerals) else 2)

            cd = 0
            try:
                positives = [int(v) for v in getattr(g, "skills_cd", []) if int(v) > 0]
                cd = min(positives) if positives else 0
            except Exception:
                cd = 0

            towers.append({
                "cd": int(cd),
                "id": int(g.id),
                "player": int(g.player),
                "pos": {"x": int(x), "y": int(y)},
                "type": int(ttype),
            })
        return towers

    def _build_towers_delta(self) -> list:
        """
        towers: 文档说是“回合内新建/变化的塔”(delta)。
        我们把 generals 映射为 towers，并做一次 delta 比较输出。
        """
        current_snapshot: Dict[int, tuple] = {}
        delta = []

        for g in self.generals:
            if g.player == -1:
                continue
            x, y = g.position
            ttype = 0 if isinstance(g, MainGenerals) else (1 if isinstance(g, SubGenerals) else 2)

            # cd: 取最小正 cd（否则 0）
            cd = 0
            try:
                positives = [int(v) for v in getattr(g, "skills_cd", []) if int(v) > 0]
                cd = min(positives) if positives else 0
            except Exception:
                cd = 0

            snap = (int(g.player), int(x), int(y), int(ttype), int(cd))
            gid = int(g.id)
            current_snapshot[gid] = snap

            if gid not in self._prev_towers_snapshot or self._prev_towers_snapshot[gid] != snap:
                delta.append({
                    "cd": int(cd),
                    "id": int(gid),
                    "player": int(g.player),
                    "pos": {"x": int(x), "y": int(y)},
                    "type": int(ttype),
                })

        # update snapshot
        self._prev_towers_snapshot = current_snapshot
        return delta

    def _build_round_state(self) -> dict:
        """
        用你现有 GameState 派生 AntWar round_state 必需字段集合。
        """
        coins = [int(self.coin[0]), int(self.coin[1])]
        camps = self._base_hp_from_main_general()

        # tech_level: [行动力, 攀岩, 免疫沼泽, 超级武器]
        speedLv = [int(self.tech_level[0][0]), int(self.tech_level[1][0])]
        anthpLv = [int(self.tech_level[0][1]), int(self.tech_level[1][1])]

        rs = {
            "anthpLv": anthpLv,
            "ants": self._build_ants(),
            "camps": camps,
            "coins": coins,
            "error": "",
            "message": "[,]",
            "pheromone": self._build_pheromone(),
            "speedLv": speedLv,
            "towers": self._build_towers_full(),
            "winner": int(self.winner),
        }
        return rs

    def append_ant_replay_frame(self, force: bool = False) -> None:
        """
        写入一帧 AntWar replay:
          {seed?, op0, op1, round_state}
        - 正常情况下由 update_round 在每个完整回合末调用
        - force=True: 即使某一方 ops 缺失也强制写一帧（用于异常/终局补帧）
        """
        if self._ant_replay is None:
            return

        op0 = [self._op_to_ant_op(op) for op in (self._last_ops[0] or []) if op]
        op1 = [self._op_to_ant_op(op) for op in (self._last_ops[1] or []) if op]

        if (not force) and (len(op0) == 0 and len(op1) == 0):
            return

        frame = {
            "op0": op0 if op0 else [{"args": -1, "id": -1, "pos": {"x": -1, "y": -1}, "type": 8}],
            "op1": op1 if op1 else [{"args": -1, "id": -1, "pos": {"x": -1, "y": -1}, "type": 8}],
            "round_state": self._build_round_state(),
        }
        self._ant_replay.append(frame)

        # clear ops after write
        self._last_ops = [[], []]

    # ------------------- ant_game helpers -------------------
    def _ant_attack(self) -> None:
        """Simplified ant attack phase.

        The C++ implementation fires towers at ants; in the Python port we
        simply remove any ant that lands on a cell with an opposing general
        (simulating a hit) to keep things sensible for the frontend.
        """
        to_kill = []
        for ant in self.ants:
            # if ant steps onto a tile occupied by an enemy general, kill it
            x, y = ant.pos
            if 0 <= x < row and 0 <= y < col:
                cell = self.board[x][y]
                if cell.generals is not None and cell.generals.player != ant.player:
                    to_kill.append(ant)
        for ant in to_kill:
            ant.set_hp(True and -ant.hp)  # force fail
    
    def _ant_move(self) -> None:
        """Move each alive ant one step towards opponent base."""
        if not self.ants:
            return
        for ant in list(self.ants):
            status = ant.get_status((PLAYER_0_BASE_CAMP, PLAYER_1_BASE_CAMP))
            if status == Ant.Status.Alive:
                target = PLAYER_1_BASE_CAMP if ant.player == 0 else PLAYER_0_BASE_CAMP
                mov = self.map.get_move(ant, target)
                ant.move(mov)
    
    def _ant_update_pheromone(self) -> None:
        """Decay pheromone and then update based on finished ants."""
        # global reduction of pheromone first, matching C++ next_round
        self.map.next_round()
        for ant in self.ants:
            st = ant.get_status((PLAYER_0_BASE_CAMP, PLAYER_1_BASE_CAMP))
            if st in (Ant.Status.Success, Ant.Status.Fail, Ant.Status.TooOld):
                self.map.update_pheromone(ant)
    
    def _ant_manage(self) -> None:
        """Handle ant success/fail/age removal and adjust base HP and coins."""
        new_list = []
        for ant in self.ants:
            st = ant.get_status((PLAYER_0_BASE_CAMP, PLAYER_1_BASE_CAMP))
            if st == Ant.Status.Success:
                # decrement opponent base hp and reward attacker
                if ant.player == 0:
                    self.base_hp[1] -= 1
                else:
                    self.base_hp[0] -= 1
                self.coin[ant.player] += 5  # success bounty
            elif st == Ant.Status.Fail:
                # reward coin to opposing player based on level
                opp = 1 - ant.player
                lvl = ant.level
                reward = 3 if lvl == 0 else (5 if lvl == 1 else 7)
                self.coin[opp] += reward
                # count kill for tiebreaker
                self.kill_count[opp] += 1
            elif st == Ant.Status.TooOld:
                # age‑death gives no coins
                pass
            else:
                new_list.append(ant)
        self.ants = new_list

    def _ant_generate(self) -> None:
        """Generate new ants according to base upgrade rule.

        Each player produces one ant when the round number is divisible by
        K, where K = 4/2/1 for production levels 1/2/3 respectively.
        Production level is stored in the main general's produce_level.
        """
        # find produce_level for each player
        prod_level = [1, 1]
        for g in self.generals:
            if isinstance(g, MainGenerals) and 0 <= g.player <= 1:
                prod_level[g.player] = g.produce_level
        coords = [PLAYER_0_BASE_CAMP, PLAYER_1_BASE_CAMP]
        for player in (0, 1):
            lvl = prod_level[player]
            # map level to K
            K = 4 if lvl == 1 else (2 if lvl == 2 else 1)
            if K <= 0:
                K = 4
            if self.round % K != 0:
                continue
            x, y = coords[player]
            ant = Ant(player=player, id=self._ant_id, x=x, y=y, level=0)
            self._ant_id += 1
            self.ants.append(ant)

    def _ant_increase_age(self) -> None:
        for ant in self.ants:
            ant.increase_age()


def init_generals(gamestate: GameState):
    # init random position
    positions = []
    for i in range(row):
        for j in range(col):
            if gamestate.board[i][j].type == CellType(0):
                positions.append([i, j])
    random.shuffle(positions)

    # generate main generals
    for player in range(2):
        gen = MainGenerals(player=player, id=gamestate.next_generals_id)
        gamestate.next_generals_id += 1
        pos = positions.pop()
        gen.position[0] = pos[0]
        gen.position[1] = pos[1]
        gamestate.generals.append(gen)
        gamestate.board[pos[0]][pos[1]].generals = gen
        gamestate.board[pos[0]][pos[1]].player = player
        gamestate.board[pos[0]][pos[1]].army = 50

    # generate sub generals
    for player in range(subgen_num):
        gen = SubGenerals(player=-1, id=gamestate.next_generals_id)
        gamestate.next_generals_id += 1
        pos = positions.pop()
        gen.position[0] = pos[0]
        gen.position[1] = pos[1]
        gamestate.generals.append(gen)
        gamestate.board[pos[0]][pos[1]].generals = gen
        gamestate.board[pos[0]][pos[1]].army = random.randint(10, 20)

    # generate farmer
    for i in range(farmer_num):
        gen = Farmer(player=-1, produce_level=1, id=gamestate.next_generals_id)
        gamestate.next_generals_id += 1
        pos = positions.pop()
        gen.position[0] = pos[0]
        gen.position[1] = pos[1]
        gamestate.generals.append(gen)
        gamestate.board[pos[0]][pos[1]].generals = gen
        gamestate.board[pos[0]][pos[1]].army = random.randint(3, 5)


def update_round(gamestate: GameState):
    changed = set()
    for i in range(row):
        for j in range(col):
            # 将军
            if gamestate.board[i][j].generals is not None:
                gamestate.board[i][j].generals.rest_move = gamestate.board[i][j].generals.mobility_level

            if isinstance(gamestate.board[i][j].generals, MainGenerals):
                gamestate.board[i][j].army += gamestate.board[i][j].generals.produce_level
                changed.add(i * col + j)
            elif isinstance(gamestate.board[i][j].generals, SubGenerals):
                if gamestate.board[i][j].generals.player != -1:
                    gamestate.board[i][j].army += gamestate.board[i][j].generals.produce_level
                    changed.add(i * col + j)
            elif isinstance(gamestate.board[i][j].generals, Farmer):
                if gamestate.board[i][j].generals.player != -1:
                    gamestate.coin[gamestate.board[i][j].generals.player] += gamestate.board[i][j].generals.produce_level

            # 每10回合增兵；记录轮数避免同一回合多次调用重复加兵
            if gamestate.round % 10 == 0 and gamestate.round != gamestate._last_bonus_round:
                if gamestate.board[i][j].player != -1:
                    gamestate.board[i][j].army += 1
                    changed.add(i * col + j)

            # 沼泽减兵
            if (
                gamestate.board[i][j].type == CellType(1)
                and gamestate.board[i][j].player != -1
                and gamestate.board[i][j].army > 0
            ):
                if gamestate.tech_level[gamestate.board[i][j].player][2] == 0:
                    gamestate.board[i][j].army -= 1
                    if gamestate.board[i][j].army == 0 and gamestate.board[i][j].generals is None:
                        gamestate.board[i][j].player = -1
                    # FIX: 原来是 i * row + j，应该是 i * col + j
                    changed.add(i * col + j)

    # 超级武器判定
    for weapon in gamestate.active_super_weapon:
        if weapon.type == WeaponType(0):
            for _i in range(max(0, weapon.position[0] - 1), min(row, weapon.position[0] + 2)):
                for _j in range(max(0, weapon.position[1] - 1), min(col, weapon.position[1] + 2)):
                    if gamestate.board[_i][_j].army > 0:
                        gamestate.board[_i][_j].army = max(0, gamestate.board[_i][_j].army - 3)
                        gamestate.board[_i][_j].player = (
                            -1
                            if (gamestate.board[_i][_j].army == 0 and gamestate.board[_i][_j].generals is None)
                            else gamestate.board[_i][_j].player
                        )
                        changed.add(_i * col + _j)

    # 更新超级武器信息
    gamestate.super_weapon_cd = [i - 1 if i > 0 else i for i in gamestate.super_weapon_cd]
    for weapon in gamestate.active_super_weapon:
        weapon.rest -= 1

    # cd和duration 减少
    for gen in gamestate.generals:
        gen.skills_cd = [i - 1 if i > 0 else i for i in gen.skills_cd]
        gen.skill_duration = [i - 1 if i > 0 else i for i in gen.skill_duration]

    # 移动步数恢复
    gamestate.rest_move_step = [gamestate.tech_level[0][0], gamestate.tech_level[1][0]]

    # 你原来的增量 replay dict 仍然可以生成（供 AI/调试），但不再直接写入 replay_file
    _ = get_single_round_replay(gamestate, [[int(i / col), i % col] for i in changed], -1, [8])

    gamestate.active_super_weapon = list(filter(lambda x: (x.rest > 0), gamestate.active_super_weapon))

    # ---------- ant_game lifecycle ----------
    gamestate._ant_attack()
    gamestate._ant_move()
    gamestate._ant_update_pheromone()
    gamestate._ant_manage()
    gamestate._ant_generate()
    gamestate._ant_increase_age()
    # ---------- end ant lifecycle ----------

    # 每个完整回合给两位玩家+1金币（可选规则），同样打表防止多次调用
    if gamestate.round != gamestate._last_bonus_round:
        gamestate.coin[0] += 1
        gamestate.coin[1] += 1
        gamestate._last_bonus_round = gamestate.round

    # 在回合数+1之前写入 AntWar replay frame
    gamestate.append_ant_replay_frame(force=False)

    gamestate.round += 1
