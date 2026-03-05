"""
真正通过 `ant_game - deploy` C++ 可执行文件驱动的 PettingZoo AEC 封装。

约定（与你确认的方案一致）：
- 仍然是 AEC：agents = ["player_0", "player_1"]，交替调用 step。
- 每次 step 只提交“当前玩家的一条 Operation 或结束回合指令”：
  - type=0: 结束本玩家本回合（不发送任何 Operation 给 C++）
  - type>0: 解释为一条 Operation，先累积在 Python 端，等两名玩家
    都结束回合后，一次性把本回合的两个 Operation 列表发给 C++，
    再从 C++ 读取新一轮的状态 JSON。

注意：由于 C++ 输出 JSON 结构在仓库中没有完整文档，这里按惯例做了
“best-effort” 解析，只依赖非常通用的字段（coins、camps、winner 等），
若字段缺失则使用缺省值，保证环境逻辑可以正常运行。
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from pettingzoo import AECEnv
    from gymnasium import spaces
except Exception:  # pragma: no cover - 运行时可能没有安装这两个库
    AECEnv = object  # type: ignore
    spaces = None  # type: ignore


@dataclass
class EnvConfig:
    max_rounds: int = 512
    max_ops_per_turn: int = 16


class CppAntGameProcess:
    """
    负责和 C++ 可执行程序进行 JSON+长度前缀 的双向通信。
    """

    def __init__(self, exe_path: str):
        self.exe_path = exe_path
        self.proc: Optional[subprocess.Popen] = None

    # ---- 基础 IO ----
    def _ensure_started(self):
        if self.proc is not None:
            return
        # 使用仓库根目录作为工作目录
        cwd = str(Path(__file__).resolve().parent.parent)
        self.proc = subprocess.Popen(
            [self.exe_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            cwd=cwd,
        )

    def _write_json(self, obj: dict):
        assert self.proc is not None and self.proc.stdin is not None
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.proc.stdin.write(struct.pack(">I", len(data)))
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def _read_json(self) -> dict:
        assert self.proc is not None and self.proc.stdout is not None
        hdr = self.proc.stdout.read(4)
        if not hdr:
            raise RuntimeError("C++ antgame process closed stdout unexpectedly.")
        (length,) = struct.unpack(">I", hdr)
        body = self.proc.stdout.read(length)
        if len(body) != length:
            raise RuntimeError("Failed to read full JSON body from C++ antgame.")
        return json.loads(body.decode("utf-8"))

    # ---- 协议封装 ----
    def send_init(self, seed: int | None, replay_path: str) -> dict:
        """
        发送 from_judger_init 对应的 JSON，读取 C++ 返回的初始状态 JSON。
        """
        self._ensure_started()
        init_msg = {
            "player_list": [1, 1],  # 1 表示 AI
            "player_num": 2,
            "config": {"random_seed": int(seed or 0)},
            "replay": replay_path,
        }
        self._write_json(init_msg)
        # C++ 会通过 output_info 返回一条包含初始信息的 JSON，这里直接读取
        state = self._read_json()
        return state

    def send_round(self, p0_ops: List[dict], p1_ops: List[dict]) -> dict:
        """
        发送一轮中双方的操作列表：
        - 先发 player=0 的 from_judger_round
        - 再发 player=1 的 from_judger_round
        然后读取 C++ 该轮结束后 dump_round_state 输出的一条 JSON。
        """
        self._ensure_started()

        def _round_msg(player: int, ops: List[dict]) -> dict:
            # content 按 from_judger_round 的设计是一个字符串，这里约定为
            # JSON 字符串形式的 Operation 数组，转到 C++ 侧由 comm_judger 中
            # 的逻辑解析成 std::vector<Operation>。
            ops_str = json.dumps(ops, ensure_ascii=False)
            return {"player": int(player), "content": ops_str, "time": 0}

        self._write_json(_round_msg(0, p0_ops))
        self._write_json(_round_msg(1, p1_ops))
        state = self._read_json()
        return state

    def close(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None


class AntGameAECEnv(AECEnv):
    """
    通过 C++ ant_game 可执行文件驱动的 PettingZoo AEC 环境。

    - agents: ["player_0", "player_1"]
    - action: MultiDiscrete([type, x, y, id, args])
        * type == 0: 结束本玩家本轮，不发送 Operation（即空操作列表）
        * type > 0: 解释为一条 Operation，累积在当前轮的 ops 列表中
    - 每当两名玩家都执行了“结束本轮”（type==0）后：
        * 将双方累积的 Operation 列表作为一整轮发送给 C++，
        * 从 C++ 读取该轮结束后的 JSON 状态，
        * 更新内部观测和奖励，并开始下一轮。
    """

    metadata = {
        "render_modes": ["ansi"],
        "name": "antgame_cpp_pettingzoo_v0",
    }

    def __init__(self, render_mode: str | None = None, config: EnvConfig | None = None):
        self.render_mode = render_mode
        self.config = config or EnvConfig()

        # agent 列表
        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents.copy()

        # 内部轮次与 C++ 通信进程
        root = Path(__file__).resolve().parent.parent
        exe_name = "main.exe" if os.name == "nt" else "main"
        exe_path = root / "ant_game - deploy" / "output" / exe_name
        self._cpp = CppAntGameProcess(str(exe_path))

        self._last_state: Optional[dict] = None
        self._round_idx: int = 0
        self._pending_ops: Dict[int, List[dict]] = {0: [], 1: []}
        self._ended_this_round: Dict[int, bool] = {0: False, 1: False}

        # AEC 必需字段
        self.rewards: Dict[str, float] = {a: 0.0 for a in self.possible_agents}
        self.terminations: Dict[str, bool] = {a: False for a in self.possible_agents}
        self.truncations: Dict[str, bool] = {a: False for a in self.possible_agents}
        self.infos: Dict[str, Dict[str, Any]] = {a: {} for a in self.possible_agents}
        self._cumulative_rewards: Dict[str, float] = {a: 0.0 for a in self.possible_agents}
        self._current: int = 0
        self.agent_selection: str = self.possible_agents[self._current]

        self._build_spaces()

    # ------------------------------------------------------------------
    # PettingZoo Core API
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        options = options or {}
        del options

        self.agents = self.possible_agents.copy()
        self.rewards = {a: 0.0 for a in self.possible_agents}
        self.terminations = {a: False for a in self.possible_agents}
        self.truncations = {a: False for a in self.possible_agents}
        self.infos = {a: {} for a in self.possible_agents}
        self._cumulative_rewards = {a: 0.0 for a in self.possible_agents}
        self._current = 0
        self.agent_selection = self.possible_agents[self._current]
        self._pending_ops = {0: [], 1: []}
        self._ended_this_round = {0: False, 1: False}
        self._round_idx = 0

        # 发送初始化消息给 C++，读取首帧状态
        replay_dir = Path("replays")
        replay_dir.mkdir(exist_ok=True)
        replay_path = str(replay_dir / "antgame_cpp_pettingzoo.json")
        self._last_state = self._cpp.send_init(seed, replay_path)

    def observe(self, agent: str):
        player = 0 if agent == "player_0" else 1
        return self._build_obs(player)

    def step(self, action: List[int] | np.ndarray):
        if any(self.terminations.values()) or any(self.truncations.values()):
            return

        # 规范化 action
        if isinstance(action, np.ndarray):
            action = action.tolist()
        if not isinstance(action, list) or len(action) < 1:
            raise ValueError("action must be a non-empty list or np.ndarray")

        player = self._current
        agent = self.possible_agents[player]
        if agent != self.agent_selection:
            # 顺序错误的 step 直接忽略
            return

        atype = int(action[0])

        # 清空该 agent 本步 reward
        self.rewards[agent] = 0.0

        if atype == 0:
            # 结束本玩家本轮
            self._ended_this_round[player] = True
        else:
            # 解码为一条 Operation 并累积
            op = self._decode_to_operation(action)
            self._pending_ops[player].append(op)

        # 若双方都结束此轮，则把整轮 ops 发送给 C++ 并更新状态
        if self._ended_this_round[0] and self._ended_this_round[1]:
            self._flush_round_and_update_state()
            self._ended_this_round = {0: False, 1: False}
            self._pending_ops = {0: [], 1: []}
            self._round_idx += 1

        # 切换当前玩家（AEC 交替）
        self._current = 1 - player
        self.agent_selection = self.possible_agents[self._current]

    def render(self):
        if self.render_mode != "ansi":
            return None
        if self._last_state is None:
            return "<uninitialized>"
        try:
            coins = self._extract_coins(self._last_state)
            camps_hp = self._extract_camps_hp(self._last_state)
            return f"Round={self._round_idx} Coins={coins} CampsHP={camps_hp}"
        except Exception:
            return f"Round={self._round_idx} raw_state={self._last_state}"

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    def _build_spaces(self):
        # Action: [type, x, y, id, args]
        self._action_space = spaces.MultiDiscrete(
            np.array(
                [
                    40,   # type（0=EndTurn，其它映射到 Operation::Type）
                    20,   # x（MAP_SIZE=19，给一点冗余）
                    20,   # y
                    256,  # id
                    32,   # args（升级类型/道具类型等）
                ],
                dtype=np.int64,
            )
        )

        self._observation_space = spaces.Dict(
            {
                "coins": spaces.Box(low=0, high=10_000, shape=(2,), dtype=np.int32),
                "camps_hp": spaces.Box(low=0, high=1_000, shape=(2,), dtype=np.int32),
                "current_player": spaces.Discrete(2),
                "round": spaces.Discrete(self.config.max_rounds + 1),
                # 将原始 JSON 字符串压缩成最多 4096 字节的向量，便于需要时做原始特征
                "raw_state_bytes": spaces.Box(
                    low=0, high=255, shape=(4096,), dtype=np.uint8
                ),
            }
        )

    def action_space(self, agent: str):
        return self._action_space

    def observation_space(self, agent: str):
        return self._observation_space

    def _build_obs(self, player: int) -> Dict[str, Any]:
        state = self._last_state or {}
        coins = self._extract_coins(state)
        camps_hp = self._extract_camps_hp(state)
        raw_bytes = json.dumps(state, ensure_ascii=False).encode("utf-8")
        buf = np.zeros((4096,), dtype=np.uint8)
        n = min(len(raw_bytes), 4096)
        buf[:n] = np.frombuffer(raw_bytes[:n], dtype=np.uint8)
        obs = {
            "coins": coins,
            "camps_hp": camps_hp,
            "current_player": int(player),
            "round": int(self._round_idx),
            "raw_state_bytes": buf,
        }
        return obs

    def _extract_coins(self, state: dict) -> np.ndarray:
        coins = state.get("coins")
        if isinstance(coins, list) and len(coins) == 2:
            try:
                return np.array([int(coins[0]), int(coins[1])], dtype=np.int32)
            except Exception:
                pass
        return np.zeros((2,), dtype=np.int32)

    def _extract_camps_hp(self, state: dict) -> np.ndarray:
        """
        C++ 里是通过 Output::add_camps 写入的，常见结构是：
          "camps": [[hp0, ...], [hp1, ...]]
        这里只尽量解析第一个元素为血量。
        """
        camps = state.get("camps")
        if isinstance(camps, list) and len(camps) == 2:
            try:
                hp0 = int(camps[0][0]) if isinstance(camps[0], list) and camps[0] else 0
                hp1 = int(camps[1][0]) if isinstance(camps[1], list) and camps[1] else 0
                return np.array([hp0, hp1], dtype=np.int32)
            except Exception:
                pass
        # 兜底：如果没有 camps 字段，则尝试 winner 信息或使用默认 50
        return np.array([50, 50], dtype=np.int32)

    def _decode_to_operation(self, action: List[int]) -> dict:
        """
        将 MultiDiscrete 动作解码为单条 Operation：
          { "type": int, "id": int, "args": int, "pos": {"x": int, "y": int} }
        这与 ant_game - deploy/include/operation.h 中 Operation 定义保持一致。
        """
        op_type = int(action[0])
        x = int(action[1])
        y = int(action[2])
        op_id = int(action[3])
        args = int(action[4])
        return {
            "type": op_type,
            "id": op_id,
            "args": args,
            "pos": {"x": x, "y": y},
        }

    def _flush_round_and_update_state(self) -> None:
        """
        将当前轮双方累积的 Operation 列表发送给 C++，并更新内部状态/奖励。
        """
        try:
            new_state = self._cpp.send_round(self._pending_ops[0], self._pending_ops[1])
        except Exception as exc:
            # 通信发生错误时，直接截断并给双方 0 奖励
            self._last_state = None
            self.terminations = {a: True for a in self.agents}
            self.rewards = {a: 0.0 for a in self.agents}
            self.infos["player_0"]["error"] = str(exc)
            self.infos["player_1"]["error"] = str(exc)
            return

        self._last_state = new_state

        # 胜负判定：若 JSON 中有 "winner" 字段（0/1），则结束对局
        winner = new_state.get("winner", -1)
        if winner in (0, 1):
            self.terminations = {a: True for a in self.agents}
            self.rewards = {a: 0.0 for a in self.agents}
            win_agent = self.agents[int(winner)]
            lose_agent = self.agents[1 - int(winner)]
            self.rewards[win_agent] = 1.0
            self.rewards[lose_agent] = -1.0


def env(render_mode: str | None = None, config: EnvConfig | None = None) -> AntGameAECEnv:
    return AntGameAECEnv(render_mode=render_mode, config=config)


def parallel_env(*args, **kwargs):
    try:
        from pettingzoo.utils.conversions import aec_to_parallel
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pettingzoo not installed for parallel conversion") from e
    return aec_to_parallel(AntGameAECEnv(*args, **kwargs))


