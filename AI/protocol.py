from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from typing import Iterable

try:
    from common import BaseAgent
except ModuleNotFoundError as exc:
    if exc.name != "common":
        raise
    from AI.common import BaseAgent

from SDK.backend import load_backend
from SDK.engine import GameState, PublicRoundState
from SDK.model import Operation
from SDK.constants import OperationType


@dataclass(slots=True)
class ProtocolController:
    player: int
    state: GameState
    agent: BaseAgent

    def decide(self) -> list[Operation]:
        return self.agent.choose_operations(self.state, self.player)

    def apply_self_operations(self, operations: Iterable[Operation]) -> list[Operation]:
        return self.state.apply_operation_list(self.player, operations)

    def apply_opponent_operations(self, operations: Iterable[Operation]) -> list[Operation]:
        return self.state.apply_operation_list(1 - self.player, operations)

    def finish_round(self, public_round_state: PublicRoundState) -> None:
        self.state.advance_round()
        self.state.sync_public_round_state(public_round_state)


class ProtocolIO:
    def __init__(self, stdin=None, stdout=None, stderr=None) -> None:
        self.stdin = stdin or sys.stdin.buffer
        self.stdout = stdout or sys.stdout.buffer
        self.stderr = stderr or sys.stderr

    def log(self, message: str) -> None:
        self.stderr.write(f"[AI] {message}\n")
        self.stderr.flush()

    def recv_line(self) -> str | None:
        raw = self.stdin.readline()
        if not raw:
            return None
        return raw.decode("utf-8", errors="replace").rstrip("\n")

    def send_packet(self, payload: str) -> None:
        if not payload.endswith("\n"):
            payload += "\n"
        data = payload.encode("utf-8")
        self.stdout.write(struct.pack(">I", len(data)))
        self.stdout.write(data)
        self.stdout.flush()

    def recv_init(self) -> tuple[int, int]:
        line = self.recv_line()
        if line is None:
            raise RuntimeError("missing init line")
        player, seed = map(int, line.split())
        return player, seed

    def recv_operations(self) -> list[Operation]:
        line = self.recv_line()
        if line is None:
            raise RuntimeError("missing operation count")
        count = int(line.strip())
        operations: list[Operation] = []
        for _ in range(count):
            payload = self.recv_line()
            if payload is None:
                raise RuntimeError("unexpected EOF while reading operations")
            parts = [int(item) for item in payload.split()]
            op_type = OperationType(parts[0])
            if len(parts) == 1:
                operations.append(Operation(op_type))
            elif len(parts) == 2:
                operations.append(Operation(op_type, parts[1]))
            else:
                operations.append(Operation(op_type, parts[1], parts[2]))
        return operations

    def recv_round_state(self) -> PublicRoundState | None:
        line = self.recv_line()
        if line is None:
            return None
        round_index = int(line.strip())
        tower_count = int((self.recv_line() or "0").strip())
        towers = []
        for _ in range(tower_count):
            towers.append(tuple(map(int, (self.recv_line() or "").split())))
        ant_count = int((self.recv_line() or "0").strip())
        ants = []
        for _ in range(ant_count):
            ants.append(tuple(map(int, (self.recv_line() or "").split())))
        coins = tuple(map(int, (self.recv_line() or "0 0").split()[:2]))
        camps_hp = tuple(map(int, (self.recv_line() or "0 0").split()[:2]))
        return PublicRoundState(round_index=round_index, towers=towers, ants=ants, coins=coins, camps_hp=camps_hp)

    def send_operations(self, operations: Iterable[Operation]) -> None:
        items = list(operations)
        lines = [str(len(items))]
        lines.extend(" ".join(str(token) for token in operation.to_protocol_tokens()) for operation in items)
        self.send_packet("\n".join(lines) + "\n")


def run_agent(agent: BaseAgent, io: ProtocolIO | None = None) -> None:
    io = io or ProtocolIO()
    player, seed = io.recv_init()
    controller = ProtocolController(player=player, state=load_backend(prefer_native=False).initial_state(seed=seed), agent=agent)
    agent.on_match_start(player, seed)

    while True:
        if player == 0:
            my_ops = controller.decide()
            controller.apply_self_operations(my_ops)
            agent.on_self_operations(my_ops)
            io.send_operations(my_ops)
            try:
                opponent_ops = io.recv_operations()
            except Exception:
                break
            controller.apply_opponent_operations(opponent_ops)
            agent.on_opponent_operations(opponent_ops)
            round_state = io.recv_round_state()
            if round_state is None:
                break
            controller.finish_round(round_state)
            agent.on_round_state(round_state)
        else:
            try:
                opponent_ops = io.recv_operations()
            except Exception:
                break
            controller.apply_opponent_operations(opponent_ops)
            agent.on_opponent_operations(opponent_ops)
            my_ops = controller.decide()
            controller.apply_self_operations(my_ops)
            agent.on_self_operations(my_ops)
            io.send_operations(my_ops)
            round_state = io.recv_round_state()
            if round_state is None:
                break
            controller.finish_round(round_state)
            agent.on_round_state(round_state)
