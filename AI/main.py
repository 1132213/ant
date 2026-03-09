from __future__ import annotations

try:
    from ai import AI as PackagedAI
except ModuleNotFoundError as exc:  # pragma: no cover - repository layout
    if exc.name != "ai":
        raise
    PackagedAI = None

try:
    from protocol import run_agent
except ModuleNotFoundError as exc:  # pragma: no cover - repository entrypoint fallback
    if exc.name != "protocol":
        raise
    from AI.protocol import run_agent


def main(ai_cls=None) -> None:
    agent_cls = ai_cls or PackagedAI
    if agent_cls is None:
        raise RuntimeError("main.py expects ai.py to export class AI, or an explicit ai_cls argument")
    run_agent(agent_cls())


if __name__ == "__main__":  # pragma: no cover - exercised in packaged layout
    main()
