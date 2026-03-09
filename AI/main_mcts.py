try:
    from ai_mcts import AI
    from main import main
except ModuleNotFoundError as exc:
    if exc.name not in {"ai_mcts", "main"}:
        raise
    from AI.ai_mcts import AI
    from AI.main import main


if __name__ == "__main__":
    main(AI)
