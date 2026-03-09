try:
    from ai_random import AI
    from main import main
except ModuleNotFoundError as exc:
    if exc.name not in {"ai_random", "main"}:
        raise
    from AI.ai_random import AI
    from AI.main import main


if __name__ == "__main__":
    main(AI)
