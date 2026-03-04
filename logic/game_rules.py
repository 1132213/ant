from logic.constant import row, col
from logic.gamedata import MainGenerals
from logic.gamestate import GameState


def is_game_over(state: GameState) -> int:
    """
    Determine winner.
    - Returns -1 if ongoing
    - Returns 0 or 1 if that player wins
    Rules:
      1) If a player's main general is eliminated or base HP drops to zero,
         the opponent wins immediately.
      2) If round <= 512, game continues
      3) Otherwise tiebreaker per official rules (see implementation)
    """
    # base camp HP victory condition (per official rules)
    if hasattr(state, "base_hp"):
        # immediate wins when one camp drops to 0 or below
        if state.base_hp[0] <= 0 and state.base_hp[1] <= 0:
            return 0  # convention: player0 wins on simultaneous kill
        if state.base_hp[0] <= 0:
            return 1
        if state.base_hp[1] <= 0:
            return 0

    # fall back to main-general elimination if no base_hp attr
    main_alive = {0: 0, 1: 0}
    for g in state.generals:
        if isinstance(g, MainGenerals):
            main_alive[g.player] = 1
    if main_alive[0] == 0 and main_alive[1] == 1:
        return 1
    if main_alive[1] == 0 and main_alive[0] == 1:
        return 0

    # if still ongoing and round limit not reached, continue
    if state.round <= 512:
        return -1

    # after max rounds, apply tiebreaker hierarchy per rules
    # 1) remaining base HP
    if hasattr(state, "base_hp") and state.base_hp[0] != state.base_hp[1]:
        return 0 if state.base_hp[0] > state.base_hp[1] else 1
    # 2) number of opponent ants defeated
    if hasattr(state, "kill_count") and state.kill_count[0] != state.kill_count[1]:
        return 0 if state.kill_count[0] > state.kill_count[1] else 1
    # 3) fewer superweapon uses wins
    if hasattr(state, "superweapon_used") and state.superweapon_used[0] != state.superweapon_used[1]:
        return 0 if state.superweapon_used[0] < state.superweapon_used[1] else 1
    # 4) AI runtime not tracked here; fall through to first-player advantage
    return 0


def tiebreak_now(state: GameState) -> int:
    """Apply tiebreak immediately without waiting for round > 512."""
    current = state.round
    state.round = 513
    try:
        return is_game_over(state)
    finally:
        state.round = current

