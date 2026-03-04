from logic.game_rules import is_game_over, tiebreak_now
from logic.constant import row, col


def test_is_game_over_by_main_general_death(plain_state):
    s = plain_state
    from tests.conftest import place_general

    g0 = place_general(s, "main", 0, 0, 0)
    g1 = place_general(s, "main", 0, 1, 1)
    # Remove player 1 main general from the game
    s.generals.remove(g1)
    assert is_game_over(s) == 0


def test_tiebreak_kills_and_superweapon(plain_state):
    s = plain_state
    from tests.conftest import place_general

    # need main generals so is_game_over doesn't return early
    place_general(s, "main", 0, 0, 0)
    place_general(s, "main", 0, 1, 1)

    # simulate reaching max rounds
    s.round = 513

    # equal base_hp initially
    s.base_hp = [10, 10]

    # if one player has more kills, they win
    s.kill_count = [5, 2]
    assert is_game_over(s) == 0
    s.kill_count = [2, 5]
    assert is_game_over(s) == 1

    # if kills tied, fewer superweapon uses wins
    s.kill_count = [3, 3]
    s.superweapon_used = [1, 4]
    assert is_game_over(s) == 0
    s.superweapon_used = [4, 1]
    assert is_game_over(s) == 1

    # if everything equal, first player wins by convention
    s.superweapon_used = [2, 2]
    assert is_game_over(s) == 0


def test_game_over_by_base_hp(plain_state):
    s = plain_state
    # manipulating explicit base_hp should determine winner immediately
    s.base_hp = [0, 10]
    assert is_game_over(s) == 1
    s.base_hp = [5, 0]
    assert is_game_over(s) == 0
    s.base_hp = [0, 0]
    assert is_game_over(s) == 0

