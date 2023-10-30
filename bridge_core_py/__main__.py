from bridge_core_py.cards import Suit
from bridge_core_py.interactive import run
from bridge_core_py.player import PlayerDirection
from tests.test_game_playing import mock_game





if __name__ == '__main__':
    # game = mock_game(seed=4)
    # n = None
    # for i in range(5, 100):
    #     game = mock_game(seed=i)
    #     bid = game.bid
    #     for player in [game.players[direction] for direction in PlayerDirection]:
    #         # print(player.cards)
    #         suits = [c.suit for c in player.cards]
    #         if not all(suit in suits for suit in Suit) and bid.suit in suits:
    #             n = i
    #             print(player)
    #             break
    #         # else:
    #             # print(False)
    #     if n != None:
    #         print(n)
    #         break
    # game = mock_game()
    # print(game.game_observation())
    run()
