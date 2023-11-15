from bridge_core_py.bids import SpecialBid, Suit, TrickBid, Tricks
from bridge_core_py.cards import Card, Rank
from bridge_core_py.cards import Suit as CardSuit
from bridge_core_py.core import Game
from bridge_core_py.player import PlayerDirection


def test_game_dummy_showing_declarer_as_dummy():
    game = Game()

    game.step(TrickBid(Suit.CLUBS, Tricks.ONE))
    game.step(SpecialBid.PASS)
    game.step(SpecialBid.PASS)
    game.step(SpecialBid.PASS)

    game.step(Card(CardSuit.SPADES, Rank.FIVE))

    dummy_player = game.round_player.next()

    assert (
        game.players[game.declarer].cards
        == game.player_observation(dummy_player)["game"]["dummy"]
    )
