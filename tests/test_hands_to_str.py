import pytest
from bridge_core_py.bids import SpecialBid, Suit, TrickBid, Tricks
from bridge_core_py.cards import Card, Rank, Suit as CardSuit
from bridge_core_py.core import Game


def mock_game(seed: int = 42):
    game = Game(seed=seed)
    game.step(TrickBid(Suit.CLUBS, Tricks.ONE))
    game.step(SpecialBid.PASS)
    game.step(SpecialBid.PASS)
    game.step(SpecialBid.PASS)

    return game


def test_hands_init():
    game = mock_game(0)

    hands = game.hands_to_str()
    assert hands == "N:.AQJT43.KQT7.KQ6 Q863..J9865.8543 KJT95.K862.A.JT2 A742.975.432.A97"


def test_hands_after_one_move():
    game = mock_game(0)
    game.step(Card(CardSuit.SPADES, Rank.QUEEN))

    hands = game.hands_to_str()
    assert hands == "N:.AQJT43.KQT7.KQ6 863..J9865.8543 KJT95.K862.A.JT2 A742.975.432.A97"


def test_hands_after_one_round():
    game = mock_game(0)
    game.step(Card(CardSuit.SPADES, Rank.QUEEN))
    game.step(Card(CardSuit.SPADES, Rank.KING))
    game.step(Card(CardSuit.SPADES, Rank.ACE))
    game.step(Card(CardSuit.CLUBS, Rank.SIX))

    hands = game.hands_to_str()
    assert hands == "N:.AQJT43.KQT7.KQ 863..J9865.8543 JT95.K862.A.JT2 742.975.432.A97"
