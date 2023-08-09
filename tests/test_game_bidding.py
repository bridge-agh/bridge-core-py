import pytest
from bridge_core_py.bids import SpecialBid, Suit, TrickBid, Tricks
from bridge_core_py.core import Game, GameStage


def test_game_init():
    game = Game()

    assert game.stage == GameStage.BIDDING

    for player in game.players:
        assert len(game.players[player].cards) == 13


def test_4pass():
    game = Game()

    game.step(SpecialBid.PASS)
    game.step(SpecialBid.PASS)
    game.step(SpecialBid.PASS)
    game.step(SpecialBid.PASS)

    assert game.stage == GameStage.SCORING


def test_double():
    game = Game()

    game.step(TrickBid(Suit.CLUBS, Tricks.TWO))
    game.step(SpecialBid.DOUBLE)
    game.step(SpecialBid.PASS)
    game.step(SpecialBid.DOUBLE)

    assert game.stage == GameStage.BIDDING
    assert game.multiplier == 4

def test_reset_double():
    game = Game()
    
    game.step(TrickBid(Suit.CLUBS, Tricks.TWO))
    game.step(SpecialBid.DOUBLE)
    game.step(SpecialBid.PASS)
    game.step(TrickBid(Suit.CLUBS, Tricks.THREE))
    
    assert game.stage == GameStage.BIDDING
    assert game.multiplier == 1


def test_basic_bidding1():
    game = Game()

    game.step(TrickBid(Suit.HEARTS, Tricks.ONE))
    game.step(TrickBid(Suit.CLUBS, Tricks.TWO))
    game.step(TrickBid(Suit.HEARTS, Tricks.TWO))
    game.step(SpecialBid.PASS)
    game.step(SpecialBid.PASS)
    game.step(SpecialBid.PASS)

    assert game.stage == GameStage.PLAYING


def test_basic_bidding2():
    game = Game()

    game.step(TrickBid(Suit.SPADES, Tricks.SEVEN))
    game.step(TrickBid(Suit.NO_TRUMP, Tricks.SEVEN))
    game.step(SpecialBid.DOUBLE)
    game.step(SpecialBid.DOUBLE)
    game.step(SpecialBid.DOUBLE)
    game.step(SpecialBid.DOUBLE)
    game.step(SpecialBid.PASS)
    game.step(SpecialBid.PASS)
    game.step(SpecialBid.PASS)

    assert game.stage == GameStage.PLAYING


def test_illegal_bidding_same_bid1():
    game = Game()

    with pytest.raises(ValueError):
        game.step(TrickBid(Suit.HEARTS, Tricks.ONE))
        game.step(TrickBid(Suit.HEARTS, Tricks.ONE))


def test_illegal_bidding_same_bid2():
    game = Game()

    with pytest.raises(ValueError):
        game.step(TrickBid(Suit.CLUBS, Tricks.SEVEN))
        game.step(TrickBid(Suit.CLUBS, Tricks.SEVEN))


def test_illegal_bidding_lower_bid1():
    game = Game()

    with pytest.raises(ValueError):
        game.step(TrickBid(Suit.HEARTS, Tricks.TWO))
        game.step(TrickBid(Suit.HEARTS, Tricks.ONE))


def test_illegal_bidding_lower_bid2():
    game = Game()

    with pytest.raises(ValueError):
        game.step(TrickBid(Suit.CLUBS, Tricks.TWO))
        game.step(TrickBid(Suit.HEARTS, Tricks.THREE))
        game.step(TrickBid(Suit.CLUBS, Tricks.TWO))
