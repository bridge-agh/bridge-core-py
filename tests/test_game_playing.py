import pytest
from bridge_core_python.bids import SpecialBid, Suit, TrickBid, Tricks
from bridge_core_python.cards import Card, Rank, Suit as CardSuit
from bridge_core_python.core import Game, GameStage
from bridge_core_python.player import PlayerDirection


def mock_game(seed: int = 42):
    game = Game(seed=seed)

    game.step(TrickBid(Suit.CLUBS, Tricks.ONE))
    game.step(SpecialBid.PASS)
    game.step(SpecialBid.PASS)
    game.step(SpecialBid.PASS)

    return game


def test_game_init():
    game = mock_game()

    assert game.stage == GameStage.PLAYING
    assert game.current_player == game.declarer.prev()
    assert game.bid == TrickBid(Suit.CLUBS, Tricks.ONE)

    cards = set()
    for player in game.players:
        cards.update(game.players[player].cards)

    assert len(cards) == 52


def test_same_suit():
    game = mock_game()

    assert game.current_player == PlayerDirection.NORTH

    game.step(Card(CardSuit.DIAMONDS, Rank.SIX))
    game.step(Card(CardSuit.DIAMONDS, Rank.KING))
    game.step(Card(CardSuit.DIAMONDS, Rank.JACK))
    game.step(Card(CardSuit.DIAMONDS, Rank.TWO))

    assert game.current_player == PlayerDirection.EAST
    assert len(game.EW_tricks) == 1
    assert len(game.NS_tricks) == 0


def test_same_suit_missing_card():
    game = mock_game()

    assert game.current_player == PlayerDirection.NORTH

    with pytest.raises(ValueError):
        game.step(Card(CardSuit.DIAMONDS, Rank.SIX))
        game.step(Card(CardSuit.DIAMONDS, Rank.SIX))


def test_different_suit_no_trump():
    game = mock_game(seed=0)

    assert game.current_player == PlayerDirection.WEST

    game.step(Card(CardSuit.HEARTS, Rank.FIVE))
    game.step(Card(CardSuit.HEARTS, Rank.TEN))
    game.step(Card(CardSuit.DIAMONDS, Rank.EIGHT))
    game.step(Card(CardSuit.HEARTS, Rank.TWO))

    assert game.current_player == PlayerDirection.NORTH
    assert len(game.EW_tricks) == 0
    assert len(game.NS_tricks) == 1


def test_different_suit_no_trump_missing_card():
    game = mock_game(seed=0)

    assert game.current_player == PlayerDirection.WEST

    with pytest.raises(ValueError):
        game.step(Card(CardSuit.HEARTS, Rank.FIVE))
        game.step(Card(CardSuit.HEARTS, Rank.TEN))
        game.step(Card(CardSuit.SPADES, Rank.ACE))


def test_different_suit_trump():
    game = mock_game(seed=0)

    assert game.current_player == PlayerDirection.WEST

    game.step(Card(CardSuit.HEARTS, Rank.FIVE))
    game.step(Card(CardSuit.HEARTS, Rank.TEN))
    game.step(Card(CardSuit.CLUBS, Rank.THREE))
    game.step(Card(CardSuit.HEARTS, Rank.TWO))

    assert game.current_player == PlayerDirection.EAST
    assert len(game.EW_tricks) == 1
    assert len(game.NS_tricks) == 0


def test_different_suit_trump_missing_card():
    game = mock_game(seed=0)

    assert game.current_player == PlayerDirection.WEST

    with pytest.raises(ValueError):
        game.step(Card(CardSuit.HEARTS, Rank.FIVE))
        game.step(Card(CardSuit.HEARTS, Rank.TEN))
        game.step(Card(CardSuit.CLUBS, Rank.ACE))


def test_wrong_card():
    game = mock_game()

    assert game.current_player == PlayerDirection.NORTH

    with pytest.raises(ValueError):
        game.step(Card(CardSuit.HEARTS, Rank.FIVE))
        game.step(Card(CardSuit.HEARTS, Rank.KING))
        game.step(Card(CardSuit.DIAMONDS, Rank.ACE))  # has it, but suit is wrong


def test_trick_first_no_trump_winning_no_trump():
    game = mock_game(seed=0)

    assert game.current_player == PlayerDirection.WEST

    game.step(Card(CardSuit.SPADES, Rank.ACE))
    game.step(Card(CardSuit.HEARTS, Rank.THREE))
    game.step(Card(CardSuit.SPADES, Rank.QUEEN))
    game.step(Card(CardSuit.SPADES, Rank.KING))

    assert game.current_player == PlayerDirection.WEST
    assert len(game.EW_tricks) == 1
    assert len(game.NS_tricks) == 0


def test_trick_first_no_trump_winning_trump():
    game = mock_game(seed=0)

    assert game.current_player == PlayerDirection.WEST

    game.step(Card(CardSuit.SPADES, Rank.ACE))
    game.step(Card(CardSuit.CLUBS, Rank.QUEEN))
    game.step(Card(CardSuit.SPADES, Rank.QUEEN))
    game.step(Card(CardSuit.SPADES, Rank.KING))

    assert game.current_player == PlayerDirection.NORTH
    assert len(game.EW_tricks) == 0
    assert len(game.NS_tricks) == 1


def test_trick_first_trump_winning_trump():
    game = mock_game(seed=0)

    assert game.current_player == PlayerDirection.WEST

    game.step(Card(CardSuit.CLUBS, Rank.SEVEN))
    game.step(Card(CardSuit.CLUBS, Rank.KING))
    game.step(Card(CardSuit.CLUBS, Rank.THREE))
    game.step(Card(CardSuit.CLUBS, Rank.TWO))

    assert game.current_player == PlayerDirection.NORTH
    assert len(game.EW_tricks) == 0
    assert len(game.NS_tricks) == 1
