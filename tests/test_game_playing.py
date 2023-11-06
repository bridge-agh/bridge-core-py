import pytest
from bridge_core_py.bids import SpecialBid, Suit, TrickBid, Tricks
from bridge_core_py.cards import Card, Rank, Suit as CardSuit
from bridge_core_py.core import Game, GameStage
from bridge_core_py.player import PlayerDirection


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
    assert game.current_player == game.declarer.next()
    assert game.bid == TrickBid(Suit.CLUBS, Tricks.ONE)

    cards = set()
    for player in game.players:
        cards.update(game.players[player].cards)

    assert len(cards) == 52


def test_same_suit():
    game = mock_game()

    assert game.current_player == PlayerDirection.SOUTH

    game.step(Card(CardSuit.DIAMONDS, Rank.JACK))
    game.step(Card(CardSuit.DIAMONDS, Rank.TWO))
    game.step(Card(CardSuit.DIAMONDS, Rank.SIX))
    game.step(Card(CardSuit.DIAMONDS, Rank.KING))

    assert game.current_player == PlayerDirection.EAST
    assert len(game.EW_tricks) == 1
    assert len(game.NS_tricks) == 0


def test_same_suit_missing_card():
    game = mock_game()

    assert game.current_player == PlayerDirection.SOUTH

    with pytest.raises(ValueError):
        game.step(Card(CardSuit.DIAMONDS, Rank.SIX))
        game.step(Card(CardSuit.DIAMONDS, Rank.SIX))


def test_different_suit_no_trump():
    game = mock_game(seed=19)

    assert game.current_player == PlayerDirection.WEST

    game.step(Card(CardSuit.DIAMONDS, Rank.FOUR))
    game.step(Card(CardSuit.SPADES, Rank.KING))
    game.step(Card(CardSuit.DIAMONDS, Rank.SIX))
    game.step(Card(CardSuit.DIAMONDS, Rank.THREE))

    assert game.current_player == PlayerDirection.EAST
    assert len(game.EW_tricks) == 1
    assert len(game.NS_tricks) == 0


def test_different_suit_no_trump_missing_card():
    game = mock_game(seed=19)

    assert game.current_player == PlayerDirection.WEST

    with pytest.raises(ValueError):
        game.step(Card(CardSuit.DIAMONDS, Rank.FOUR))
        game.step(Card(CardSuit.CLUBS, Rank.ACE))


def test_different_suit_trump():
    game = mock_game(seed=19)

    assert game.current_player == PlayerDirection.WEST

    game.step(Card(CardSuit.DIAMONDS, Rank.FOUR))
    game.step(Card(CardSuit.CLUBS, Rank.FIVE))
    game.step(Card(CardSuit.DIAMONDS, Rank.SIX))
    game.step(Card(CardSuit.DIAMONDS, Rank.THREE))

    assert game.current_player == PlayerDirection.NORTH
    assert len(game.EW_tricks) == 0
    assert len(game.NS_tricks) == 1


def test_different_suit_trump_missing_card():
    game = mock_game(seed=19)

    assert game.current_player == PlayerDirection.WEST

    with pytest.raises(ValueError):
        game.step(Card(CardSuit.DIAMONDS, Rank.FOUR))
        game.step(Card(CardSuit.CLUBS, Rank.ACE))


def test_wrong_card():
    game = mock_game()

    assert game.current_player == PlayerDirection.SOUTH

    with pytest.raises(ValueError):
        game.step(Card(CardSuit.HEARTS, Rank.FOUR))
        game.step(Card(CardSuit.HEARTS, Rank.JACK))
        game.step(Card(CardSuit.SPADES, Rank.QUEEN))  # has it, but suit is wrong


def test_trick_first_no_trump_winning_no_trump():
    game = mock_game(seed=19)

    assert game.current_player == PlayerDirection.WEST

    game.step(Card(CardSuit.DIAMONDS, Rank.FOUR))
    game.step(Card(CardSuit.SPADES, Rank.KING))
    game.step(Card(CardSuit.DIAMONDS, Rank.ACE))
    game.step(Card(CardSuit.DIAMONDS, Rank.THREE))

    assert game.current_player == PlayerDirection.EAST
    assert len(game.EW_tricks) == 1
    assert len(game.NS_tricks) == 0


def test_trick_first_no_trump_winning_trump():
    game = mock_game(seed=19)

    assert game.current_player == PlayerDirection.WEST

    game.step(Card(CardSuit.DIAMONDS, Rank.FOUR))
    game.step(Card(CardSuit.CLUBS, Rank.FIVE))
    game.step(Card(CardSuit.DIAMONDS, Rank.SIX))
    game.step(Card(CardSuit.DIAMONDS, Rank.THREE))

    assert game.current_player == PlayerDirection.NORTH
    assert len(game.EW_tricks) == 0
    assert len(game.NS_tricks) == 1


def test_trick_first_trump_winning_trump():
    game = mock_game(seed=19)

    assert game.current_player == PlayerDirection.WEST

    game.step(Card(CardSuit.CLUBS, Rank.TEN))
    game.step(Card(CardSuit.CLUBS, Rank.QUEEN))
    game.step(Card(CardSuit.CLUBS, Rank.SIX))
    game.step(Card(CardSuit.CLUBS, Rank.THREE))

    assert game.current_player == PlayerDirection.NORTH
    assert len(game.EW_tricks) == 0
    assert len(game.NS_tricks) == 1
