from bridge_core_py.bids import SpecialBid, TrickBid, Suit, Tricks, is_legal
from bridge_core_py.cards import Suit as CardSuit


def test_tricks_value():
    assert Tricks.ONE.tricks == 7
    assert Tricks.TWO.tricks == 8
    assert Tricks.THREE.tricks == 9
    assert Tricks.FOUR.tricks == 10
    assert Tricks.FIVE.tricks == 11
    assert Tricks.SIX.tricks == 12
    assert Tricks.SEVEN.tricks == 13


def test_islegal():
    assert is_legal(None, [],  TrickBid(Suit.SPADES, Tricks.ONE))
    assert not is_legal(None, [],  SpecialBid.DOUBLE)
    assert is_legal(None, [],  SpecialBid.PASS)
    assert is_legal(TrickBid(Suit.SPADES, Tricks.ONE), [TrickBid(Suit.SPADES, Tricks.ONE)],  TrickBid(Suit.SPADES, Tricks.TWO))
    assert is_legal(TrickBid(Suit.SPADES, Tricks.TWO), [TrickBid(Suit.SPADES, Tricks.TWO)],  TrickBid(Suit.SPADES, Tricks.THREE))
    assert is_legal(TrickBid(Suit.SPADES, Tricks.THREE), [TrickBid(Suit.SPADES, Tricks.THREE)],  TrickBid(Suit.SPADES, Tricks.FOUR))
    assert is_legal(TrickBid(Suit.SPADES, Tricks.FOUR), [TrickBid(Suit.SPADES, Tricks.FOUR)],  TrickBid(Suit.SPADES, Tricks.FIVE))
    assert is_legal(TrickBid(Suit.SPADES, Tricks.FIVE), [TrickBid(Suit.SPADES, Tricks.FIVE)],  TrickBid(Suit.SPADES, Tricks.SIX))
    assert is_legal(TrickBid(Suit.SPADES, Tricks.SIX), [TrickBid(Suit.SPADES, Tricks.SIX)],  TrickBid(Suit.SPADES, Tricks.SEVEN))
    assert not is_legal(TrickBid(Suit.SPADES, Tricks.SEVEN), [TrickBid(Suit.SPADES, Tricks.SEVEN)],  TrickBid(Suit.NO_TRUMP, Tricks.ONE))
    assert is_legal(TrickBid(Suit.NO_TRUMP, Tricks.ONE), [TrickBid(Suit.NO_TRUMP, Tricks.ONE)],  TrickBid(Suit.NO_TRUMP, Tricks.TWO))
    assert is_legal(TrickBid(Suit.NO_TRUMP, Tricks.TWO), [TrickBid(Suit.NO_TRUMP, Tricks.TWO)],  TrickBid(Suit.NO_TRUMP, Tricks.THREE))
    assert is_legal(TrickBid(Suit.NO_TRUMP, Tricks.THREE), [TrickBid(Suit.NO_TRUMP, Tricks.THREE)],  TrickBid(Suit.NO_TRUMP, Tricks.FOUR))
    assert is_legal(TrickBid(Suit.NO_TRUMP, Tricks.FOUR), [TrickBid(Suit.NO_TRUMP, Tricks.FOUR)],  TrickBid(Suit.NO_TRUMP, Tricks.FIVE))
    assert is_legal(TrickBid(Suit.NO_TRUMP, Tricks.FIVE), [TrickBid(Suit.NO_TRUMP, Tricks.FIVE)],  TrickBid(Suit.NO_TRUMP, Tricks.SIX))
    assert is_legal(TrickBid(Suit.NO_TRUMP, Tricks.SIX), [TrickBid(Suit.NO_TRUMP, Tricks.SIX)],  TrickBid(Suit.NO_TRUMP, Tricks.SEVEN))
    assert not is_legal(TrickBid(Suit.NO_TRUMP, Tricks.SEVEN), [TrickBid(Suit.NO_TRUMP, Tricks.SEVEN)],  TrickBid(Suit.SPADES, Tricks.ONE))
    assert not is_legal(TrickBid(Suit.SPADES, Tricks.SEVEN), [TrickBid(Suit.SPADES, Tricks.SEVEN)],  TrickBid(Suit.SPADES, Tricks.ONE))
    assert is_legal(TrickBid(Suit.SPADES, Tricks.SEVEN), [TrickBid(Suit.SPADES, Tricks.SEVEN)],  TrickBid(Suit.NO_TRUMP, Tricks.SEVEN))
    assert is_legal(TrickBid(Suit.HEARTS, Tricks.SEVEN), [TrickBid(Suit.HEARTS, Tricks.SEVEN)],  TrickBid(Suit.SPADES, Tricks.SEVEN))

    assert is_legal(TrickBid(Suit.SPADES, Tricks.SEVEN), [TrickBid(Suit.SPADES, Tricks.SEVEN)],  SpecialBid.DOUBLE)
    assert is_legal(TrickBid(Suit.SPADES, Tricks.SEVEN), [TrickBid(Suit.SPADES, Tricks.SEVEN), SpecialBid.PASS, SpecialBid.PASS],  SpecialBid.DOUBLE)
    assert not is_legal(None, [],  SpecialBid.DOUBLE)
    assert not is_legal(None, [SpecialBid.PASS],  SpecialBid.DOUBLE)
    assert not is_legal(TrickBid(Suit.SPADES, Tricks.SEVEN), [TrickBid(Suit.SPADES, Tricks.SEVEN), SpecialBid.PASS],  SpecialBid.DOUBLE)
    assert not is_legal(TrickBid(Suit.SPADES, Tricks.SEVEN), [TrickBid(Suit.SPADES, Tricks.SEVEN), SpecialBid.DOUBLE],  SpecialBid.DOUBLE)
    assert not is_legal(TrickBid(Suit.SPADES, Tricks.SEVEN), [TrickBid(Suit.SPADES, Tricks.SEVEN), SpecialBid.REDOUBLE],  SpecialBid.DOUBLE)

    assert is_legal(TrickBid(Suit.SPADES, Tricks.SEVEN), [TrickBid(Suit.SPADES, Tricks.SEVEN), SpecialBid.DOUBLE],  SpecialBid.REDOUBLE)
    assert is_legal(TrickBid(Suit.SPADES, Tricks.SEVEN), [TrickBid(Suit.SPADES, Tricks.SEVEN), SpecialBid.DOUBLE, SpecialBid.PASS, SpecialBid.PASS],  SpecialBid.REDOUBLE)

    assert not is_legal(TrickBid(Suit.SPADES, Tricks.SEVEN), [TrickBid(Suit.SPADES, Tricks.SEVEN), SpecialBid.DOUBLE, SpecialBid.PASS, TrickBid(Suit.HEARTS, Tricks.FIVE)],  SpecialBid.REDOUBLE)
    assert not is_legal(TrickBid(Suit.SPADES, Tricks.SEVEN), [TrickBid(Suit.SPADES, Tricks.SEVEN), SpecialBid.DOUBLE, SpecialBid.PASS],  SpecialBid.REDOUBLE)
    assert not is_legal(TrickBid(Suit.SPADES, Tricks.SEVEN), [TrickBid(Suit.SPADES, Tricks.SEVEN)],  SpecialBid.REDOUBLE)
    assert not is_legal(None, [], SpecialBid.REDOUBLE)
    assert not is_legal(None, [SpecialBid.PASS], SpecialBid.REDOUBLE)
    assert not is_legal(TrickBid(Suit.SPADES, Tricks.SEVEN), [TrickBid(Suit.SPADES, Tricks.SEVEN), SpecialBid.DOUBLE, SpecialBid.REDOUBLE, SpecialBid.PASS, TrickBid(Suit.HEARTS, Tricks.FIVE), SpecialBid.DOUBLE],  SpecialBid.REDOUBLE)


def test_card_bids_suit():
    assert Suit.CLUBS == CardSuit.CLUBS
    assert Suit.DIAMONDS == CardSuit.DIAMONDS
    assert Suit.HEARTS == CardSuit.HEARTS
    assert Suit.SPADES == CardSuit.SPADES

    assert not Suit.CLUBS == CardSuit.DIAMONDS
    assert not Suit.DIAMONDS == CardSuit.HEARTS
    assert not Suit.HEARTS == CardSuit.SPADES
    assert not Suit.SPADES == CardSuit.CLUBS

    assert not Suit.NO_TRUMP == CardSuit.CLUBS
    assert not Suit.NO_TRUMP == CardSuit.DIAMONDS
    assert not Suit.NO_TRUMP == CardSuit.HEARTS
    assert not Suit.NO_TRUMP == CardSuit.SPADES
