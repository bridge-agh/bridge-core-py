from bridge_core_py.bids import SpecialBid, Suit, TrickBid, Tricks
from bridge_core_py.cards import Card, Rank, Suit as CardSuit
from bridge_core_py.core import Game, GameStage
from bridge_core_py.player import PlayerDirection


def test_basic_game():
    game = Game()

    game.step(TrickBid(Suit.CLUBS, Tricks.ONE))
    game.step(SpecialBid.PASS)
    game.step(SpecialBid.PASS)
    game.step(SpecialBid.PASS)

    # first trick (NS)
    game.step(Card(CardSuit.SPADES, Rank.FIVE)) # S
    game.step(Card(CardSuit.SPADES, Rank.SIX)) # W
    game.step(Card(CardSuit.SPADES, Rank.ACE)) # N
    game.step(Card(CardSuit.SPADES, Rank.THREE)) # E

    assert game.current_player == PlayerDirection.NORTH

    # second trick (EW)
    game.step(Card(CardSuit.SPADES, Rank.QUEEN)) # N
    game.step(Card(CardSuit.SPADES, Rank.FOUR)) # E
    game.step(Card(CardSuit.SPADES, Rank.SEVEN)) # S
    game.step(Card(CardSuit.SPADES, Rank.KING)) # W

    assert game.current_player == PlayerDirection.WEST

    # third trick (EW)
    game.step(Card(CardSuit.CLUBS, Rank.ACE)) # W
    game.step(Card(CardSuit.CLUBS, Rank.SEVEN)) # N
    game.step(Card(CardSuit.CLUBS, Rank.SIX)) # E
    game.step(Card(CardSuit.CLUBS, Rank.FIVE)) # S

    assert game.current_player == PlayerDirection.WEST

    # fourth trick (NS)
    game.step(Card(CardSuit.HEARTS, Rank.QUEEN)) # W
    game.step(Card(CardSuit.HEARTS, Rank.ACE)) # N
    game.step(Card(CardSuit.HEARTS, Rank.TWO)) # E
    game.step(Card(CardSuit.HEARTS, Rank.FOUR)) # S

    assert game.current_player == PlayerDirection.NORTH

    # fifth trick (EW)
    game.step(Card(CardSuit.SPADES, Rank.TWO)) # N
    game.step(Card(CardSuit.SPADES, Rank.JACK)) # E
    game.step(Card(CardSuit.SPADES, Rank.EIGHT)) # S
    game.step(Card(CardSuit.CLUBS, Rank.TWO)) # W

    assert game.current_player == PlayerDirection.WEST

    # sixth trick (EW)
    game.step(Card(CardSuit.HEARTS, Rank.NINE)) # W
    game.step(Card(CardSuit.HEARTS, Rank.TEN)) # N
    game.step(Card(CardSuit.HEARTS, Rank.KING)) # E
    game.step(Card(CardSuit.HEARTS, Rank.SIX)) # S

    assert game.current_player == PlayerDirection.EAST

    # seventh trick (EW)
    game.step(Card(CardSuit.HEARTS, Rank.SEVEN)) # E
    game.step(Card(CardSuit.HEARTS, Rank.EIGHT)) # S
    game.step(Card(CardSuit.HEARTS, Rank.JACK)) # W
    game.step(Card(CardSuit.HEARTS, Rank.THREE)) # N

    assert game.current_player == PlayerDirection.WEST

    # eighth trick (EW)
    game.step(Card(CardSuit.DIAMONDS, Rank.EIGHT)) # W
    game.step(Card(CardSuit.DIAMONDS, Rank.QUEEN)) # N
    game.step(Card(CardSuit.DIAMONDS, Rank.KING)) # E
    game.step(Card(CardSuit.DIAMONDS, Rank.FOUR)) # S

    assert game.current_player == PlayerDirection.EAST

    # ninth trick (EW)
    game.step(Card(CardSuit.DIAMONDS, Rank.ACE)) # E
    game.step(Card(CardSuit.DIAMONDS, Rank.FIVE)) # S
    game.step(Card(CardSuit.DIAMONDS, Rank.TWO)) # W
    game.step(Card(CardSuit.DIAMONDS, Rank.SIX)) # N

    assert game.current_player == PlayerDirection.EAST

    # tenth trick (NS)
    game.step(Card(CardSuit.DIAMONDS, Rank.TEN)) # E
    game.step(Card(CardSuit.DIAMONDS, Rank.JACK)) # S
    game.step(Card(CardSuit.DIAMONDS, Rank.THREE)) # W
    game.step(Card(CardSuit.DIAMONDS, Rank.NINE)) # N

    assert game.current_player == PlayerDirection.SOUTH

    # eleventh trick (NS)
    game.step(Card(CardSuit.CLUBS, Rank.KING)) # S
    game.step(Card(CardSuit.CLUBS, Rank.THREE)) # W
    game.step(Card(CardSuit.CLUBS, Rank.NINE)) # N
    game.step(Card(CardSuit.CLUBS, Rank.JACK)) # E

    assert game.current_player == PlayerDirection.SOUTH

    # twelfth trick (NS)
    game.step(Card(CardSuit.CLUBS, Rank.QUEEN)) # S
    game.step(Card(CardSuit.CLUBS, Rank.FOUR)) # W
    game.step(Card(CardSuit.HEARTS, Rank.FIVE)) # N
    game.step(Card(CardSuit.DIAMONDS, Rank.SEVEN)) # E

    assert game.current_player == PlayerDirection.SOUTH

    # thirteenth trick (EW)
    game.step(Card(CardSuit.CLUBS, Rank.EIGHT)) # S
    game.step(Card(CardSuit.CLUBS, Rank.TEN)) # W
    game.step(Card(CardSuit.SPADES, Rank.TEN)) # N
    game.step(Card(CardSuit.SPADES, Rank.NINE)) # E

    assert game.current_player == PlayerDirection.WEST

    assert game.stage == GameStage.SCORING
    assert len(game.NS_tricks) == 5
    assert len(game.EW_tricks) == 8

    assert game.NS_tricks[0][0] == PlayerDirection.SOUTH
    assert game.NS_tricks[0][1] == PlayerDirection.NORTH
    assert game.NS_tricks[0][2] == [
        Card(CardSuit.SPADES, Rank.FIVE),
        Card(CardSuit.SPADES, Rank.SIX),
        Card(CardSuit.SPADES, Rank.ACE),
        Card(CardSuit.SPADES, Rank.THREE),
    ]

    assert game.NS_tricks[1][0] == PlayerDirection.WEST
    assert game.NS_tricks[1][1] == PlayerDirection.NORTH
    assert game.NS_tricks[1][2] == [
        Card(CardSuit.HEARTS, Rank.QUEEN),
        Card(CardSuit.HEARTS, Rank.ACE),
        Card(CardSuit.HEARTS, Rank.TWO),
        Card(CardSuit.HEARTS, Rank.FOUR),
    ]

    assert game.NS_tricks[2][0] == PlayerDirection.EAST
    assert game.NS_tricks[2][1] == PlayerDirection.SOUTH
    assert game.NS_tricks[2][2] == [
        Card(CardSuit.DIAMONDS, Rank.TEN),
        Card(CardSuit.DIAMONDS, Rank.JACK),
        Card(CardSuit.DIAMONDS, Rank.THREE),
        Card(CardSuit.DIAMONDS, Rank.NINE),
    ]

    assert game.NS_tricks[3][0] == PlayerDirection.SOUTH
    assert game.NS_tricks[3][1] == PlayerDirection.SOUTH
    assert game.NS_tricks[3][2] == [
        Card(CardSuit.CLUBS, Rank.KING),
        Card(CardSuit.CLUBS, Rank.THREE),
        Card(CardSuit.CLUBS, Rank.NINE),
        Card(CardSuit.CLUBS, Rank.JACK),
    ]

    assert game.NS_tricks[4][0] == PlayerDirection.SOUTH
    assert game.NS_tricks[4][1] == PlayerDirection.SOUTH
    assert game.NS_tricks[4][2] == [
        Card(CardSuit.CLUBS, Rank.QUEEN),
        Card(CardSuit.CLUBS, Rank.FOUR),
        Card(CardSuit.HEARTS, Rank.FIVE),
        Card(CardSuit.DIAMONDS, Rank.SEVEN),
    ]

    assert game.EW_tricks[0][0] == PlayerDirection.NORTH
    assert game.EW_tricks[0][1] == PlayerDirection.WEST
    assert game.EW_tricks[0][2] == [
        Card(CardSuit.SPADES, Rank.QUEEN),
        Card(CardSuit.SPADES, Rank.FOUR),
        Card(CardSuit.SPADES, Rank.SEVEN),
        Card(CardSuit.SPADES, Rank.KING),
    ]

    assert game.EW_tricks[1][0] == PlayerDirection.WEST
    assert game.EW_tricks[1][1] == PlayerDirection.WEST
    assert game.EW_tricks[1][2] == [
        Card(CardSuit.CLUBS, Rank.ACE),
        Card(CardSuit.CLUBS, Rank.SEVEN),
        Card(CardSuit.CLUBS, Rank.SIX),
        Card(CardSuit.CLUBS, Rank.FIVE),
    ]

    assert game.EW_tricks[2][0] == PlayerDirection.NORTH
    assert game.EW_tricks[2][1] == PlayerDirection.WEST
    assert game.EW_tricks[2][2] == [
        Card(CardSuit.SPADES, Rank.TWO),
        Card(CardSuit.SPADES, Rank.JACK),
        Card(CardSuit.SPADES, Rank.EIGHT),
        Card(CardSuit.CLUBS, Rank.TWO),
    ]

    assert game.EW_tricks[3][0] == PlayerDirection.WEST
    assert game.EW_tricks[3][1] == PlayerDirection.EAST
    assert game.EW_tricks[3][2] == [
        Card(CardSuit.HEARTS, Rank.NINE),
        Card(CardSuit.HEARTS, Rank.TEN),
        Card(CardSuit.HEARTS, Rank.KING),
        Card(CardSuit.HEARTS, Rank.SIX),
    ]

    assert game.EW_tricks[4][0] == PlayerDirection.EAST
    assert game.EW_tricks[4][1] == PlayerDirection.WEST
    assert game.EW_tricks[4][2] == [
        Card(CardSuit.HEARTS, Rank.SEVEN),
        Card(CardSuit.HEARTS, Rank.EIGHT),
        Card(CardSuit.HEARTS, Rank.JACK),
        Card(CardSuit.HEARTS, Rank.THREE),
    ]

    assert game.EW_tricks[5][0] == PlayerDirection.WEST
    assert game.EW_tricks[5][1] == PlayerDirection.EAST
    assert game.EW_tricks[5][2] == [
        Card(CardSuit.DIAMONDS, Rank.EIGHT),
        Card(CardSuit.DIAMONDS, Rank.QUEEN),
        Card(CardSuit.DIAMONDS, Rank.KING),
        Card(CardSuit.DIAMONDS, Rank.FOUR),
    ]

    assert game.EW_tricks[6][0] == PlayerDirection.EAST
    assert game.EW_tricks[6][1] == PlayerDirection.EAST
    assert game.EW_tricks[6][2] == [
        Card(CardSuit.DIAMONDS, Rank.ACE),
        Card(CardSuit.DIAMONDS, Rank.FIVE),
        Card(CardSuit.DIAMONDS, Rank.TWO),
        Card(CardSuit.DIAMONDS, Rank.SIX),
    ]

    assert game.EW_tricks[7][0] == PlayerDirection.SOUTH
    assert game.EW_tricks[7][1] == PlayerDirection.WEST
    assert game.EW_tricks[7][2] == [
        Card(CardSuit.CLUBS, Rank.EIGHT),
        Card(CardSuit.CLUBS, Rank.TEN),
        Card(CardSuit.SPADES, Rank.TEN),
        Card(CardSuit.SPADES, Rank.NINE),
    ]

