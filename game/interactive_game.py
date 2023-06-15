from bridge_core_python.bids import SpecialBid, Suit, TrickBid, Tricks
from bridge_core_python.core import Game, GameStage
from bridge_core_python.cards import Card, Rank, Suit as CardSuit
import pprint

str_to_suit = {
    "C": Suit.CLUBS,
    "D": Suit.DIAMONDS,
    "H": Suit.HEARTS,
    "S": Suit.SPADES,
    "NT": Suit.NO_TRUMP,
}

str_to_tricks = {
    "1": Tricks.ONE,
    "2": Tricks.TWO,
    "3": Tricks.THREE,
    "4": Tricks.FOUR,
    "5": Tricks.FIVE,
    "6": Tricks.SIX,
    "7": Tricks.SEVEN,
}

str_to_cardsuit = {
    "C": CardSuit.CLUBS,
    "D": CardSuit.DIAMONDS,
    "H": CardSuit.HEARTS,
    "S": CardSuit.SPADES,
}

str_to_rank = {
    "2": Rank.TWO,
    "3": Rank.THREE,
    "4": Rank.FOUR,
    "5": Rank.FIVE,
    "6": Rank.SIX,
    "7": Rank.SEVEN,
    "8": Rank.EIGHT,
    "9": Rank.NINE,
    "T": Rank.TEN,
    "J": Rank.JACK,
    "Q": Rank.QUEEN,
    "K": Rank.KING,
    "A": Rank.ACE,
}


def string_to_bid(str: str):
    str = str.upper()
    type = None

    if str[0] in "1234567":
        type = TrickBid
    elif str[0] in "DP":
        type = SpecialBid

    if type == TrickBid:
        tricks = str_to_tricks[str[0]]
        suit = str_to_suit[str[1:]]
        return TrickBid(suit, tricks)
    elif type == SpecialBid:
        if str == "D":
            return SpecialBid.DOUBLE
        elif str == "P":
            return SpecialBid.PASS
    else:
        raise ValueError("Invalid bid string")


def string_to_card(str: str):
    str = str.upper()
    if str[0] in "23456789TJQKA":
        rank = str_to_rank[str[0]]
        suit = str_to_cardsuit[str[1]]
        card = Card(suit, rank)
        return card
    else:
        raise ValueError("Invalid card string")


RED = "\33[91m"
RESET = "\33[0m"


def run():
    game = Game(seed=0)

    print(f"{RED}[INFO] You can change the seed in game/game.py{RESET}")
    pp = pprint.PrettyPrinter(depth=10, sort_dicts=False)

    pp.pprint(game.game_observation())

    # bidding
    while game.stage == GameStage.BIDDING:
        pp.pprint(game.player_observation(game.current_player))

        try:
            bid_input = input(f"{game.current_player}: ")
            bid = string_to_bid(bid_input)
        except ValueError as e:
            print(e)
            continue

        try:
            game.step(bid)
        except ValueError as e:
            print(e)
            continue

    # playing
    while game.stage == GameStage.PLAYING:
        pp.pprint(game.player_observation(game.current_player))

        try:
            card_input = input(f"{game.current_player}: ")
            card = string_to_card(card_input)
        except ValueError as e:
            print(e)
            continue

        try:
            game.step(card)
        except ValueError as e:
            print(e)
            continue

    # scoring
    if game.stage == GameStage.SCORING:
        pp.pprint(game.game_observation())
