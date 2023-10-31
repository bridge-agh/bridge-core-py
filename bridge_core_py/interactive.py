import pprint

from bridge_core_py.bids import SpecialBid, Suit, TrickBid, Tricks
from bridge_core_py.cards import Card, Rank
from bridge_core_py.cards import Suit as CardSuit
from bridge_core_py.core import Game, GameStage

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


def game_observation(game: Game):
    print()
    print("-" * 80)
    print()
    print(f"[{ct(game.stage, ORANGE)}]")
    print(f"Current player: {ct(game.current_player, BLUE)}")
    print()

    if game.stage == GameStage.BIDDING:
        print(f"Current deal: {ct(game.bid, ORANGE)}")
        print(f"Current multiplier: {ct(game.multiplier, GREEN)}")
        print(f"Bid history: {ct(game.bid_history, GREEN)}")
        print(f"Current declarer: {ct(game.declarer, BLUE)}")

    if game.stage == GameStage.PLAYING:
        print(f"Deal: {ct(game.bid, ORANGE)}")
        print(
            f"NS tricks: {ct(len(game.NS_tricks), BLUE)} | EW tricks: {ct(len(game.EW_tricks), BLUE)}"
        )
        print(f"Round started by: {ct(game.round_player, BLUE)}")
        print()

        if game.is_dummy_showing_cards:
            print(f"Dummy: {ct(game.declarer, BLUE)}")
            print(f"Dummy cards: {ct(game.players[game.declarer].cards, GREEN)}")
            print()

        print(f"Current trick: {ct(game.round_cards, GREEN)}")
        print(
            f"Current trick winner: {ct(game.trick_check() if game.current_player != game.round_player else None, ORANGE)}"
        )

    if game.stage == GameStage.SCORING:
        print(
            f"NS tricks: {ct(len(game.NS_tricks), BLUE)} | EW tricks: {ct(len(game.EW_tricks), BLUE)}"
        )

        declarer = game.declarer
        if len(game.tricks[declarer]) >= game.bid.tricks.tricks:
            print(
                f"{ct(str(declarer) + str(declarer.next().next()), ORANGE)} won the contract"
            )
        else:
            print(f"{declarer}{declarer.next().next()} lost the contract")

        return

    print()
    print(f"Your actions: {ct(game.actions(), GREEN)}")


def ct(text, color):
    return f"{color}{text}{RESET}"


RED = "\33[91m"
ORANGE = "\33[33m"
GREEN = "\33[92m"
BLUE = "\33[94m"
RESET = "\33[0m"

DEBUG = False


def run():
    game = Game(seed=0)
    pp = pprint.PrettyPrinter(depth=10, sort_dicts=False)

    print(ct("[INFO] You can change the seed in game/interactive_game.py", RED))
    print(
        ct(
            "[INFO] You can disable debug observation info in game/interactive_game.py",
            RED,
        )
    )
    print()

    if DEBUG:
        pp.pprint(game.game_observation())

    # bidding
    while game.stage == GameStage.BIDDING:
        if DEBUG:
            pp.pprint(game.player_observation(game.current_player))

        game_observation(game)

        try:
            bid_input = input(f"{game.current_player}: ")
            bid = string_to_bid(bid_input)
        except ValueError as e:
            print(ct(e, RED))
            continue

        try:
            game.step(bid)
        except ValueError as e:
            print(ct(e, RED))
            continue

    # playing
    while game.stage == GameStage.PLAYING:
        if DEBUG:
            pp.pprint(game.player_observation(game.current_player))

        game_observation(game)

        try:
            card_input = input(f"{game.current_player}: ")
            card = string_to_card(card_input)
        except ValueError as e:
            print(ct(e, RED))
            continue

        try:
            game.step(card)
        except ValueError as e:
            print(ct(e, RED))
            continue

    # scoring
    if game.stage == GameStage.SCORING:
        if DEBUG:
            pp.pprint(game.game_observation())

        game_observation(game)
