from typing import Union
from bridge_core_py.auto_enum import AutoEnum


class Suit(AutoEnum):
    CLUBS = "♣"
    DIAMONDS = "♦"
    HEARTS = "♥"
    SPADES = "♠"
    NO_TRUMP = "NT"

    def __eq__(self, other: "Suit") -> bool:
        return self.value == other.value

    def __lt__(self, other: "Suit") -> bool:
        return self.value < other.value


class Tricks(AutoEnum):
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"

    def __init__(self, display: str):
        super().__init__(display)
        self.tricks = int(display) + 6

    def __lt__(self, other: "Tricks") -> bool:
        return self.tricks < other.tricks


class SpecialBid(AutoEnum):
    PASS = "PASS"
    DOUBLE = "DOUBLE"
    REDOUBLE = "REDOUBLE"


class TrickBid:
    def __init__(self, suit: Suit, tricks: Tricks):
        self.suit = suit
        self.tricks = tricks

    def __str__(self) -> str:
        return f"{self.tricks}{self.suit}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrickBid):
            return False
        return self.suit == other.suit and self.tricks == other.tricks

    def __lt__(self, other: "TrickBid") -> bool:
        if self.tricks.tricks < other.tricks.tricks:
            return True
        elif self.tricks.tricks == other.tricks.tricks:
            return self.suit.value < other.suit.value
        else:
            return False

    def __repr__(self) -> str:
        return self.__str__()


def is_legal(current_bid: TrickBid, bid_history: list[Union[TrickBid, SpecialBid]],
             new_bid: Union[TrickBid, SpecialBid]) -> bool:
    if isinstance(new_bid, SpecialBid):
        if new_bid == SpecialBid.PASS:
            return True
        n_bids = len(bid_history)
        last_bid = bid_history[n_bids - 1] if n_bids > 0 else None
        if new_bid == SpecialBid.DOUBLE:
            if isinstance(last_bid, TrickBid):
                return True
        elif new_bid == SpecialBid.REDOUBLE and n_bids > 0:
            for i in range(0, n_bids - 1):
                # REDOUBLE was already used by player's team
                if bid_history[n_bids - i - 1] == SpecialBid.REDOUBLE and i % 2 != 0:
                    return False
            i = 0
            while n_bids - i - 1 >= 1 and bid_history[n_bids - i - 1] == SpecialBid.PASS:
                i += 1
            # DOUBLE was used by opponent's team
            if bid_history[n_bids - i - 1] == SpecialBid.DOUBLE and i % 2 == 0:
                return True
        return False
    else:
        if current_bid is None:
            return True
        else:
            return new_bid > current_bid
