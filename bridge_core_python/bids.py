from typing import Union
from bridge_core_python.auto_enum import AutoEnum


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


class TrickBid:
    def __init__(self, suit: Suit, tricks: Tricks):
        self.suit = suit
        self.tricks = tricks

    def __str__(self) -> str:
        return f"{self.tricks}{self.suit}"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrickBid): return False
        return self.suit == other.suit and self.tricks == other.tricks
        

    def __lt__(self, other: "TrickBid") -> bool:
        if self.tricks.tricks < other.tricks.tricks:
            return True
        elif self.tricks.tricks == other.tricks.tricks:
            return self.suit.value < other.suit.value
        else:
            return False
        
    def __str__(self) -> str:
        return f"{self.tricks}{self.suit}"

    def __repr__(self) -> str:
        return self.__str__()


def is_legal(current_bid: TrickBid, new_bid: Union[TrickBid, SpecialBid]) -> bool:
    if isinstance(new_bid, SpecialBid):
        if current_bid is None and new_bid == SpecialBid.DOUBLE:
            return False
        return True
    else:
        if current_bid is None:
            return True
        else:
            return new_bid > current_bid
