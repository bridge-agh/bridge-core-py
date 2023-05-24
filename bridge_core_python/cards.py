from bridge_core_python.auto_enum import AutoEnum


class Suit(AutoEnum):
    CLUBS = "♣"
    DIAMONDS = "♦"
    HEARTS = "♥"
    SPADES = "♠"

    def __eq__(self, other: "Suit") -> bool:
        return self.value == other.value


class Rank(AutoEnum):
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"


class Card:
    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit
        self.rank = rank

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    