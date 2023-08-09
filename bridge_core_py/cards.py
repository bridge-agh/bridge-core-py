from bridge_core_py.auto_enum import AutoEnum


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

    def __eq__(self, other: "Rank") -> bool:
        return self.value == other.value
    
    def __lt__(self, other: "Rank") -> bool:
        return self.value < other.value


class Card:
    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit
        self.rank = rank

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __eq__(self, other: "Card") -> bool:
        return self.suit == other.suit and self.rank == other.rank

    def __lt__(self, other: "Card") -> bool:
        if self.suit == other.suit:
            return self.rank.value < other.rank.value
        else:
            return self.suit.value < other.suit.value
    
    # for tests
    def __hash__(self) -> int:
        return self.__repr__().__hash__()

    