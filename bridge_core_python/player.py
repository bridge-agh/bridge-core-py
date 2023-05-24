from bridge_core_python.auto_enum import AutoEnum
from bridge_core_python.cards import Card


class PlayerDirection(AutoEnum):
    NORTH = "N"
    EAST = "E"
    SOUTH = "S"
    WEST = "W"

    def next(self):
        return PlayerDirection((self.value + 1) % len(PlayerDirection))

    def prev(self):
        return PlayerDirection((self.value - 1) % len(PlayerDirection))


class Player:
    
    def __init__(self, cards: list[Card]):
        self.cards = cards

    def __str__(self) -> str:
        return f"{', '.join([str(card) for card in self.cards])}"


