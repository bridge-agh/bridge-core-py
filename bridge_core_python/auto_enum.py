from enum import Enum


class AutoEnum(Enum):

    def __new__(cls, *args):
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, display: str):
        self.display = display
    
    def __str__(self) -> str:
        return self.display
