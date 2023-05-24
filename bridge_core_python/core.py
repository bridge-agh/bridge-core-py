from typing import Union
from bridge_core_python.auto_enum import AutoEnum
from bridge_core_python.cards import Card, Rank
from bridge_core_python.bids import Suit, TrickBid, SpecialBid, Tricks, is_legal
import numpy as np

from player import Player, PlayerDirection


class GameStage(AutoEnum):
    BIDDING = "BIDDING"
    PLAYING = "PLAYING"
    SCORING = "SCORING"


class Game:
    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed=seed)

        # shuffle dekc
        deck = [Card(suit, rank) for suit in Suit for rank in Rank]
        rng.shuffle(deck)

        # deal cards
        self.players = {
            PlayerDirection.NORTH: Player(deck[0:13]),
            PlayerDirection.EAST: Player(deck[13:26]),
            PlayerDirection.SOUTH: Player(deck[26:39]),
            PlayerDirection.WEST: Player(deck[39:52]),
        }

        # define tricks
        self.NS_tricks = []
        self.EW_tricks = []

        self.tricks = {
            PlayerDirection.NORTH: self.NS_tricks,
            PlayerDirection.EAST: self.EW_tricks,
            PlayerDirection.SOUTH: self.NS_tricks,
            PlayerDirection.WEST: self.EW_tricks,
        }

        # determine dealer
        self.current_player = rng.choice(list(self.players.keys()))

        self.stage = GameStage.BIDDING

        self.bid_history: list[Union[TrickBid, SpecialBid]] = []
        self.bid = None
        self.declarer = None
        self.multiplier = 1

        self.is_dziadek_showing_karty = False
        self.round_player = None
        self.round_cards: list[Card] = []

    def card_check(self, card: Card):
        # czy gracz ma taka karte
        if not card in self.players[self.current_player].cards:
            return False

        # pierwszy rzut dowolna karta
        if self.round_player == self.current_player:
            return True

        # karta musi byc tego samego koloru co pierwsza karta
        elif self.round_cards[0].suit == card.suit:
            return True

        # jezeli gracz nie ma karty w tym kolorze to dowolna karta
        elif not any(
            card.suit == card.suit for card in self.players[self.current_player].cards
        ):
            return True

        return False

    def trick_check(self):
        round_trump = self.round_cards[0].suit

        winning_card = self.round_cards[0]
        winning_player_index = 0
        for card, i in enumerate(self.round_cards[1:]):
            # karta jest tego samego koloru co pierwsza karta i jest wieksza od wygrywajacej karty
            if card.suit == round_trump and card.rank > winning_card.rank:
                winning_card = card
                winning_player_index = i + 1

            # karta jest atutem
            elif card.suit == self.bid.suit:
                # wygrywajaca karta tez jest atutem
                if winning_card.suit == card.suit:
                    # karta jest silniejsza od wygrywajacej
                    if card.rank > winning_card.rank:
                        winning_card = card
                        winning_player_index = i + 1
                # wygrywajaca karta nie jest atutem
                else:
                    winning_card = card
                    winning_player_index = i + 1

        winning_player = self.round_player
        for i in range(winning_player_index):
            winning_player = winning_player.next()

        return winning_player

    def actions(self):
        if self.stage == GameStage.BIDDING:
            if self.bid is None:
                return [SpecialBid.PASS] + [
                    TrickBid(suit, trick) for suit in Suit for trick in Tricks
                ]
            else:
                return [SpecialBid.PASS, SpecialBid.DOUBLE] + [
                    TrickBid(suit, trick)
                    for suit in Suit
                    for trick in Tricks
                    if trick > self.bid.tricks
                    or (trick == self.bid.tricks and suit > self.bid.suit)
                ]
        elif self.stage == GameStage.PLAYING:
            # jezeli mamy karty do koloru
            if any(
                card.suit == self.bid.suit
                for card in self.players[self.current_player].cards
            ):
                return [
                    card
                    for card in self.players[self.current_player].cards
                    if card.suit == self.bid.suit
                ]

            # jezeli nie mamy karty do koloru
            else:
                return self.players[self.current_player].cards

    def step(self, action: Union[Card, TrickBid, SpecialBid]):
        if self.stage == GameStage.BIDDING:
            assert isinstance(action, (TrickBid, SpecialBid))
            assert is_legal(self.bid, action)

            self.bid_history.append(action)

            ### end bidding cases

            # if first 4 bids are pass, end bidding, end game
            if self.bid is None and len(self.bid_history) == 4:
                self.stage = GameStage.SCORING
                return

            # if 3 consecutive passes, end bidding, start playing
            if len(self.bid_history) >= 4 and all(
                bid == SpecialBid.PASS for bid in self.bid_history[-3:]
            ):
                self.stage = GameStage.PLAYING
                self.current_player = self.declarer.prev()
                self.round_player = self.current_player
                return

            ### new bid cases

            # played trick bid
            if isinstance(action, TrickBid):
                self.bid = action
                self.declarer = self.current_player
                self.multiplier = 1
            # doubled current bid
            elif action == SpecialBid.DOUBLE:
                self.multiplier = 2

        elif self.stage == GameStage.PLAYING:
            assert isinstance(action, Card)
            assert self.card_check(action)

            self.is_dziadek_showing_karty = True

            self.round_cards.append(action)
            self.players[self.current_player].cards.remove(action)

            if len(self.round_cards) == 4:
                winning_player = self.trick_check()
                self.tricks[winning_player].append(
                    self.round_player, winning_player, self.round_cards
                )

                # check if game is over
                if len(self.NS_tricks) + len(self.EW_tricks) == 13:
                    self.stage = GameStage.SCORING
                    return

                # reset round
                self.round_cards = []
                self.round_player = winning_player
                self.current_player = winning_player
                return

        self.current_player = self.current_player.next()
