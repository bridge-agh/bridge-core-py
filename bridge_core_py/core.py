from typing import Union

import numpy as np

from bridge_core_py.auto_enum import AutoEnum
from bridge_core_py.bids import SpecialBid, Suit, TrickBid, Tricks, is_legal
from bridge_core_py.cards import Card, Rank
from bridge_core_py.cards import Suit as CardSuit
from bridge_core_py.player import Player, PlayerDirection


class GameStage(AutoEnum):
    BIDDING = "BIDDING"
    PLAYING = "PLAYING"
    SCORING = "SCORING"


class AssistantAction:
    pass


class Game:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed=seed)

        # shuffle deck
        deck = [Card(suit, rank) for suit in CardSuit for rank in Rank]
        self.rng.shuffle(deck)

        # deal cards
        self.players = {
            PlayerDirection.NORTH: Player(sorted(deck[0:13])),
            PlayerDirection.EAST: Player(sorted(deck[13:26])),
            PlayerDirection.SOUTH: Player(sorted(deck[26:39])),
            PlayerDirection.WEST: Player(sorted(deck[39:52])),
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
        self.current_player = self.rng.choice(list(self.players.keys()))
        self.first_dealer = self.current_player

        self.stage = GameStage.BIDDING

        self.bid_history: list[Union[TrickBid, SpecialBid]] = []
        self.bid = None
        self.declarer = None
        self.multiplier = 1

        self.is_dummy_showing_cards = False
        self.round_player = None
        self.round_cards: list[Card] = []

    def card_check(self, card: Card):
        # card must be in player's hand
        if card not in self.players[self.current_player].cards:
            return False

        # first card can be any card
        if self.round_player == self.current_player:
            return True

        # card must be in the same suit as the first card
        elif self.round_cards[0].suit == card.suit:
            return True

        # if player has no cards in the same suit as the first card
        # then any card can be played
        elif not any(
            self.round_cards[0].suit == cardi.suit
            for cardi in self.players[self.current_player].cards
        ):
            return True

        return False

    def trick_check(self):
        winning_card = self.round_cards[0]
        winning_player_index = 0
        for i, card in enumerate(self.round_cards[1:]):
            # if suit is the same as winning and rank is higher then card wins
            # if trump was played before, only trump cards can win
            # if card has trump suit then it will be checked here
            if card.suit == winning_card.suit:
                if card.rank > winning_card.rank:
                    winning_card = card
                    winning_player_index = i + 1

            # if card is trump then the winning card is not trump
            # because of the previous if
            elif card.suit == self.bid.suit:
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
                return [bid for bid in SpecialBid if is_legal(self.bid, self.bid_history, bid)] + [
                    TrickBid(suit, trick)
                    for suit in Suit
                    for trick in Tricks
                    if trick > self.bid.tricks
                    or (trick == self.bid.tricks and suit > self.bid.suit)
                ]
        elif self.stage == GameStage.PLAYING:
            if self.round_player != self.current_player:
                # if we have a card to the suit of the bid
                if any(
                    card.suit == self.round_cards[0].suit
                    for card in self.players[self.current_player].cards
                ):
                    return [
                        card
                        for card in self.players[self.current_player].cards
                        if card.suit == self.round_cards[0].suit
                    ]

            # if first card of the round or no cards to the suit of the bid
            return self.players[self.current_player].cards

    def step(self, action: Union[Card, TrickBid, SpecialBid, AssistantAction]):
        if isinstance(action, AssistantAction):
            self.step(self.decide_with_assistant())
            return

        if self.stage == GameStage.BIDDING:
            assert isinstance(action, (TrickBid, SpecialBid))
            if not is_legal(current_bid=self.bid, bid_history=self.bid_history, new_bid=action):
                raise ValueError("Illegal bid")

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
                self.current_player = self.declarer.next()
                self.round_player = self.current_player
                return

            ### new bid cases

            # played trick bid
            if isinstance(action, TrickBid):
                self.bid = action
                self.declarer = self.current_player
                self.multiplier = 1
            # double current bid or redouble doubled bid
            elif action in [SpecialBid.DOUBLE, SpecialBid.REDOUBLE]:
                self.multiplier *= 2
        elif self.stage == GameStage.PLAYING:
            assert isinstance(action, Card)
            if not self.card_check(action):
                raise ValueError("Illegal card")

            self.is_dummy_showing_cards = True

            self.round_cards.append(action)
            self.players[self.current_player].cards.remove(action)

            if len(self.round_cards) == 4:
                winning_player = self.trick_check()
                self.tricks[winning_player].append(
                    (self.round_player, winning_player, self.round_cards)
                )

                # reset round
                self.round_cards = []
                self.round_player = winning_player
                self.current_player = winning_player

                # check if game is over
                if len(self.NS_tricks) + len(self.EW_tricks) == 13:
                    self.stage = GameStage.SCORING
                    return

                return

        self.current_player = self.current_player.next()

    def player_observation(self, player: PlayerDirection):
        return {
            "game_stage": self.stage,
            "current_player": self.current_player,
            "bidding": {
                "first_dealer": self.first_dealer,
                "bid_history": self.bid_history,
                "bid": self.bid,
                "declarer": self.declarer,
                "multiplier": self.multiplier,
            },
            "game": {
                "round_player": self.round_player,
                "round_cards": self.round_cards,
                "dummy": (
                    self.players[self.declarer].cards
                    if player is self.declarer.next().next()
                    else self.players[self.declarer.next().next()].cards
                )
                if self.is_dummy_showing_cards
                else [],
                "tricks": {
                    "NS": self.NS_tricks,
                    "EW": self.EW_tricks,
                },
            },
            "hand": self.players[player].cards,
        }

    def game_observation(self):
        return {
            "game_stage": self.stage,
            "current_player": self.current_player,
            "bidding": {
                "first_dealer": self.first_dealer,
                "bid_history": self.bid_history,
                "bid": self.bid,
                "declarer": self.declarer,
                "multiplier": self.multiplier,
            },
            "game": {
                "round_player": self.round_player,
                "round_cards": self.round_cards,
                "tricks": {
                    "NS": self.NS_tricks,
                    "EW": self.EW_tricks,
                },
            },
            "hands": {
                "N": self.players[PlayerDirection.NORTH].cards,
                "E": self.players[PlayerDirection.EAST].cards,
                "W": self.players[PlayerDirection.WEST].cards,
                "S": self.players[PlayerDirection.SOUTH].cards,
            },
            "handsPBN": self.hands_to_str()
        }

    def hands_to_str(self):
        hands_str = f"{self.first_dealer}:"
        suits = [suit for suit in CardSuit][::-1]
        directions = [d for d in PlayerDirection]
        i = directions.index(self.declarer)
        directions = directions[i:] + directions[:i]
        for direction in directions:
            player = self.players[direction]
            cards = player.cards[::-1]
            # print(direction, cards)

            current_suit_index = 0
            for card in cards:
                while card.suit != suits[current_suit_index]:
                    current_suit_index += 1
                    hands_str += "."
                hands_str += card.rank.display
            hands_str += " "
        hands_str = hands_str[:-1]
        return hands_str

    def decide_with_assistant(self):
        bid_length = len(self.bid_history)
        if bid_length >= 6 and self.stage is GameStage.BIDDING:
            action = SpecialBid.PASS
        else:
            actions = self.actions()
            action = self.rng.choice(actions)
        return action
