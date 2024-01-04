import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import pickle
import os
import chex
import sys

from bridge_core_py.core import Card, CardSuit, Rank, TrickBid, SpecialBid, Tricks, Suit
from bridge_core_py.az_network import AlphaZeroNetwork, DiscreteActionHead


sys.path.append(os.path.dirname(__file__))


OBS_HAND_BASE = 428
OBS_BID_BASE = 8
OBS_PASS_BASE = 4


@hk.transform_with_state
def forward(observation):
    chex.assert_shape(observation, [None, 480])
    x = observation.astype(jnp.float32)
    x = hk.Linear(4 * 4 * 32)(x)
    x = jnp.reshape(x, [-1, 4, 4, 32])
    net = AlphaZeroNetwork(action_head=DiscreteActionHead(num_actions=38))
    return net(x, is_training=False)


def make_observation(player_observation):
    x = np.zeros(480)

    bid_history: list[TrickBid | SpecialBid] = player_observation['bidding']['bid_history']
    current_bid = None
    player_idx = player_observation['bidding']['first_dealer'].value
    for bid in bid_history:
        if isinstance(bid, TrickBid):
            current_bid = bid
            offset = (bid.tricks.value * 5 + bid.suit.value) * 12 + player_idx * 3
            x[OBS_BID_BASE + offset] = 1
        elif bid is SpecialBid.PASS:
            if current_bid is None:
                x[OBS_PASS_BASE + player_idx] = 1
        elif bid is SpecialBid.DOUBLE:
            offset = (current_bid.tricks.value * 5 + current_bid.suit.value) * 12 + player_idx * 3
            x[OBS_BID_BASE + offset + 1] = 1
        elif bid is SpecialBid.REDOUBLE:
            offset = (current_bid.tricks.value * 5 + current_bid.suit.value) * 12 + player_idx * 3
            x[OBS_BID_BASE + offset + 2] = 1

    hand: list[Card] = player_observation['hand']
    for card in hand:
        offset = card.suit.value + card.rank.value * 4
        x[OBS_HAND_BASE + offset] = 1

    return jnp.array([x])


IDX_TO_ACTION = [
    SpecialBid.PASS,
    SpecialBid.DOUBLE,
    SpecialBid.REDOUBLE,
] + [
    TrickBid(suit, tricks)
    for tricks in Tricks
    for suit in Suit
]


def idx_to_action(idx):
    return IDX_TO_ACTION[idx]


class Assistant:
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), 'bridge_v1.pkl'), 'rb') as f:
            self.variables = pickle.load(f)
        self.rng = jax.random.key(0)

    def get_bid_action(self, player_observation, legal_actions):
        observation = make_observation(player_observation)
        out, _ = forward.apply(self.variables.params, self.variables.state, None, observation)
        pi = out.pi[0]
        legal_action_mask = jnp.zeros(38, dtype=jnp.bool_)
        for action in legal_actions:
            legal_action_mask = legal_action_mask.at[IDX_TO_ACTION.index(action)].set(True)
        pi = jnp.where(legal_action_mask, pi, -1e9)
        self.rng, subkey = jax.random.split(self.rng)
        action_idx = jax.random.categorical(subkey, pi)
        action = idx_to_action(action_idx)
        if action not in legal_actions:
            action = SpecialBid.PASS
        return action
