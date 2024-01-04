import jax
import chex
import haiku as hk


class ConvolutionalBlock(hk.Module):
    def __init__(self, features, bn_config, name=None):
        super().__init__(name=name)
        self.features = features
        self.bn_config = bn_config

    def __call__(self, x, is_training):
        x = hk.Conv2D(output_channels=self.features, kernel_shape=3)(x)
        x = hk.BatchNorm(**self.bn_config)(x, is_training)
        x = jax.nn.relu(x)
        return x


class ResidualBlock(hk.Module):
    def __init__(self, features, bn_config, name=None):
        super().__init__(name=name)
        self.features = features
        self.bn_config = bn_config

    def __call__(self, x, is_training):
        skip = x
        x = hk.Conv2D(output_channels=self.features, kernel_shape=3)(x)
        x = hk.BatchNorm(**self.bn_config)(x, is_training)
        x = jax.nn.relu(x)
        x = hk.Conv2D(output_channels=self.features, kernel_shape=3)(x)
        x = hk.BatchNorm(**self.bn_config)(x, is_training)
        x = x + skip
        x = jax.nn.relu(x)
        return x


class PolicyHead(hk.Module):
    def __init__(self, action_head, bn_config, name=None):
        super().__init__(name=name)
        self.action_head = action_head
        self.bn_config = bn_config

    def __call__(self, x, is_training):
        x = hk.Conv2D(output_channels=2, kernel_shape=1)(x)
        x = hk.BatchNorm(**self.bn_config)(x, is_training)
        x = jax.nn.relu(x)
        x = self.action_head(x)
        return x


class DiscreteActionHead(hk.Module):
    def __init__(self, num_actions, name=None):
        super().__init__(name=name)
        self.num_actions = num_actions

    def __call__(self, x):
        x = hk.Flatten()(x)
        x = hk.Linear(self.num_actions)(x)
        return x


class ValueHead(hk.Module):
    def __init__(self, bn_config, name=None):
        super().__init__(name=name)
        self.bn_config = bn_config

    def __call__(self, x, is_training):
        x = hk.Conv2D(output_channels=1, kernel_shape=1)(x)
        x = hk.BatchNorm(**self.bn_config)(x, is_training)
        x = jax.nn.relu(x)
        x = hk.Flatten()(x)
        x = hk.Linear(256)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(1)(x).squeeze(-1)
        x = jax.nn.tanh(x)
        return x


@chex.dataclass(frozen=True)
class NetworkVariables:
    params: hk.Params
    state: hk.State


@chex.dataclass(frozen=True)
class NetworkOutputs:
    pi: chex.Array
    v: chex.Array


class AlphaZeroNetwork(hk.Module):
    def __init__(self, action_head, blocks=9, features=64, bn_config=None, name=None):
        super().__init__(name=name)

        bn_config = bn_config or {}
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)
        bn_config.setdefault("decay_rate", 0.99)

        self.action_head = action_head
        self.blocks = blocks
        self.features = features
        self.bn_config = bn_config

    def __call__(self, x, is_training):
        x = ConvolutionalBlock(features=self.features, bn_config=self.bn_config)(x, is_training)
        for _ in range(self.blocks):
            x = ResidualBlock(features=self.features, bn_config=self.bn_config)(x, is_training)
        pi = PolicyHead(action_head=self.action_head, bn_config=self.bn_config)(x, is_training)
        v = ValueHead(bn_config=self.bn_config)(x, is_training)
        return NetworkOutputs(pi=pi, v=v)
