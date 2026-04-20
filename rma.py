
import jax
import jax.numpy as jnp
from flax import linen as nn
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.networks import MLP
from typing import Sequence, Tuple

class TeacherEncoder(nn.Module):
    """Encodes privileged information into a latent extrinsics vector."""
    latent_size: int = 8
    layer_sizes: Sequence[int] = (256, 128)

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        for size in self.layer_sizes:
            x = nn.Dense(size)(x)
            x = nn.relu(x)
        x = nn.Dense(self.latent_size)(x)
        return x

class TeacherNetwork(nn.Module):
    """Actor-Critic network that takes proprioceptive state and extrinsics."""
    action_size: int
    policy_layer_sizes: Sequence[int] = (512, 256, 128)
    value_layer_sizes: Sequence[int] = (512, 256, 128)

    @nn.compact
    def __call__(self, state: jnp.ndarray, extrinsics: jnp.ndarray):
        # RMA teacher policy takes (xt, zt)
        # Note: at-1 is usually included in xt in mujoco_playground environments
        x = jnp.concatenate([state, extrinsics], axis=-1)
        
        # Policy
        policy_mlp = MLP(
            layer_sizes=list(self.policy_layer_sizes) + [self.action_size],
            activation=nn.relu,
            kernel_init=nn.initializers.lecun_normal(),
        )
        logits = policy_mlp(x)
        
        # Value
        value_mlp = MLP(
            layer_sizes=list(self.value_layer_sizes) + [1],
            activation=nn.relu,
            kernel_init=nn.initializers.lecun_normal(),
        )
        value = value_mlp(x)
        value = jnp.squeeze(value, axis=-1)
        
        return logits, value

class StudentEncoder(nn.Module):
    """Adaptation module that predicts extrinsics from state-action history."""
    latent_size: int = 8

    @nn.compact
    def __call__(self, history: jnp.ndarray):
        # history shape: [batch, time, state_action_dim]
        # time dimension is typically 50
        
        x = nn.Dense(32)(history)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        
        # 1-D CNN: (out_channels, kernel_size, stride)
        x = nn.Conv(features=32, kernel_size=(8,), strides=(4,), padding='VALID')(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=32, kernel_size=(5,), strides=(1,), padding='VALID')(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=32, kernel_size=(5,), strides=(1,), padding='VALID')(x)
        x = nn.relu(x)
        
        x = x.reshape((x.shape[0], -1))  # Flatten
        z_hat = nn.Dense(self.latent_size)(x)
        
        return z_hat
