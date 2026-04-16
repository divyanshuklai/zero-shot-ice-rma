import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, List, Optional, Type, Union

class ExtrinsicEncoder(nn.Module):
    """
    Environmental Factor Encoder (mu) from RMA Paper.
    Encodes privileged information et into latent extrinsics zt.
    Architecture: MLP(17 -> 256 -> 128 -> 8)
    """
    def __init__(self, input_dim: int = 17, output_dim: int = 8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.network(x)

class TeacherFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for the Teacher.
    Separates proprioception from privileged info, encodes privileged info,
    and concatenates them for the policy network.
    """
    def __init__(
        self, 
        observation_space: gym.spaces.Dict, 
        extrinsic_input_dim: int = 17, 
        latent_dim: int = 8
    ):
        # We define features_dim as Proprioception (42) + Latent (8) = 50
        proprio_dim = observation_space["proprio"].shape[0]
        super().__init__(observation_space, features_dim=proprio_dim + latent_dim)
        
        self.encoder = ExtrinsicEncoder(extrinsic_input_dim, latent_dim)

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        # observations is a dict because we use a Dict observation space
        proprio = observations["proprio"]
        privileged = observations["privileged"]
        
        # Encode privileged info into latent extrinsics zt
        zt = self.encoder(privileged)
        
        # Concatenate [xt, at-1, zt]
        return th.cat([proprio, zt], dim=1)

class TeacherPolicy(ActorCriticPolicy):
    """
    Teacher Policy implementing the RMA architecture.
    Uses TeacherFeaturesExtractor to process privileged information.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: th.optim.lr_scheduler._LRScheduler,
        *args,
        **kwargs,
    ):
        # The paper specifies a 3-layer MLP with 128 units for the base policy.
        # In SB3, net_arch=[128, 128] creates two hidden layers + the output layer.
        kwargs["net_arch"] = dict(pi=[128, 128], vf=[128, 128])
        kwargs["activation_fn"] = nn.ReLU
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Use our custom feature extractor
            features_extractor_class=TeacherFeaturesExtractor,
            features_extractor_kwargs=dict(extrinsic_input_dim=17, latent_dim=8),
            *args,
            **kwargs,
        )
