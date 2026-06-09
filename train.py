import sys

sys.path.insert(0, "/home/atharvayeole/Projects/vision4leg")

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from vision4leg.envs.a1_mujoco_env import (
    A1MujocoEnv,
    STATE_DIM,
    VISUAL_DIM,
    IMG_HEIGHT,
    IMG_WIDTH,
)
from torchrl.networks.nets import LocoTransformer
from torchrl.networks.base import LocoTransformerEncoder
import gymnasium as gym


class LocoTransformerExtractor(BaseFeaturesExtractor):
    """Wraps LocoTransformer as an SB3 features extractor."""

    def __init__(self, observation_space: gym.spaces.Box):
        # output dim of transformer → 256
        super().__init__(observation_space, features_dim=128)

        self.encoder = LocoTransformerEncoder(
            in_channels=4,  # depth only (4 channels)
            state_input_dim=STATE_DIM,
            hidden_shapes=[256],
            token_dim=64,
        )

        self.net = LocoTransformer(
            encoder=self.encoder,
            output_shape=128,
            state_input_shape=STATE_DIM,
            visual_input_shape=(4, IMG_HEIGHT, IMG_WIDTH),
            transformer_params=[(2, 64)],  # 1 layer, 2 heads, 64 feedforward
            append_hidden_shapes=[128],
            add_ln=True,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (batch, 4126) — split into state and depth
        state = obs[:, :STATE_DIM]
        depth_flat = obs[:, STATE_DIM:]

        # reshape depth to (batch, 4, 64, 64)
        # we use 4 channels to match in_channels=4
        # stack same depth image 4 times (simulate 4-frame history)
        depth_img = depth_flat.view(-1, 1, IMG_HEIGHT, IMG_WIDTH)
        depth_img = depth_img.expand(-1, 4, -1, -1)

        # concatenate back for LocoTransformer
        visual_flat = depth_img.reshape(-1, 4 * IMG_HEIGHT * IMG_WIDTH)
        x = torch.cat([state, visual_flat], dim=-1)

        return self.net(x)


env = A1MujocoEnv(use_depth=True)

policy_kwargs = dict(
    features_extractor_class=LocoTransformerExtractor,
    features_extractor_kwargs={},
    net_arch=[],  # no extra MLP on top, transformer handles it
)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    n_steps=1024,
    batch_size=32,
    learning_rate=1e-4,
    clip_range=0.1,
    tensorboard_log="./logs_transformer/",
)

model.learn(total_timesteps=1_000_000)
model.save("a1_loco_transformer_ppo")
print("Done!")
