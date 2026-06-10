import sys
import os
import glob
from typing import Callable

import torch
import torch.nn as nn
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

# Path injection
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from vision4leg.envs.a1_mujoco_env import (
    A1MujocoEnv,
    STATE_DIM,
    VISUAL_DIM,
    IMG_HEIGHT,
    IMG_WIDTH,
)
from torchrl.networks.nets import LocoTransformer
from torchrl.networks.base import LocoTransformerEncoder


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def make_env():
    """Create env WITH depth. Requires DummyVecEnv on Mac to avoid OpenGL context crashes."""
    return Monitor(A1MujocoEnv(use_depth=True))


class LocoTransformerExtractor(BaseFeaturesExtractor):
    """Wraps LocoTransformer as an SB3 features extractor."""

    def __init__(self, observation_space: gym.spaces.Box):
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
            transformer_params=[(2, 64)],
            append_hidden_shapes=[128],
            add_ln=True,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        state = obs[:, :STATE_DIM]
        depth_flat = obs[:, STATE_DIM:]

        depth_img = depth_flat.view(-1, 1, IMG_HEIGHT, IMG_WIDTH)
        depth_img = depth_img.expand(-1, 4, -1, -1)

        visual_flat = depth_img.reshape(-1, 4 * IMG_HEIGHT * IMG_WIDTH)
        x = torch.cat([state, visual_flat], dim=-1)

        return self.net(x)


if __name__ == "__main__":
    num_envs = 8
    TOTAL_TIMESTEPS = 2_000_000
    CHECKPOINT_DIR = "./checkpoints/"
    VECNORM_PATH = "vec_normalize.pkl"

    env = DummyVecEnv([make_env for _ in range(num_envs)])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    policy_kwargs = dict(
        features_extractor_class=LocoTransformerExtractor,
        features_extractor_kwargs={},
        net_arch=[],  # Transformer handles everything
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="mps",
        n_steps=1024,  # 1024 * 16 = 16k buffer
        batch_size=4096,
        n_epochs=5,
        learning_rate=linear_schedule(1e-4),
        clip_range=0.1,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        tensorboard_log="./logs_transformer/",
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // num_envs, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="a1_loco",
    )

    print(f"\n{'=' * 60}")
    print(f"  LocoTransformer — Torque Control — {num_envs} envs (DummyVecEnv)")
    print(f"{'=' * 60}\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_cb,
    )

    model.save("a1_loco_transformer_ppo")
    env.save(VECNORM_PATH)
    print("Training complete! Model saved.")
