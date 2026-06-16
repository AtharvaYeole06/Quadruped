import sys
import os
import time
import torch
import numpy as np
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
curr_dir = os.path.dirname(__file__)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from vision4leg.envs.a1_mujoco_env import A1MujocoEnv
from legacy_nets import LocoTransformer
from torchrl.networks.base import LocoTransformerEncoder
import train
import test as root_test

class LegacyObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.old_state_dim = 30
        self.visual_dim = 4 * 64 * 64
        self.observation_space = Box(
            low=-np.inf, high=np.inf, 
            shape=(self.old_state_dim + self.visual_dim,), 
            dtype=np.float32
        )
        
    def observation(self, obs):
        state = obs[:self.old_state_dim]
        visual = obs[-self.visual_dim:]
        return np.concatenate([state, visual])

class LegacyLocoTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box):
        super().__init__(observation_space, features_dim=128)
        self.encoder = LocoTransformerEncoder(
            in_channels=4,
            state_input_dim=30,
            hidden_shapes=[256],
            token_dim=64,
        )
        self.net = LocoTransformer(
            encoder=self.encoder,
            output_shape=128,
            state_input_shape=30,
            visual_input_shape=(4, 64, 64),
            transformer_params=[(2, 128), (2, 128)],
            append_hidden_shapes=[128],
            add_ln=True,
            token_norm=True,
        )
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

train.LocoTransformerExtractor = LegacyLocoTransformerExtractor
root_test.LocoTransformerExtractor = LegacyLocoTransformerExtractor
sys.modules["__main__"].LocoTransformerExtractor = LegacyLocoTransformerExtractor

def make_env():
    env = A1MujocoEnv(render_mode="human", use_depth=True)
    return LegacyObservationWrapper(env)

if __name__ == "__main__":
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "a1_loco_transformer_ppo.zip")
    VECNORM_PATH = os.path.join(os.path.dirname(__file__), "vec_normalize.pkl")

    env = DummyVecEnv([make_env])
    
    if os.path.exists(VECNORM_PATH):
        env = VecNormalize.load(VECNORM_PATH, env)
        env.training = False
        env.norm_reward = False

    custom_objects = {
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.2,
        "policy_kwargs": dict(
            features_extractor_class=LegacyLocoTransformerExtractor,
            net_arch=[],
        ),
    }
    model = PPO.load(MODEL_PATH, env=env, device="mps", custom_objects=custom_objects)

    raw_env = env.envs[0].env
    while hasattr(raw_env, 'env'):
        raw_env = raw_env.env

    original_step = raw_env.step
    def patched_step(action):
        obs, reward, terminated, truncated, info = original_step(action)
        return obs, reward, False, False, info
    raw_env.step = patched_step

    mj_model = raw_env._robot._model
    mj_data = raw_env._robot._data
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    obs = env.reset()
    
    while viewer.is_running():
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        viewer.sync()
        time.sleep(0.01)
