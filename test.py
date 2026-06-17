import os
import sys
import glob
import time
import argparse
import warnings

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from vision4leg.envs.a1_mujoco_env import (
    IMG_HEIGHT,
    IMG_WIDTH,
    STATE_DIM,
    A1MujocoEnv,
)
from torchrl.networks.base import LocoTransformerEncoder
from torchrl.networks.nets import LocoTransformer

class LocoTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=128)

        self.encoder = LocoTransformerEncoder(
            in_channels=4,
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
        return self.net(obs)

def make_env(control_mode="torque"):
    return A1MujocoEnv(use_depth=True, control_mode=control_mode)

def find_model_path(args):
    if args.checkpoint:
        return args.checkpoint

    final_model = os.path.join(CURRENT_DIR, "a1_loco_transformer_ppo.zip")
    if os.path.exists(final_model):
        return final_model

    checkpoint_dir = os.path.join(CURRENT_DIR, "checkpoints")
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "a1_loco_*_steps.zip"))
    if checkpoints:
        checkpoints.sort(key=lambda f: int(os.path.basename(f).split("_")[-2]))
        return checkpoints[-1]

    sys.exit(1)

def get_raw_env(env):
    raw = env.envs[0]
    while hasattr(raw, "env"):
        raw = raw.env
    return raw

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--slow", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--mode", type=str, default="position", choices=["position", "torque"], 
                        help="Control mode the checkpoint was trained with")
    args = parser.parse_args()

    SIM_DT = 0.02
    if args.slow:
        playback_dt = SIM_DT * 2
    elif args.fast:
        playback_dt = SIM_DT / 2
    else:
        playback_dt = SIM_DT

    # Pass the mode to make_env
    env = DummyVecEnv([lambda: make_env(control_mode=args.mode)])

    norm_path = os.path.join(CURRENT_DIR, "vec_normalize.pkl")
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False

    model_path = find_model_path(args)
    model = PPO.load(
        model_path,
        env=env,
        device="mps",
    )

    raw_env = get_raw_env(env)
    mj_model = raw_env._robot._model
    mj_data = raw_env._robot._data

    original_step = raw_env.step
    def patched_step(action):
        obs, reward, terminated, truncated, info = original_step(action)
        return obs, reward, False, False, info
    raw_env.step = patched_step

    obs = env.reset()
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mj_model.body("trunk").id
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -20

    try:
        while viewer.is_running():
            step_start = time.time()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            viewer.sync()

            elapsed = time.time() - step_start
            sleep_time = playback_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()

if __name__ == "__main__":
    main()
