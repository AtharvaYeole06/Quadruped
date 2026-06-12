"""
Test script for the A1 LocoTransformer walking policy.
Launches MuJoCo viewer with camera tracking, live stats overlay, and episode summaries.

Usage:
    mjpython test.py                    # Load latest saved model
    mjpython test.py --checkpoint PATH  # Load specific checkpoint
    mjpython test.py --slow             # Half-speed playback
    mjpython test.py --fast             # 2x speed playback
"""
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
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

warnings.filterwarnings("ignore")

# ── Path setup ──────────────────────────────────────────────────────────────
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


# ── Architecture (must match train.py exactly) ──────────────────────────────
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
        # The environment now correctly returns STATE_DIM (49) + 4 stacked depth frames
        return self.net(obs)


def make_env():
    return A1MujocoEnv(use_depth=True)


def find_model_path(args):
    """Find the best model to load."""
    if args.checkpoint:
        return args.checkpoint

    # Priority: final model > latest checkpoint
    final_model = os.path.join(CURRENT_DIR, "a1_loco_transformer_ppo.zip")
    if os.path.exists(final_model):
        return final_model

    # Fall back to latest checkpoint
    checkpoint_dir = os.path.join(CURRENT_DIR, "checkpoints")
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "a1_loco_*_steps.zip"))
    if checkpoints:
        checkpoints.sort(key=lambda f: int(os.path.basename(f).split("_")[-2]))
        return checkpoints[-1]

    print("ERROR: No model found! Train first with: python train.py")
    sys.exit(1)


def get_raw_env(env):
    """Navigate wrapper stack to get the base A1MujocoEnv."""
    raw = env.envs[0]
    while hasattr(raw, "env"):
        raw = raw.env
    return raw


def print_banner(model_path, speed_label):
    """Print startup info."""
    name = os.path.basename(model_path)
    print(f"\n{'═' * 60}")
    print(f"  🤖 A1 LocoTransformer — Test Mode")
    print(f"{'═' * 60}")
    print(f"  Model:    {name}")
    print(f"  Speed:    {speed_label}")
    print(f"  Controls: Close viewer window to exit")
    print(f"{'═' * 60}\n")


def print_episode_stats(ep_num, ep_len, ep_reward, distance, avg_vel, max_vel, reason):
    """Print summary after each episode."""
    sim_time = ep_len * 0.02  # action_repeat=20 × 0.001s
    print(f"  Episode {ep_num:3d} │ "
          f"steps: {ep_len:4d}/1000 │ "
          f"time: {sim_time:5.1f}s │ "
          f"reward: {ep_reward:7.1f} │ "
          f"dist: {distance:5.2f}m │ "
          f"avg_vel: {avg_vel:4.2f} m/s │ "
          f"max_vel: {max_vel:4.2f} m/s │ "
          f"{reason}")


def print_summary(all_stats):
    """Print overall summary across all episodes."""
    if not all_stats:
        return

    lengths = [s["length"] for s in all_stats]
    rewards = [s["reward"] for s in all_stats]
    distances = [s["distance"] for s in all_stats]
    avg_vels = [s["avg_vel"] for s in all_stats]

    print(f"\n{'═' * 60}")
    print(f"  📊 Summary — {len(all_stats)} Episodes")
    print(f"{'═' * 60}")
    print(f"  Avg length:    {np.mean(lengths):6.0f} / 1000 steps ({np.mean(lengths)*0.02:.1f}s)")
    print(f"  Avg reward:    {np.mean(rewards):6.1f}")
    print(f"  Avg distance:  {np.mean(distances):6.2f} m")
    print(f"  Avg velocity:  {np.mean(avg_vels):6.2f} m/s")
    print(f"  Best episode:  {max(lengths)} steps ({max(lengths)*0.02:.1f}s), {max(distances):.2f}m")
    print(f"  Worst episode: {min(lengths)} steps ({min(lengths)*0.02:.1f}s)")
    print(f"{'═' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Test A1 walking policy")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--slow", action="store_true", help="Half-speed playback")
    parser.add_argument("--fast", action="store_true", help="2x speed playback")
    args = parser.parse_args()

    # ── Speed control ───────────────────────────────────────────────────────
    SIM_DT = 0.02  # Each env step = 0.02s of sim time (action_repeat=20)
    if args.slow:
        playback_dt = SIM_DT * 2  # Half speed
        speed_label = "0.5x (slow motion)"
    elif args.fast:
        playback_dt = SIM_DT / 2  # Double speed
        speed_label = "2.0x (fast)"
    else:
        playback_dt = SIM_DT  # Real-time
        speed_label = "1.0x (real-time)"

    # ── Build env (same pipeline as training) ───────────────────────────────
    env = DummyVecEnv([make_env])

    norm_path = os.path.join(CURRENT_DIR, "vec_normalize.pkl")
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
    else:
        pass

    # ── Load model ──────────────────────────────────────────────────────────
    model_path = find_model_path(args)
    model = PPO.load(
        model_path,
        env=env,
        device="mps",
    )

    # ── Access MuJoCo internals ─────────────────────────────────────────────
    raw_env = get_raw_env(env)
    mj_model = raw_env._robot._model
    mj_data = raw_env._robot._data

    # ── Launch viewer ───────────────────────────────────────────────────────
    obs = env.reset()
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    # Set up camera to track the robot
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mj_model.body("trunk").id
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -20

    # ── Episode tracking ────────────────────────────────────────────────────
    ep_num = 0
    ep_len = 0
    ep_reward = 0.0
    start_pos = raw_env._robot.GetBasePosition().copy()
    max_vel = 0.0
    vel_sum = 0.0
    all_stats = []

    try:
        while viewer.is_running():
            step_start = time.time()

            # Predict & step
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            # Track stats
            ep_len += 1
            ep_reward += reward[0]
            vel = raw_env._robot.GetBaseVelocity()[0]
            vel_sum += vel
            max_vel = max(max_vel, vel)

            # Sync viewer
            viewer.sync()

            # Episode done
            if done[0]:
                ep_num += 1
                end_pos = raw_env._robot.GetBasePosition()
                distance = end_pos[0] - start_pos[0]
                avg_vel = vel_sum / max(ep_len, 1)
                reason = "✅ survived" if ep_len >= 1000 else "💥 fell"

                # print_episode_stats(ep_num, ep_len, ep_reward, distance, avg_vel, max_vel, reason)

                all_stats.append({
                    "length": ep_len,
                    "reward": ep_reward,
                    "distance": distance,
                    "avg_vel": avg_vel,
                })

                # Reset for next episode
                obs = env.reset()
                ep_len = 0
                ep_reward = 0.0
                start_pos = raw_env._robot.GetBasePosition().copy()
                max_vel = 0.0
                vel_sum = 0.0

            # Real-time pacing
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
