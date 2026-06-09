import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from vision4leg.robots.a1_mujoco import A1Mujoco, INIT_MOTOR_ANGLES, KP, KD

IMG_HEIGHT = 64
IMG_WIDTH = 64
STATE_DIM = 30
VISUAL_DIM = IMG_HEIGHT * IMG_WIDTH  # flattened depth image


class A1MujocoEnv(gym.Env):
    """Gymnasium env for A1 locomotion with depth camera in MuJoCo."""

    def __init__(self, render_mode=None, use_depth=True):
        super().__init__()
        self.render_mode = render_mode
        self.use_depth = use_depth
        self._robot = A1Mujoco()

        # depth camera renderer
        if self.use_depth:
            self._renderer = mujoco.Renderer(
                self._robot._model, height=IMG_HEIGHT, width=IMG_WIDTH
            )
            self._renderer.enable_depth_rendering()

        # action: 12 torques
        self.action_space = spaces.Box(
            low=-33.5, high=33.5, shape=(12,), dtype=np.float32
        )

        # obs: state(30) + depth(64*64) if use_depth else state(30)
        obs_dim = STATE_DIM + VISUAL_DIM if use_depth else STATE_DIM
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._prev_pos = None
        self._last_action = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._robot.Reset()
        self._prev_pos = self._robot.GetBasePosition().copy()
        self._last_action = np.zeros(12)
        return self._get_obs(), {}

    def step(self, action):
        self._robot.Step(action)
        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = not self._robot.is_safe
        truncated = False
        self._prev_pos = self._robot.GetBasePosition().copy()
        self._last_action = action.copy()
        return obs, reward, terminated, truncated, {}

    def _get_depth(self):
        """Render depth image from robot's front camera."""
        self._renderer.update_scene(self._robot._data)
        depth = self._renderer.render()
        # normalize to [0, 1] clipping at 5m
        depth = np.clip(depth, 0, 5.0) / 5.0
        return depth.flatten().astype(np.float32)

    def _get_obs(self):
        state = np.concatenate(
            [
                self._robot.GetMotorAngles(),  # 12
                self._robot.GetMotorVelocities(),  # 12
                self._robot.GetBaseRollPitchYaw(),  # 3
                self._robot.GetBaseRollPitchYawRate(),  # 3
            ]
        ).astype(np.float32)

        if self.use_depth:
            depth = self._get_depth()
            return np.concatenate([state, depth])
        return state

    def _compute_reward(self, action):
        cur_pos = self._robot.GetBasePosition()

        # forward velocity
        forward_vel = (cur_pos[0] - self._prev_pos[0]) / self._robot.time_step

        # stay upright
        rpy = self._robot.GetTrueBaseRollPitchYaw()
        orientation_penalty = 0.1 * (rpy[0] ** 2 + rpy[1] ** 2)

        # target height ~0.27
        height_penalty = 0.1 * abs(cur_pos[2] - 0.27)

        # energy penalty
        torques = self._robot.GetMotorTorques()
        energy_penalty = 0.0001 * np.sum(torques**2)

        # smoothness penalty
        smoothness_penalty = 0.001 * np.sum((action - self._last_action) ** 2)

        alive = 1.0 if self._robot.is_safe else 0.0

        return (
            2.0 * forward_vel
            + alive
            - orientation_penalty
            - height_penalty
            - energy_penalty
            - smoothness_penalty
        )

    def render(self):
        pass
