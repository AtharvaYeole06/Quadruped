import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from vision4leg.robots.a1_mujoco import A1Mujoco, MAX_TORQUE

IMG_HEIGHT = 64
IMG_WIDTH = 64
STATE_DIM = 33  # 12 angles + 12 velocities + 3 RPY + 3 RPY_rate + 3 base_velocity
VISUAL_DIM = IMG_HEIGHT * IMG_WIDTH

# ── Config ──────────────────────────────────────────────────────────────────
TARGET_VELOCITY = 0.5   # m/s
TORQUE_SCALE = 10.0     # Policy outputs [-1,1] → [-10, 10] Nm
MAX_EPISODE_STEPS = 1000


class A1MujocoEnv(gym.Env):
    """Gymnasium env for A1 locomotion with depth camera + torque control."""

    def __init__(self, render_mode=None, use_depth=True):
        super().__init__()
        self.render_mode = render_mode
        self.use_depth = use_depth
        self._robot = A1Mujoco(action_repeat=20)  # 50 Hz control

        # Depth camera
        if self.use_depth:
            self._renderer = mujoco.Renderer(
                self._robot._model, height=IMG_HEIGHT, width=IMG_WIDTH
            )
            self._renderer.enable_depth_rendering()

        # Action: 12 normalized torques [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        # Obs: state(33) + depth(64×64)
        obs_dim = STATE_DIM + VISUAL_DIM if use_depth else STATE_DIM
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._last_action = None
        self._step_count = 0
        self._last_contacts = np.zeros(4, dtype=bool)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._robot.Reset()
        self._last_action = np.zeros(12)
        self._step_count = 0
        self._last_contacts = np.zeros(4, dtype=bool)
        return self._get_obs(), {}

    def step(self, action):
        torques = action * TORQUE_SCALE
        self._robot.Step(torques)

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = not self._robot.is_safe

        self._step_count += 1
        truncated = self._step_count >= MAX_EPISODE_STEPS

        self._last_action = action.copy()
        return obs, reward, terminated, truncated, {}

    def _get_depth(self):
        self._renderer.update_scene(self._robot._data)
        depth = self._renderer.render()
        depth = np.clip(depth, 0, 5.0) / 5.0
        return depth.flatten().astype(np.float32)

    def _get_obs(self):
        state = np.concatenate([
            self._robot.GetMotorAngles(),          # 12
            self._robot.GetMotorVelocities(),      # 12
            self._robot.GetBaseRollPitchYaw(),     # 3
            self._robot.GetBaseRollPitchYawRate(),  # 3
            self._robot.GetBaseVelocity(),          # 3 — needed for velocity tracking!
        ]).astype(np.float32)

        if self.use_depth:
            return np.concatenate([state, self._get_depth()])
        return state

    def _compute_reward(self, action):
        base_vel = self._robot.GetBaseVelocity()
        rpy = self._robot.GetBaseRollPitchYaw()
        rpy_rate = self._robot.GetBaseRollPitchYawRate()
        torques = self._robot.GetMotorTorques()

        # Task reward: encourage forward velocity tracking
        forward_vel = base_vel[0]
        vel_error = forward_vel - TARGET_VELOCITY
        task_reward = np.exp(-(vel_error ** 2) / 0.1)

        # Stability penalties
        orientation_err = rpy[0]**2 + rpy[1]**2              # Keep base flat
        z_vel_err = base_vel[2]**2                           # Prevent bouncing
        ang_vel_err = rpy_rate[0]**2 + rpy_rate[1]**2        # Prevent wobbling
        
        # Energy and smoothness penalties
        torque_err = np.sum(torques**2)                      # Minimize energy
        action_rate_err = np.sum((action - self._last_action) ** 2) # Smooth actions

        penalty = (
            1.0 * orientation_err +
            2.0 * z_vel_err +
            0.05 * ang_vel_err +
            0.0002 * torque_err +
            0.005 * action_rate_err
        )

        # Bounded multiplicative reward with small survival bonus
        reward = task_reward * np.exp(-penalty) + 0.2 
        return float(reward)

    def render(self):
        pass
