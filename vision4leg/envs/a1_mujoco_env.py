import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from collections import deque
from vision4leg.robots.a1_mujoco import A1Mujoco, MAX_TORQUE, INIT_MOTOR_ANGLES

IMG_HEIGHT = 64
IMG_WIDTH = 64
STATE_DIM = 50  # 12 pos + 12 vel + 3 RPY + 3 RPY_rate + 3 base_vel + 4 foot_contact + 12 prev_action + 1 target_vel
VISUAL_DIM = 4 * IMG_HEIGHT * IMG_WIDTH

TORQUE_SCALE = MAX_TORQUE  # For torque mode: [-1,1] → [-33.5, 33.5] Nm
ACTION_SCALE = 0.25         # For position mode: [-1,1] → ±0.25 rad offset
MAX_EPISODE_STEPS = 1000


class A1MujocoEnv(gym.Env):
    """Gymnasium env for A1 locomotion with depth camera."""

    def __init__(self, render_mode=None, use_depth=True, control_mode="torque"):
        super().__init__()
        self.render_mode = render_mode
        self.use_depth = use_depth
        self.control_mode = control_mode
        self._robot = A1Mujoco(action_repeat=10, control_mode=control_mode)
        self.target_velocity = 0.0

        # Depth camera
        if self.use_depth:
            self._renderer = mujoco.Renderer(
                self._robot._model, height=IMG_HEIGHT, width=IMG_WIDTH
            )
            self._renderer.enable_depth_rendering()

        # Action: 12 normalized values [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        # Obs: state(50) + depth(4×64×64)
        obs_dim = STATE_DIM + VISUAL_DIM if use_depth else STATE_DIM
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._last_action = np.zeros(12)
        self._step_count = 0
        self._last_contacts = np.zeros(4, dtype=bool)
        if self.use_depth:
            self.frame_stack = deque(maxlen=4)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Check if user manually passed a target velocity, otherwise randomize
        if options and "target_vel" in options:
            self.target_velocity = options["target_vel"]
        else:
            self.target_velocity = np.random.uniform(0.0, 1.0)

        self._robot.Reset()
        self._last_action = np.zeros(12)
        self._step_count = 0
        self._last_contacts = np.zeros(4, dtype=bool)
        if self.use_depth:
            self.frame_stack.clear()
            depth = self._get_depth()
            for _ in range(4):
                self.frame_stack.append(depth)
        return self._get_obs(), {}

    def step(self, action):
        if self.control_mode == "torque":
            motor_command = action * TORQUE_SCALE
        else:
            motor_command = INIT_MOTOR_ANGLES + action * ACTION_SCALE

        self._robot.Step(motor_command)

        if self.use_depth:
            self.frame_stack.append(self._get_depth())

        obs = self._get_obs()
        reward = self._compute_reward(action)
        
        # Check if the robot rolled onto its back (turtle mode exploit)
        rpy = self._robot.GetBaseRollPitchYaw()
        is_flipped = abs(rpy[0]) > 1.0 or abs(rpy[1]) > 1.0
        
        base_height = self._robot._data.qpos[2]
        terminated = not self._robot.is_safe or base_height < 0.20 or is_flipped

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
        state = np.concatenate(
            [
                self._robot.GetMotorAngles(),  # 12
                self._robot.GetMotorVelocities(),  # 12
                self._robot.GetBaseRollPitchYaw(),  # 3
                self._robot.GetBaseRollPitchYawRate(),  # 3
                self._robot.GetBaseVelocity(),  # 3
                self._robot.GetFootContacts(),  # 4
                self._last_action,  # 12
                np.array([self.target_velocity]),  # 1
            ]
        ).astype(np.float32)

        if self.use_depth:
            stacked_depth = np.concatenate(self.frame_stack)
            return np.concatenate([state, stacked_depth])
        return state

    def _compute_reward(self, action):
        base_vel = self._robot.GetBaseVelocity()
        rpy = self._robot.GetBaseRollPitchYaw()
        rpy_rate = self._robot.GetBaseRollPitchYawRate()
        torques = self._robot.GetMotorTorques()
        joint_vel = self._robot.GetMotorVelocities()
        base_height = self._robot._data.qpos[2]

        lin_vel_err = np.square(base_vel[0] - self.target_velocity)
        reward_lin_vel = np.exp(-lin_vel_err / 0.1) * 2.0
        survival_bonus = 0.5

        penalty_lat_vel = np.square(base_vel[1]) * 1.5
        penalty_yaw_rate = np.square(rpy_rate[2]) * 0.5
        penalty_z_vel = np.square(base_vel[2]) * 2.0
        penalty_wobble = (np.square(rpy_rate[0]) + np.square(rpy_rate[1])) * 0.5
        
        height_err = np.square(base_height - 0.3)
        penalty_height = height_err * 10.0
        penalty_posture = (np.square(rpy[0]) + np.square(rpy[1])) * 3.0

        torque_penalty = np.sum(np.square(torques)) * 0.00005
        action_rate_penalty = np.sum(np.square(action - self._last_action)) * 0.05
        joint_vel_penalty = np.sum(np.square(joint_vel)) * 0.0005

        reward = (
            reward_lin_vel
            + survival_bonus
            - penalty_lat_vel
            - penalty_yaw_rate
            - penalty_z_vel
            - penalty_wobble
            - penalty_height
            - penalty_posture
            - torque_penalty
            - action_rate_penalty
            - joint_vel_penalty
        )
        return float(reward)

    def render(self):
        pass
