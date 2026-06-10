import mujoco
import numpy as np
import os
from scipy.spatial.transform import Rotation

# Joint name map: A1 names → Go1 MJCF names
JOINT_NAME_MAP = {
    "FR_hip_joint": "FR_hip_joint",
    "FR_upper_joint": "FR_thigh_joint",
    "FR_lower_joint": "FR_calf_joint",
    "FL_hip_joint": "FL_hip_joint",
    "FL_upper_joint": "FL_thigh_joint",
    "FL_lower_joint": "FL_calf_joint",
    "RR_hip_joint": "RR_hip_joint",
    "RR_upper_joint": "RR_thigh_joint",
    "RR_lower_joint": "RR_calf_joint",
    "RL_hip_joint": "RL_hip_joint",
    "RL_upper_joint": "RL_thigh_joint",
    "RL_lower_joint": "RL_calf_joint",
}

MOTOR_NAMES = list(JOINT_NAME_MAP.keys())
NUM_MOTORS = 12
INIT_POSITION = [0, 0, 0.32]
INIT_MOTOR_ANGLES = np.array([0, 0.9, -1.8] * 4)
MAX_TORQUE = 33.5  # A1 motor torque limit (Nm)

# PD gains (used ONLY during Reset to settle into standing pose)
_RESET_KP = np.array([80.0] * 12)
_RESET_KD = np.array([0.4] * 12)

SCENE_XML = os.path.join(
    os.path.dirname(__file__), "../assets/unitree_go1/scene_torque.xml"
)


class A1Mujoco:
    """A1 quadruped simulated in MuJoCo with direct torque control."""

    def __init__(self, time_step=0.001, action_repeat=1, on_rack=False, sensors=None):
        self.time_step = time_step
        self._action_repeat = action_repeat
        self._on_rack = on_rack
        self._sensors = sensors or []
        self.num_motors = NUM_MOTORS

        self._model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self._data = mujoco.MjData(self._model)

        self._motor_id_list = self._build_motor_id_list()
        self._foot_geom_ids = self._build_foot_geom_ids()

        self._is_safe = True
        self._step_counter = 0
        self._joint_states = []
        self._base_position = np.zeros(3)
        self._base_orientation = np.array([0, 0, 0, 1.0])

        self.Reset()

    def _build_motor_id_list(self):
        ids = []
        for a1_name in MOTOR_NAMES:
            mjcf_name = JOINT_NAME_MAP[a1_name]
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, mjcf_name)
            ids.append(jid - 1)  # skip freejoint
        return ids

    def _build_foot_geom_ids(self):
        foot_names = ["FR", "FL", "RR", "RL"]
        ids = []
        for name in foot_names:
            gid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
            ids.append(gid)
        return ids

    def Reset(self, default_motor_angles=None, reset_time=0.5):
        mujoco.mj_resetData(self._model, self._data)

        self._data.qpos[0:3] = INIT_POSITION
        self._data.qpos[3:7] = [1, 0, 0, 0]  # wxyz identity

        if self._on_rack:
            self._data.qpos[2] = 1.0

        angles = (
            default_motor_angles
            if default_motor_angles is not None
            else INIT_MOTOR_ANGLES
        )
        for i, jid in enumerate(self._motor_id_list):
            self._data.qpos[7 + jid] = angles[i]

        mujoco.mj_forward(self._model, self._data)

        # Settle with PD control (only during reset, not during training)
        for _ in range(500):
            q = np.array([self._data.qpos[7 + jid] for jid in self._motor_id_list])
            qdot = np.array([self._data.qvel[6 + jid] for jid in self._motor_id_list])
            torques = _RESET_KP * (INIT_MOTOR_ANGLES - q) + _RESET_KD * (0.0 - qdot)
            torques = np.clip(torques, -MAX_TORQUE, MAX_TORQUE)
            for i, jid in enumerate(self._motor_id_list):
                self._data.ctrl[jid] = torques[i]
            mujoco.mj_step(self._model, self._data)

        self._is_safe = True
        self._step_counter = 0
        self.ReceiveObservation()

    def Step(self, action):
        for _ in range(self._action_repeat):
            self.ApplyAction(action)
            mujoco.mj_step(self._model, self._data)
            self._step_counter += 1
        self.ReceiveObservation()

    def ApplyAction(self, torques):
        """Direct torque control — action IS the torque (already scaled by env)."""
        clipped = np.clip(torques, -MAX_TORQUE, MAX_TORQUE)
        for i, jid in enumerate(self._motor_id_list):
            self._data.ctrl[jid] = clipped[i]

    def ReceiveObservation(self):
        self._joint_states = [
            (self._data.qpos[7 + jid], self._data.qvel[6 + jid])
            for jid in self._motor_id_list
        ]
        self._base_position = self._data.qpos[0:3].copy()
        quat_wxyz = self._data.qpos[3:7]
        self._base_orientation = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        )

    def GetBasePosition(self):
        return self._base_position.copy()

    def GetBaseVelocity(self):
        return self._data.qvel[0:3].copy()

    def GetTrueBaseOrientation(self):
        return self._base_orientation.copy()

    def GetBaseOrientation(self):
        return self._base_orientation.copy()

    def GetTrueBaseRollPitchYaw(self):
        r = Rotation.from_quat(self._base_orientation)
        return r.as_euler("xyz")

    def GetBaseRollPitchYaw(self):
        return self.GetTrueBaseRollPitchYaw()

    def GetTrueBaseRollPitchYawRate(self):
        return self._data.qvel[3:6].copy()

    def GetBaseRollPitchYawRate(self):
        return self.GetTrueBaseRollPitchYawRate()

    def GetTrueMotorAngles(self):
        return np.array([s[0] for s in self._joint_states])

    def GetMotorAngles(self):
        return self.GetTrueMotorAngles()

    def GetTrueMotorVelocities(self):
        return np.array([s[1] for s in self._joint_states])

    def GetMotorVelocities(self):
        return self.GetTrueMotorVelocities()

    def GetTrueMotorTorques(self):
        return self._data.actuator_force[self._motor_id_list].copy()

    def GetMotorTorques(self):
        return self.GetTrueMotorTorques()

    def GetFootContacts(self):
        contacts = [False] * 4
        for i in range(self._data.ncon):
            c = self._data.contact[i]
            for foot_idx, gid in enumerate(self._foot_geom_ids):
                if c.geom1 == gid or c.geom2 == gid:
                    contacts[foot_idx] = True
        return contacts

    def GetTimeSinceReset(self):
        return self._step_counter * self.time_step

    def Terminate(self):
        pass

    def GetAllSensors(self):
        return self._sensors

    @property
    def is_safe(self):
        rpy = self.GetTrueBaseRollPitchYaw()
        base_z = self._base_position[2]
        if abs(rpy[0]) > 1.0 or abs(rpy[1]) > 1.0 or base_z < 0.2:
            self._is_safe = False
        return self._is_safe
