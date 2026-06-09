import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class MujocoClient:
    """Wraps MuJoCo model+data, mimicking the BulletClient interface."""

    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._renderer = None

    def resetSimulation(self):
        mujoco.mj_resetData(self.model, self.data)

    def setTimeStep(self, dt):
        self.model.opt.timestep = dt

    def setGravity(self, x, y, z):
        self.model.opt.gravity[:] = [x, y, z]

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def getBasePositionAndOrientation(self):
        pos = self.data.qpos[0:3].copy()
        quat_wxyz = self.data.qpos[3:7].copy()  # MuJoCo: w,x,y,z
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        )  # PyBullet: x,y,z,w
        return pos, quat_xyzw

    def getBaseVelocity(self):
        lin_vel = self.data.qvel[0:3].copy()
        ang_vel = self.data.qvel[3:6].copy()
        return lin_vel, ang_vel

    def resetBasePositionAndOrientation(self, pos, quat_xyzw):
        self.data.qpos[0:3] = pos
        # convert xyzw → wxyz for MuJoCo
        self.data.qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        mujoco.mj_forward(self.model, self.data)

    def resetBaseVelocity(self, lin_vel, ang_vel):
        self.data.qvel[0:3] = lin_vel
        self.data.qvel[3:6] = ang_vel

    def getJointStates(self, joint_ids):
        """Returns list of (pos, vel) tuples for given joint indices."""
        return [
            (self.data.qpos[7 + i], self.data.qvel[6 + i])
            for i in range(len(joint_ids))
        ]

    def resetJointState(self, joint_id, angle, velocity=0.0):
        self.data.qpos[7 + joint_id] = angle
        self.data.qvel[6 + joint_id] = velocity
        mujoco.mj_forward(self.model, self.data)

    def setJointMotorControlArray(self, joint_ids, torques):
        for i, (jid, torque) in enumerate(zip(joint_ids, torques)):
            self.data.ctrl[jid] = torque

    def setJointMotorControl2(self, joint_id, torque):
        self.data.ctrl[joint_id] = torque

    def getContactPoints(self, body_id, link_id):
        """Returns list of contacts for a given geom/link."""
        contacts = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = c.geom1
            g2 = c.geom2
            if g2 == link_id or g1 == link_id:
                contacts.append(c)
        return contacts

    def getEulerFromQuaternion(self, quat_xyzw):
        r = Rotation.from_quat(quat_xyzw)  # scipy: x,y,z,w
        return r.as_euler("xyz")

    def getQuaternionFromEuler(self, euler):
        r = Rotation.from_euler("xyz", euler)
        return r.as_quat()  # returns x,y,z,w

    def multiplyTransforms(self, posA, ornA, posB, ornB):
        rA = Rotation.from_quat(ornA)
        rB = Rotation.from_quat(ornB)
        rC = rA * rB
        posC = np.array(posA) + rA.apply(np.array(posB))
        return posC, rC.as_quat()

    def getCameraImage(self, width, height):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height, width)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def getDynamicsInfo(self, body_id):
        mass = self.model.body_mass[body_id]
        return (mass,)

    def changeDynamics(self, body_id, mass=None, friction=None):
        if mass is not None:
            self.model.body_mass[body_id] = mass
        if friction is not None:
            # apply to all geoms of this body
            for i in range(self.model.ngeom):
                if self.model.geom_bodyid[i] == body_id:
                    self.model.geom_friction[i, 0] = friction
