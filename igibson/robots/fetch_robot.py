import gym
import numpy as np
import pybullet as p

from igibson.external.pybullet_tools.utils import joints_from_names, set_joint_positions
from igibson.robots.robot_locomotor import LocomotorRobot

import math

def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    _NEXT_AXIS = [1, 2, 0, 1]
    _AXES2TUPLE = {
        'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
        'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
        'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
        'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
        'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
        'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
        'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
        'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
    _TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis + 1
    j = _NEXT_AXIS[i+parity-1] + 1
    k = _NEXT_AXIS[i-parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    if repetition:
        q[0] = cj*(cc - ss)
        q[i] = cj*(cs + sc)
        q[j] = sj*(cc + ss)
        q[k] = sj*(cs - sc)
    else:
        q[0] = cj*cc + sj*ss
        q[i] = cj*sc - sj*cs
        q[j] = cj*ss + sj*cc
        q[k] = cj*cs - sj*sc
    if parity:
        q[j] *= -1.0

    return q


def create_primitive_shape(mass, shape, dim, color=(0.6, 0, 0, 1), collidable=False, init_xyz=(0, 0, 0.5),
                           init_quat=(0, 0, 0, 1)):
    # shape: p.GEOM_SPHERE or p.GEOM_BOX or p.GEOM_CYLINDER
    # dim: halfExtents (vec3) for box, (radius, length)vec2 for cylinder, (radius) for sphere
    # init_xyz vec3 being initial obj location, init_quat being initial obj orientation
    visual_shape_id = None
    collision_shape_id = -1
    if shape == p.GEOM_BOX:
        visual_shape_id = p.createVisualShape(shapeType=shape, halfExtents=dim, rgbaColor=color)
        if collidable:
            collision_shape_id = p.createCollisionShape(shapeType=shape, halfExtents=dim)
    elif shape == p.GEOM_CYLINDER:
        visual_shape_id = p.createVisualShape(shape, dim[0], [1, 1, 1], dim[1], rgbaColor=color)
        if collidable:
            collision_shape_id = p.createCollisionShape(shape, dim[0], [1, 1, 1], dim[1])
    elif shape == p.GEOM_SPHERE:
        visual_shape_id = p.createVisualShape(shape, radius=dim[0], rgbaColor=color)
        if collidable:
            collision_shape_id = p.createCollisionShape(shape, radius=dim[0])

    sid = p.createMultiBody(baseMass=mass, baseInertialFramePosition=[0, 0, 0],
                            baseCollisionShapeIndex=collision_shape_id,
                            baseVisualShapeIndex=visual_shape_id,
                            basePosition=init_xyz, baseOrientation=init_quat)
    return sid


class Fetch(LocomotorRobot):
    """
    Fetch Robot
    Reference: https://fetchrobotics.com/robotics-platforms/fetch-mobile-manipulator/
    Uses joint velocity control
    """

    def __init__(self, config):
        self.config = config
        self.wheel_velocity = config.get('wheel_velocity', 1.0)
        self.torso_lift_velocity = config.get('torso_lift_velocity', 1.0)
        self.arm_velocity = config.get('arm_velocity', 1.0)
        self.wheel_dim = 2
        self.torso_lift_dim = 1
        self.arm_dim = 7
        LocomotorRobot.__init__(self,
                                "fetch/fetch.urdf",
                                action_dim=self.wheel_dim + self.torso_lift_dim + self.arm_dim,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control="velocity",
                                self_collision=True)

        self.hand_lim_min = np.array([0.5, -0.4, 1.0])
        self.hand_lim_max = np.array([0.8, 0.4, 1.5])

        self.gripper_open = 1.0
        self.gripper_close = -0.3
        self.gripper = 0.0

        self.hand_id = 13
        self.hold_cons = None
        self.tip_cons = -100
        self.pelvis_id = 4

        self.holdobject = False
        self.holddrawer = False
        self.holddrawer_id = -1
        self.kneedown = False

        self.tip = create_primitive_shape(0.00001, p.GEOM_SPHERE, [0.02, 0.0, 0.0], [0.6, 0, 0, 1], False)


    def set_up_continuous_action_space(self):
        """
        Set up continuous action space
        """
        self.action_high = np.array([self.wheel_velocity] * self.wheel_dim +
                                    [self.torso_lift_velocity] * self.torso_lift_dim +
                                    [self.arm_velocity] * self.arm_dim)
        self.action_low = -self.action_high
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)

    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        assert False, "Fetch does not support discrete actions"

    def base_reset(self, pos, ori):
        self.base_cons = p.createConstraint(self.robot_ids[0], -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], pos, [0, 0, 0, 1])
        self.current_base_pos = np.array(pos)
        self.current_base_ori = np.array(ori)
        p.changeConstraint(self.base_cons, self.current_base_pos, self.current_base_ori, maxForce=30000.0)

    def base_change(self, pos, ori):
        self.current_base_pos = np.array(pos)
        self.current_base_ori = np.array(ori)
        p.changeConstraint(self.base_cons, self.current_base_pos, self.current_base_ori, maxForce=30000.0)

    def knee_down(self):
        self.kneedown = True

        self.hand_lim_min = np.array([0.5, -0.6, 0.5])
        self.hand_lim_max = np.array([1.2, 0.6, 1.4])


    def pose_reset(self, hand_pos, hand_ori, force):
        self.current_hand_pos = np.array(hand_pos)
        self.current_hand_ori = np.array(hand_ori)

        self.current_hand_pos_wf, self.current_hand_ori_wf = p.multiplyTransforms(self.current_base_pos,
                                                                                  self.current_base_ori,
                                                                                  self.current_hand_pos,
                                                                                  quaternion_from_euler(
                                                                                      self.current_hand_ori[0],
                                                                                      self.current_hand_ori[1],
                                                                                      self.current_hand_ori[2]))

        solutions = list(p.calculateInverseKinematics(self.robot_ids[0], self.hand_id, self.current_hand_pos_wf,
                                                      self.current_hand_ori_wf))

        del_action = [0.0 for i in range(p.getNumJoints(self.robot_ids[0]))]

        del_action[12] = solutions[0]
        del_action[13] = solutions[1]
        del_action[14] = solutions[2]
        del_action[15] = solutions[3]
        del_action[16] = solutions[4]
        del_action[17] = solutions[5]
        del_action[18] = solutions[6]

        del_action[20] = self.gripper
        del_action[22] = self.gripper

        forces = [force for i in range(p.getNumJoints(self.robot_ids[0]))]
        p.setJointMotorControlArray(self.robot_ids[0], [i for i in range(p.getNumJoints(self.robot_ids[0]))], p.POSITION_CONTROL, del_action, forces = forces)

    def clear(self):
        if self.hold_cons:
            p.removeConstraint(self.hold_cons)
            self.hold_cons = None
            self.holdobject = False
            self.holddrawer = False
            self.holddrawer_id = -1

    def give_target(self, obj):
        self.target = obj

    def hold_target(self):
        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.target)
        gripper_pos = p.getLinkState(self.robot_ids[0], self.hand_id)[0]
        gripper_orn = p.getLinkState(self.robot_ids[0], self.hand_id)[1]
        grasp_pose = p.multiplyTransforms(*p.invertTransform(gripper_pos, gripper_orn), obj_pos, obj_orn)

        self.hold_cons = p.createConstraint(
            parentBodyUniqueId=self.robot_ids[0],
            parentLinkIndex=self.hand_id,
            childBodyUniqueId=self.target,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=[0.18, 0, 0], # exp 3
            # parentFramePosition=[0.16, 0, 0], # exp 2
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=grasp_pose[1],
            childFrameOrientation=(0, 0, 0, 1),
        )
        p.changeConstraint(self.hold_cons, maxForce=100.0)

        self.holdobject = True

    def hold_drawer(self, cab_id, dra_id):
        obj_pos = p.getLinkState(cab_id, dra_id)[0]
        obj_orn = p.getLinkState(cab_id, dra_id)[1]
        gripper_pos = p.getLinkState(self.robot_ids[0], self.hand_id)[0]
        gripper_orn = p.getLinkState(self.robot_ids[0], self.hand_id)[1]
        grasp_pose = p.multiplyTransforms(*p.invertTransform(gripper_pos, gripper_orn), obj_pos, obj_orn)

        self.hold_cons = p.createConstraint(
            parentBodyUniqueId=self.robot_ids[0],
            parentLinkIndex=self.hand_id,
            childBodyUniqueId=cab_id,
            childLinkIndex=dra_id,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=grasp_pose[0],
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=grasp_pose[1],
            childFrameOrientation=(0, 0, 0, 1),
        )
        p.changeConstraint(self.hold_cons, maxForce=10000000.0)

        self.holddrawer = True
        self.holddrawer_id = dra_id

    def close_to_target(self):
        pos1 = np.array(p.getBasePositionAndOrientation(self.tip)[0])
        pos2 = np.array(p.getBasePositionAndOrientation(self.target)[0])

        if np.linalg.norm(pos1 - pos2) < 0.05:
            return True
        else:
            return False

    def give_initial_ee_pose(self, hand_pos, hand_ori):
        self.current_hand_pos = np.array(hand_pos)
        self.current_hand_ori = np.array(hand_ori)

    def hand_delta_action_valid(self, current_pos, delta_act):
        if current_pos[0] + delta_act[0] < self.hand_lim_min[0]:
            delta_act[0] = self.hand_lim_min[0] - current_pos[0]
        if current_pos[0] + delta_act[0] > self.hand_lim_max[0]:
            delta_act[0] = self.hand_lim_max[0] - current_pos[0]
        if current_pos[1] + delta_act[1] < self.hand_lim_min[1]:
            delta_act[1] = self.hand_lim_min[1] - current_pos[1]
        if current_pos[1] + delta_act[1] > self.hand_lim_max[1]:
            delta_act[1] = self.hand_lim_max[1] - current_pos[1]
        if current_pos[2] + delta_act[2] < self.hand_lim_min[2]:
            delta_act[2] = self.hand_lim_min[2] - current_pos[2]
        if current_pos[2] + delta_act[2] > self.hand_lim_max[2]:
            delta_act[2] = self.hand_lim_max[2] - current_pos[2]

        return delta_act

    def hand_delta_action_valid_ori(self, current_ori, delta_ori):
        target = np.clip(current_ori + delta_ori, -1.0, 1.0)
        delta_ori = target - current_ori

        return delta_ori

    def robot_specific_reset(self):
        """
        Fetch robot specific reset.
        Reset the torso lift joint and tuck the arm towards the body
        """
        super(Fetch, self).robot_specific_reset()

        # roll the arm to its body
        robot_id = self.robot_ids[0]
        arm_joints = joints_from_names(robot_id,
                                       [
                                           'torso_lift_joint',
                                           'shoulder_pan_joint',
                                           'shoulder_lift_joint',
                                           'upperarm_roll_joint',
                                           'elbow_flex_joint',
                                           'forearm_roll_joint',
                                           'wrist_flex_joint',
                                           'wrist_roll_joint'
                                       ])

        rest_position = (0.3, -1.4, 1.51, 0.81, 2.2, 2.9, -1.2, 0.0)

        set_joint_positions(robot_id, arm_joints, rest_position)

    def get_end_effector_position(self):
        """
        Get end-effector position
        """
        return self.parts['gripper_link'].get_position()

    def end_effector_part_index(self):
        """
        Get end-effector link id
        """
        return self.parts['gripper_link'].body_part_index

    def load(self):
        """
        Load the robot into pybullet. Filter out unnecessary self collision
        due to modeling imperfection in the URDF
        """
        ids = super(Fetch, self).load()

        self.set_position([-1.0, 0.0, 0.0])
        robot_id = self.robot_ids[0]

        disable_collision_names = [
            ['torso_lift_joint', 'shoulder_lift_joint'],
            ['torso_lift_joint', 'torso_fixed_joint'],
            ['caster_wheel_joint', 'estop_joint'],
            ['caster_wheel_joint', 'laser_joint'],
            ['caster_wheel_joint', 'torso_fixed_joint'],
            ['caster_wheel_joint', 'l_wheel_joint'],
            ['caster_wheel_joint', 'r_wheel_joint'],
        ]
        for names in disable_collision_names:
            link_a, link_b = joints_from_names(robot_id, names)
            p.setCollisionFilterPair(robot_id, robot_id, link_a, link_b, 0)

        if self.tip_cons == -100:
            self.tip_cons = p.createConstraint(
                parentBodyUniqueId=self.robot_ids[0],
                parentLinkIndex=self.hand_id,
                childBodyUniqueId=self.tip,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=(0, 0, 0),
                parentFramePosition=[0.18, 0, 0],
                childFramePosition=[0, 0, 0],
                parentFrameOrientation=(0, 0, 0, 1),
                childFrameOrientation=(0, 0, 0, 1),
            )
            p.changeConstraint(self.tip_cons, maxForce=1.0)

        return ids

    def apply_action(self, action):

        delta_pos = action[3:6]
        delta_pos = self.hand_delta_action_valid(self.current_hand_pos, delta_pos)
        if self.holddrawer:
            delta_pos[0] = 0.0
            delta_pos[2] = 0.0

        self.current_hand_pos += delta_pos
        if not(self.holddrawer):
            self.current_hand_ori = np.array(action[6:9])

        if self.kneedown and True:
            if self.hold_cons:
                self.gripper = self.gripper_close
            else:
                self.gripper = self.gripper_open
        else:
            if action[-1] > 0.0:
                self.gripper = self.gripper_close
            else:
                self.gripper = self.gripper_open


        self.current_hand_pos_wf, self.current_hand_ori_wf = p.multiplyTransforms(self.current_base_pos, self.current_base_ori, self.current_hand_pos, quaternion_from_euler(self.current_hand_ori[0], self.current_hand_ori[1], self.current_hand_ori[2]))
        solutions = list(p.calculateInverseKinematics(self.robot_ids[0], self.hand_id, self.current_hand_pos_wf, self.current_hand_ori_wf))

        del_action = [0.0 for i in range(p.getNumJoints(self.robot_ids[0]))]

        del_action[6] = solutions[0]
        del_action[7] = solutions[1]
        del_action[8] = solutions[2]
        del_action[9] = solutions[3]
        del_action[10] = solutions[4]
        del_action[11] = solutions[5]
        del_action[12] = solutions[6]

        del_action[14] = self.gripper
        del_action[16] = self.gripper

        forces = [500.0 for i in range(p.getNumJoints(self.robot_ids[0]))]
        forces[14] = 10.0
        forces[16] = 10.0

        p.setJointMotorControlArray(self.robot_ids[0], [i for i in range(p.getNumJoints(self.robot_ids[0]))], p.POSITION_CONTROL, del_action, forces = forces)
        p.changeConstraint(self.base_cons, self.current_base_pos, maxForce = 3000.0)

        return
