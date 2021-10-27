import gym
import numpy as np
import pybullet as p
from igibson.robots.robot_locomotor import LocomotorRobot
from igibson.external.pybullet_tools.utils import joints_from_names
import time
import random
import copy
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



class Humanoid_hri(LocomotorRobot):
    """
    Turtlebot robot
    Reference: http://wiki.ros.org/Robots/TurtleBot
    Uses joint velocity control
    """

    def __init__(self, config):
        self.config = config
        self.velocity = config.get("velocity", 1.0)
        self.knee = config.get("knee", False)

        if self.knee:
            file = "humanoid_hri/humanoid_hri_knee.urdf"
        else:
            file = "humanoid_hri/humanoid_hri.urdf"

        LocomotorRobot.__init__(self,
                                file,
                                action_dim=2,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control="velocity")

        self.hold_cons = None
        self.first = False
        self.tip_cons = -100
        self.gripper_open = 1.0
        self.gripper_close = -0.01
        self.hand_id = 31
        self.pelvis_id = 4

        self.kneedown = False

        self.holdobject = False
        self.holddrawer = False
        self.holddrawer_id = -1

        self.hand_lim_min = [0.2, -0.4, 0.14]
        self.hand_lim_max = [0.5, 0.2, 0.4]

        self.obj_init_pos = [0.45, -0.2, 0.3]

        # self.hand_lim_min = [0.2, -0.55, 0.1]
        # self.hand_lim_max = [0.5, 0.1, 0.4]

        self.tip = create_primitive_shape(0.00001, p.GEOM_SPHERE, [0.02, 0.0, 0.0], [0.6, 0, 0, 1], False)

        self.gripper = self.gripper_open

    def load(self):
        """
        Load the robot into pybullet. Filter out unnecessary self collision
        due to modeling imperfection in the URDF
        """
        ids = super(Humanoid_hri, self).load()

        self.set_position([1.0, 0.0, 0.0])
        robot_id = self.robot_ids[0]

        disable_collision_names = [
            ['abdomen_z', 'abdomen_y'],
            ['abdomen_y', 'jointfix_7_3'],
            ['jointfix_7_3', 'abdomen_x'],
            ['abdomen_x', 'jointfix_6_5'],
            ['jointfix_6_5', 'right_hip_x'],
            ['right_hip_x', 'right_hip_z'],
            ['right_hip_z', 'right_hip_y'],
            ['right_hip_y', 'jointfix_2_9'],
            ['jointfix_2_9', 'right_knee'],
            ['right_knee', 'jointfix_1_11'],
            ['jointfix_1_11', 'right_ankle_y'],
            ['right_ankle_y', 'right_ankle_x'],
            ['right_ankle_x', 'jointfix_0_14'],
            ['jointfix_0_14', 'left_hip_x'],
            ['left_hip_x', 'left_hip_z'],
            ['left_hip_z', 'left_hip_y'],
            ['left_hip_y', 'jointfix_5_18'],
            ['jointfix_5_18', 'left_knee'],
            ['left_knee', 'jointfix_4_20'],
            ['jointfix_4_20', 'left_ankle_y'],
            ['left_ankle_y', 'left_ankle_x'],
            ['left_ankle_x', 'jointfix_3_23'],
        ]
        for names in disable_collision_names:
            link_a, link_b = joints_from_names(robot_id, names)
            p.setCollisionFilterPair(robot_id, robot_id, link_a, link_b, 0)

        return ids

    def set_up_continuous_action_space(self):
        """
        Set up continuous action space
        """
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = np.full(
            shape=self.action_dim, fill_value=self.velocity)
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        self.action_list = [[self.velocity, self.velocity], [-self.velocity, -self.velocity],
                            [self.velocity * 0.5, -self.velocity * 0.5],
                            [-self.velocity * 0.5, self.velocity * 0.5], [0, 0]]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  # forward
            (ord('s'),): 1,  # backward
            (ord('d'),): 2,  # turn right
            (ord('a'),): 3,  # turn left
            (): 4  # stay still
        }

    def robot_specific_reset(self):
        """
        Robot specific reset. Apply zero velocity for all joints.
        """
        for j in self.ordered_joints:
            j.reset_joint_state(0.0, 0.0)

        if self.tip_cons == -100:
            self.tip_cons = p.createConstraint(
                parentBodyUniqueId=self.robot_ids[0],
                parentLinkIndex=self.hand_id,
                childBodyUniqueId=self.tip,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=(0, 0, 0),
                parentFramePosition=[0.11, 0, 0],
                childFramePosition=[0, 0, 0],
                parentFrameOrientation=(0, 0, 0, 1),
                childFrameOrientation=(0, 0, 0, 1),
            )
            p.changeConstraint(self.tip_cons, maxForce=1.0)

    def base_reset(self, pos, ori):
        self.base_cons = p.createConstraint(self.robot_ids[0], self.pelvis_id, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], pos, [0, 0, 0, 1]) # pelvis
        self.current_base_pos = np.array(pos)
        self.current_base_ori = np.array(ori)
        p.changeConstraint(self.base_cons, self.current_base_pos, self.current_base_ori, maxForce=300000.0)

    def base_change(self, pos, ori):
        self.current_base_pos = np.array(pos)
        self.current_base_ori = np.array(ori)
        p.changeConstraint(self.base_cons, self.current_base_pos, self.current_base_ori, maxForce=300000.0)

    def knee_down(self):
        self.kneedown = True
        self.obj_init_pos = [0.7, -0.2, 0.38]

        self.hand_lim_min = [0.1, -0.3, 0.15]
        self.hand_lim_max = [0.6, 0.2, 0.7]

    def pose_reset(self, hand_pos, hand_ori, force):
        self.current_hand_pos = np.array(hand_pos)
        self.current_hand_ori = np.array(hand_ori)

        goal_joints = [0.0 for i in range(p.getNumJoints(self.robot_ids[0]))]
        if self.knee:
            goal_joints[9] = 0.0
            goal_joints[18] = 0.0
            if self.kneedown:
                goal_joints[9] = -1.57
                goal_joints[18] = -1.77
                # goal_joints[7] = -1.07
            goal_joints[-2] = -1.8
            goal_joints[-4] = 0.5
            goal_joints[-5] = -0.5
            goal_joints[32] = self.gripper_open
            goal_joints[34] = self.gripper_open

        forces = [force for i in range(p.getNumJoints(self.robot_ids[0]))]
        p.setJointMotorControlArray(self.robot_ids[0], [i for i in range(p.getNumJoints(self.robot_ids[0]))], p.POSITION_CONTROL, goal_joints, forces = forces)

        p.changeDynamics(self.robot_ids[0], 33, mass = 0.1)
        p.changeDynamics(self.robot_ids[0], 35, mass = 0.1)


    def clear(self):
        if self.hold_cons:
            p.removeConstraint(self.hold_cons)
            self.hold_cons = None
            self.holdobject = False
            self.holddrawer = False
            self.holddrawer_id = -1

    def give_target(self, obj):

        self.target = obj

    def give_initial_ee_pose(self, hand_pos, hand_ori):
        self.current_hand_pos = np.array(hand_pos)
        self.current_hand_ori = np.array(hand_ori)


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
            parentFramePosition=[0.12, 0, 0],
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

    def get_a_random_target_pos(self):
        pos_wf, ori_wf = p.multiplyTransforms(self.current_base_pos,
                                              self.current_base_ori,
                                              self.obj_init_pos,
                                              [0, 0, 0, 1])
        if self.kneedown:
            return pos_wf, ori_wf
        else:
            return pos_wf

    def get_delta_rotation(self, target):
        delta = target - self.current_hand_ori
        delta = np.clip(delta, -0.03, 0.03)

        return delta


    def apply_action(self, action):
        delta_base_action = np.array(action[:3])
        delta_hand_action_pos = np.array(action[3:6])
        # delta_hand_action_ori = np.array(action[6:-1])

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

        # print(self.current_hand_pos, self.current_hand_ori)
        # time.sleep(1.0)

        delta_hand_action_pos = self.hand_delta_action_valid(self.current_hand_pos, delta_hand_action_pos)
        if self.holddrawer:
            delta_hand_action_pos[1] = 0.0
            delta_hand_action_pos[2] = 0.0
        self.current_hand_pos += delta_hand_action_pos
        # delta_hand_action_ori = self.hand_delta_action_valid_ori(self.current_hand_ori, delta_hand_action_ori)
        # self.current_hand_ori += delta_hand_action_ori
        if not(self.holddrawer):
            self.current_hand_ori = np.array(action[6:-1])
        # print(self.current_hand_ori)

        self.current_hand_pos_wf, self.current_hand_ori_wf = p.multiplyTransforms(self.current_base_pos, self.current_base_ori, self.current_hand_pos, quaternion_from_euler(self.current_hand_ori[0], self.current_hand_ori[1], self.current_hand_ori[2]))

        solutions = list(p.calculateInverseKinematics(self.robot_ids[0], self.hand_id, self.current_hand_pos_wf, self.current_hand_ori_wf))[-11:]

        goal_joints = [0.0 for i in range(p.getNumJoints(self.robot_ids[0]))]
        if self.knee:
            goal_joints[9] = 0.0
            goal_joints[18] = 0.0
            if self.kneedown:
                goal_joints[9] = -1.57
                goal_joints[18] = -1.77
                # goal_joints[7] = -1.07
            goal_joints[-2] = -1.8
            goal_joints[-4] = 0.5
            goal_joints[-5] = -0.5

        goal_joints[23] = solutions[0]# * 0.0
        goal_joints[24] = solutions[1]# * 0.0
        goal_joints[26] = solutions[2]# * 0.0
        goal_joints[28] = solutions[3]# * 0.0
        goal_joints[29] = solutions[4]# * 0.0
        goal_joints[30] = solutions[5]# * 0.0
        goal_joints[32] = self.gripper
        goal_joints[34] = self.gripper

        # if self.hold_cons and self.gripper < 0.7:
        #     self.first = False # agent 1 grasped!!!
        #
        # if self.hold_cons and self.gripper > 0.7 and not(self.first):
        #     p.removeConstraint(self.hold_cons)
        #     self.hold_cons = None
        #
        # if self.hold_cons == None and self.gripper < 0.7 and self.close_to_target():
        #     self.hold_target()

        forces = [50.0 for i in range(p.getNumJoints(self.robot_ids[0]))]
        forces[32] = 0.2
        forces[34] = 0.2

        p.setJointMotorControlArray(self.robot_ids[0], [i for i in range(p.getNumJoints(self.robot_ids[0]))], p.POSITION_CONTROL, goal_joints, forces = forces)
        p.changeConstraint(self.base_cons, self.current_base_pos, maxForce = 30000.0)

        return
