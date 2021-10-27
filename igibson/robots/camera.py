import gym
import numpy as np
import pybullet as p
from igibson.robots.robot_locomotor import LocomotorRobot
import time


class Camera(LocomotorRobot):
    def __init__(self, config):
        self.config = config
        self.velocity = config.get('velocity', 1.0)
        LocomotorRobot.__init__(self,
                                "grippers/basic_gripper/camera.urdf",
                                action_dim=4,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", True),
                                control='velocity')

    def base_reset(self, pos, ori):
        self.base_cons = p.createConstraint(self.robot_ids[0], -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], pos, [0, 0, 0, 1])
        self.current_base_pos = np.array(pos)
        self.current_base_ori = np.array(ori)
        p.changeConstraint(self.base_cons, self.current_base_pos, self.current_base_ori, maxForce=30000.0)

    def set_up_continuous_action_space(self):
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = self.velocity * np.ones([self.action_dim])
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        self.action_list = [[self.velocity, self.velocity, 0, self.velocity],
                            [-self.velocity, -self.velocity, 0, -self.velocity],
                            [self.velocity, -self.velocity, -self.velocity, 0],
                            [-self.velocity, self.velocity, self.velocity, 0], [0, 0, 0, 0]]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  ## forward
            (ord('s'),): 1,  ## backward
            (ord('d'),): 2,  ## turn right
            (ord('a'),): 3,  ## turn left
            (): 4
        }

    def apply_action(self, pos, ori):
        self.set_position(pos)
        self.set_orientation(ori)