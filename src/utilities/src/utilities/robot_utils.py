#! /usr/bin/python3

from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import rospy
import moveit_msgs
from moveit_commander.robot import RobotCommander
from moveit_commander.planning_scene_interface import PlanningSceneInterface
from utilities.filesystem_utils import load_yaml
from moveit_commander.move_group import MoveGroupCommander
from sensor_msgs.msg import JointState
from geometry_msgs.msg import( 
    Pose,
    PoseStamped
)
import sys
import math
import numpy
import logging
from system.planning_utils import (
    state_to_pose
)
from tf.transformations import (
    quaternion_matrix,
    quaternion_from_matrix
)
from moveit_msgs.msg import (
    Constraints, 
    OrientationConstraint,
)
from geometry_msgs.msg import Quaternion
import rosparam
from moveit_msgs.msg import (
    RobotState
)
from std_msgs.msg import Header
from .get_fk import GetFK
from .get_ik import GetIK
from trajectory_msgs.msg import (
    JointTrajectory,
    JointTrajectoryPoint
)
from industrial_msgs.srv import CmdJointTrajectory, CmdJointTrajectoryResponse, CmdJointTrajectoryRequest
from pilz_robot_programming.robot import (
    Robot,
    Sequence,
)
from pilz_robot_programming.commands import (
    Ptp
)
from moveit_msgs.msg import (
    MoveGroupSequenceAction,
    MotionSequenceRequest,
    MotionSequenceResponse,
    MotionPlanRequest
)
from .move_robot import GoToConfiguration
logger = logging.getLogger('rosout')

class InspectionBot:
    def __init__(self, apply_orientation_constraint=False):
        # __REQUIRED_API_VERSION__ = "1"
        # self.pilz_robot = Robot(__REQUIRED_API_VERSION__)
        self.goal_position = JointState()
        self.goal_pose = Pose()
        self.goal_position.name = ["joint_"+str(i+1) for i in range(6)]
        self.group_name = "manipulator"
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface(synchronous=True)
        self.move_group = MoveGroupCommander(self.group_name)
        self.get_fk = GetFK("tool0", "base_link")
        self.get_ik = GetIK(group=self.group_name, ik_attempts=10, avoid_collisions=True)
        # self.trajectory_executor = GoToConfiguration()
        # self.traj_viz = None

        if rospy.get_param("/robot_positions/home"):
            self.robot_home = rospy.get_param("/robot_positions/home")
            self.execute_cartesian_path([state_to_pose(self.robot_home)], avoid_collisions=True)
        else:
            raise logger.warn("Robot home position not found")
        
        if apply_orientation_constraint:
            self.constraints = Constraints()
            self.constraints.name = "tilt constraint"
            tilt_constraint = OrientationConstraint()
            tilt_constraint.header.frame_id = "base"
            # The link that must be oriented downward
            tilt_constraint.link_name = "tool0"
            tilt_constraint.orientation = Quaternion(0.0, 1.0, 0.0, 0.0)
            tilt_constraint.absolute_x_axis_tolerance = 0.6
            tilt_constraint.absolute_y_axis_tolerance = 0.6
            tilt_constraint.absolute_z_axis_tolerance = 0.05
            # The tilt constraint is the only constraint
            tilt_constraint.weight = 1
            self.constraints.orientation_constraints = [tilt_constraint]
            self.move_group.set_path_constraints(self.constraints)
        else:
            self.constraints = None
        return

    def wrap_up(self):
        self.scene.clear()
        rospy.sleep(0.2)
    
    def get_joint_state(self,state):
        config = JointState()
        config.name = ["joint_"+str(i+1) for i in range(6)]
        config.position = state
        return config
    
    def get_pose(self,matrix):
        config = Pose()
        config.position.x = matrix[0,3]
        config.position.y = matrix[1,3]
        config.position.z = matrix[2,3]
        quaternion = quaternion_from_matrix(matrix)
        config.orientation.x = quaternion[0]
        config.orientation.y = quaternion[1]
        config.orientation.z = quaternion[2]
        config.orientation.w = quaternion[3]
        return config

    def execute_cartesian_path(self,waypoints, avoid_collisions=True, async_exec=False, vel_scale=1.0):
        (plan, fraction) = self.move_group.compute_cartesian_path(waypoints, eef_step=0.01, jump_threshold=0.0, avoid_collisions=avoid_collisions
                                        )
        if fraction != 1.0:
            logger.warn("Cartesian planning failure. Only covered {0} fraction of path.".format(fraction))
            return None
        if not async_exec:
            self.move_group.execute( self.move_group.retime_trajectory(
                                self.move_group.get_current_state(),plan,velocity_scaling_factor=vel_scale),wait=True )
            self.move_group.stop()
        else:
            self.move_group.execute( self.move_group.retime_trajectory(
                                self.move_group.get_current_state(),plan,velocity_scaling_factor=vel_scale),wait=False )
        return plan

    def execute(self, goal, async_exec=False, vel_scale=1.0):
        for i in range(5):
            (error_flag, plan, planning_time, error_code) = self.move_group.plan( goal )
            if error_flag:
                break
        if error_flag:
            logger.info("Planning successful. Planning time: {0} s. Executing trajectory"
                                .format(planning_time))
        else:
            logger.warning(error_code)
            return
        if not async_exec:
            self.move_group.execute( self.move_group.retime_trajectory(
                                self.move_group.get_current_state(),plan,velocity_scaling_factor=vel_scale),wait=True )
            self.move_group.stop()
        else:
            self.move_group.execute( self.move_group.retime_trajectory(
                                self.move_group.get_current_state(),plan,velocity_scaling_factor=vel_scale),wait=False )
        return plan
    
    def get_current_forward_kinematics(self):
        current_pose = self.move_group.get_current_pose().pose
        forward_kinematics = quaternion_matrix([current_pose.orientation.x, current_pose.orientation.y,
                                        current_pose.orientation.z, current_pose.orientation.w])
        forward_kinematics[0:3,3] = [current_pose.position.x, current_pose.position.y, current_pose.position.z]
        return numpy.array(forward_kinematics)

    def pose_error(self,joints):
        import IPython
        IPython.embed()
        fk = self.get_fk.get_fk(joints)

    def execute_joint_path(self,joint_states):
        for joint_state in joint_states:
            self.execute( joint_state, vel_scale=0.01 )
        return True

def bootstrap_system(sim_camera=False):
    # Bootstrap the robot parameters
    load_yaml("system", "system")
    inspection_bot = InspectionBot()
    return inspection_bot
