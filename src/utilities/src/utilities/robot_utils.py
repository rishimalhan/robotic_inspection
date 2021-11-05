#! /usr/bin/python3

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

logger = logging.getLogger('rosout')

class InspectionBot:
    def __init__(self, add_collision_obstacles=True, apply_orientation_constraint=False):
        if rospy.get_param("/robot_positions/home"):
            self.robot_home = rospy.get_param("/robot_positions/home")
        else:
            raise logger.warn("Robot home position not found")
        self.goal_position = JointState()
        self.goal_pose = Pose()
        self.goal_position.name = ["joint_"+str(i+1) for i in range(6)]
        self.group_name = "manipulator"
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface(synchronous=True)
        self.move_group = MoveGroupCommander(self.group_name)
        self.traj_viz = None
        self.collision_boxes = {}
        if add_collision_obstacles:
            def box_added(name):
                start = rospy.get_time()
                seconds = rospy.get_time()
                while (seconds - start < 2.0) and not rospy.is_shutdown():
                    # Test if the box is in the scene.
                    is_known = name in self.scene.get_known_object_names()
                    if is_known:
                        return True
                    # Sleep so that we give other threads time on the processor
                    rospy.sleep(0.1)
                    seconds = rospy.get_time()
                # If we exited the while loop without returning then we timed out
                return False

            # Get collision boxes
            self.collision_boxes = rospy.get_param("/collision_boxes")
            for key in self.collision_boxes.keys():
                pose = PoseStamped()
                pose.header.frame_id = self.collision_boxes[key]['frame_id']
                pose.pose.position.x = self.collision_boxes[key]['position'][0]
                pose.pose.position.y = self.collision_boxes[key]['position'][1]
                pose.pose.position.z = self.collision_boxes[key]['position'][2]
                pose.pose.orientation.x = self.collision_boxes[key]['orientation'][0]
                pose.pose.orientation.y = self.collision_boxes[key]['orientation'][1]
                pose.pose.orientation.z = self.collision_boxes[key]['orientation'][2]
                pose.pose.orientation.w = self.collision_boxes[key]['orientation'][3]
                self.scene.add_box( self.collision_boxes[key]['name'], pose, 
                                    size=self.collision_boxes[key]['dimension'])
                if box_added(key):
                    logger.info("Collision object {0} added".format(key))
                else:
                    raise Exception("Unable to add collision object: ", key)
            logger.info("All collision objects added")
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
        self.execute_cartesian_path([state_to_pose(self.robot_home)])
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

    def execute_cartesian_path(self,waypoints, async_exec=False, vel_scale=1.0, acc_scale=1.0):
        (plan, fraction) = self.move_group.compute_cartesian_path(waypoints, eef_step=0.01, jump_threshold=0.0,
                                        )
        if fraction != 1.0:
            logger.warn("Cartesian planning failure. Only covered {0} fraction of path.".format(fraction))
            return
        if not async_exec:
            self.move_group.execute( self.move_group.retime_trajectory(
                                self.move_group.get_current_state(),plan,velocity_scaling_factor=vel_scale,
                                acceleration_scaling_factor=acc_scale),wait=True )
            self.move_group.stop()
        else:
            self.move_group.execute( self.move_group.retime_trajectory(
                                self.move_group.get_current_state(),plan,velocity_scaling_factor=vel_scale,
                                acceleration_scaling_factor=acc_scale),wait=False )
        return plan

    def execute(self, goal, async_exec=False, vel_scale=1.0, acc_scale=1.0):
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
                                self.move_group.get_current_state(),plan,velocity_scaling_factor=vel_scale,
                                acceleration_scaling_factor=acc_scale),wait=True )
            self.move_group.stop()
        else:
            self.move_group.execute( self.move_group.retime_trajectory(
                                self.move_group.get_current_state(),plan,velocity_scaling_factor=vel_scale,
                                acceleration_scaling_factor=acc_scale),wait=False )
        return plan
    
    def get_current_forward_kinematics(self):
        current_pose = self.move_group.get_current_pose().pose
        forward_kinematics = quaternion_matrix([current_pose.orientation.x, current_pose.orientation.y,
                                        current_pose.orientation.z, current_pose.orientation.w])
        forward_kinematics[0:3,3] = [current_pose.position.x, current_pose.position.y, current_pose.position.z]
        return numpy.array(forward_kinematics)


def bootstrap_system(sim_camera=False):
    # Bootstrap the robot parameters
    load_yaml("system", "system")
    inspection_bot = InspectionBot()
    return inspection_bot
