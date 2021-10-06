#! /usr/bin/python3

import rospy
import moveit_msgs
from moveit_commander.robot import RobotCommander
from moveit_commander.planning_scene_interface import PlanningSceneInterface
from camera_localization.bootstrap_camera import bootstrap_camera
from utilities.filesystem_utils import load_yaml
from moveit_commander.move_group import MoveGroupCommander
from sensor_msgs.msg import JointState
from geometry_msgs.msg import( 
    Pose,
    PoseStamped
)
import numpy
import logging
from tf.transformations import quaternion_matrix

logger = logging.getLogger('rosout')

class InspectionBot:
    def __init__(self, add_collision_obstacles=True):
        self.goal_position = JointState()
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
        return

    def wrap_up(self):
        self.scene.clear()
        rospy.sleep(0.1)
    
    def execute(self):
        (error_flag, plan, planning_time, error_code) = self.move_group.plan( self.goal_position )
        if error_flag:
            logger.info("Planning successful. Planning time: {0} s. Executing trajectory"
                                .format(planning_time))
        else:
            logger.warning(error_code)
            return
        self.move_group.execute( plan,wait=True )
        self.move_group.stop()
        self.move_group.execute( plan,wait=True )
        self.move_group.stop()
        return plan
    
    def get_forward_kinematics(self):
        current_pose = self.move_group.get_current_pose().pose
        forward_kinematics = quaternion_matrix([current_pose.orientation.w, current_pose.orientation.x, 
                                        current_pose.orientation.y, current_pose.orientation.z])
        forward_kinematics[0:3,3] = [current_pose.position.x, current_pose.position.y, current_pose.position.z]
        return numpy.array(forward_kinematics)


def bootstrap_system(sim_camera=False):
    # Bootstrap the robot parameters
    load_yaml("system", "system")
    bootstrap_camera()
    inspection_bot = InspectionBot()
    return inspection_bot