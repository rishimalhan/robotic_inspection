#! /usr/bin/python3

import rospy
import numpy
import logging
import rosparam
import sys
import tf
import moveit_msgs
from rospy.core import is_shutdown
from planning_utils import (
    generate_waypoints_on_cloud,
    generate_state_space
)
from utilities.robot_utils import (
    bootstrap_system
)
from camera_localization.camera_localization import camera_localization
from utilities.filesystem_utils import (
    get_pkg_path,
    load_yaml
)
from simulated_camera.simulated_camera import SimCamera
from system.planning_utils import(
    state_to_pose,
    tool0_from_camera
)
from system.perception_utils import (
    get_error_state
)
from utilities.robot_utils import InspectionBot
from utilities.voxel_grid import VoxelGrid
from utilities.visualizer import Visualizer

from search_environment import InspectionEnv

logger = logging.getLogger('rosout')


def run_localization(inspection_bot):
    # Run the process of localizing the camera with respect to the robot end-effector
    home_config = rospy.get_param("/robot_positions/home")
    image_capture = rospy.get_param("/robot_positions/image_capture")

    # Check if robot is at home position within a threshold
    current_config = inspection_bot.move_group.get_current_joint_values() # list
    config_distance = numpy.linalg.norm( numpy.subtract(home_config,current_config) )
    if config_distance > 1e-3:
        logger.warning("Robot is not at home position. Euclidean distance: {0}. \
                    Moving the robot to home position".format(config_distance))
        inspection_bot.goal_position.position = home_config
        inspection_bot.execute()

    # Plan from home to image_capture node
    logger.info("Moving to image capture node for camera localization")
    inspection_bot.goal_position.position = image_capture
    inspection_bot.execute()

    # Assuming we are at Aruco node since execute command involves stop
    base_T_camera = camera_localization()

    # Plan from image_capture node to home
    logger.info("Moving back to home")
    inspection_bot.goal_position.position = home_config
    inspection_bot.execute()

    return base_T_camera

def start_simulated_camera(inspection_bot):
    path = get_pkg_path("system")
    stl_path = path + rosparam.get_param("/stl_params/directory_path") + \
                            "/" + rosparam.get_param("/stl_params/name") + ".stl"
    sim_camera = SimCamera(inspection_bot, part_stl_path=stl_path)
    sim_camera.publish_cloud()
    return sim_camera

def main():
    rospy.init_node("main")
    transformer = tf.TransformListener(True, rospy.Duration(10.0))
    inspection_bot = bootstrap_system()
    home_state = rospy.get_param("/robot_positions/home")
    camera_home_state = rospy.get_param("/camera_home")

    # Check if robot is at home position
    inspection_bot.execute_cartesian_path([state_to_pose(tool0_from_camera(home_state,transformer))])
    # current_config = inspection_bot.move_group.get_current_joint_values() # list
    # config_distance = numpy.linalg.norm( numpy.subtract(home_config,current_config) )
    # if config_distance > 1e-3:
    #     logger.warning("Robot is not at home position. Euclidean distance: {0}. \
    #                 Moving the robot to home position".format(config_distance))
    #     inspection_bot.execute(inspection_bot.get_joint_state(home_config))

    sim_camera = start_simulated_camera(inspection_bot)
    inspection_bot.execute_cartesian_path([state_to_pose(tool0_from_camera(camera_home_state,sim_camera.transformer))])
    inspection_env = InspectionEnv(sim_camera, camera_home_state)
    camera_path = inspection_env.greedy_search()
    exec_path = inspection_env.get_executable_path( camera_path )
    inspection_bot.execute_cartesian_path(exec_path,vel_scale=0.1,acc_scale=0.1)

    inspection_bot.wrap_up()

if __name__=='__main__':
    main()
    rospy.signal_shutdown("Task complete")
