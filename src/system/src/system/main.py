#! /usr/bin/python3

from os import wait
import rospy
import numpy
import logging
import rosparam
import moveit_msgs
from rospy.core import is_shutdown
from planning_utils import generate_path_on_cloud
from utilities.robot_utils import (
    bootstrap_system
)
from camera_localization.camera_localization import camera_localization
from utilities.filesystem_utils import (
    get_pkg_path,
    load_yaml
)
from simulated_camera.simulated_camera import SimCamera

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

if __name__=='__main__':
    rospy.init_node("main")
    inspection_bot = bootstrap_system()
    home_config = rospy.get_param("/robot_positions/home")

    # Check if robot is at home position within a threshold
    current_config = inspection_bot.move_group.get_current_joint_values() # list
    config_distance = numpy.linalg.norm( numpy.subtract(home_config,current_config) )
    if config_distance > 1e-3:
        logger.warning("Robot is not at home position. Euclidean distance: {0}. \
                    Moving the robot to home position".format(config_distance))
        inspection_bot.execute(inspection_bot.get_joint_state(home_config))
    
    sim_camera = start_simulated_camera(inspection_bot)
    path = generate_path_on_cloud(sim_camera.stl_cloud, sim_camera.transformer, inspection_bot.get_pose)
    inspection_bot.execute_cartesian_path(path)
        
    inspection_bot.wrap_up()
    rospy.signal_shutdown("Task complete")
