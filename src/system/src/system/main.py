#! /usr/bin/python3

from os import wait
import rospy
import numpy
import logging
import moveit_msgs
from rospy.core import is_shutdown
from utilities.filesystem_utils import load_yaml
from utilities.robot_utils import InspectionBot
from camera_localization.bootstrap_camera import bootstrap_camera

logger = logging.getLogger('rosout')

def bootstrap_system(sim=False):
    # Bootstrap the robot parameters
    load_yaml("utilities","system")
    bootstrap_camera()
    inspection_bot = InspectionBot()
    if sim:
        inspection_bot.traj_viz = rospy.Publisher("/move_group/display_planned_path",
                                        moveit_msgs.msg.DisplayTrajectory,queue_size=10)
    return inspection_bot
    
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
        inspection_bot.execute_and_simulate()

    # Plan from home to image_capture node
    inspection_bot.goal_position.position = image_capture
    inspection_bot.execute_and_simulate()

    # Plan from image_capture node to home
    inspection_bot.goal_position.position = home_config
    inspection_bot.execute_and_simulate()

if __name__=='__main__':
    rospy.init_node("main")
    inspection_bot = bootstrap_system(sim=True)
    run_localization(inspection_bot)
    rospy.sleep(2)
    inspection_bot.wrap_up()
