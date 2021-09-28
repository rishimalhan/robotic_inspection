#! /usr/bin/python3

from os import wait
import rospy
import numpy
import logging
import moveit_msgs
from rospy.core import is_shutdown
from sensor_msgs.msg import JointState
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
        goal_position = JointState()
        goal_position.name = ["joint_"+str(i+1) for i in range(6)]
        goal_position.position = home_config
        (error_flag, plan, planning_time, error_code) = inspection_bot.move_group.plan( goal_position )
        if error_flag:
            logger.info("Planning to home position successful. Planning time: {0} s. Executing trajectory"
                                .format(planning_time))
            inspection_bot.move_group.execute( plan,wait=True )
            inspection_bot.move_group.stop()
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = inspection_bot.robot.get_current_state()
            display_trajectory.trajectory.append(plan)
            inspection_bot.traj_viz.publish(display_trajectory)
        else:
            logger.warning(error_code)


    # Plan from home to image_capture node

if __name__=='__main__':
    rospy.init_node("main")
    inspection_bot = bootstrap_system(sim=True)
    run_localization(inspection_bot)
    rospy.sleep(2)
    # inspection_bot.wrap_up()
