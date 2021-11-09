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
from utilities.filesystem_utils import (
    get_pkg_path,
    load_yaml
)
from simulated_camera.simulated_camera import SimCamera
from camera.camera import Camera
from system.planning_utils import(
    state_to_pose,
    tool0_from_camera,
    update_cloud,
    update_cloud_live
)
from utilities.robot_utils import InspectionBot
from utilities.voxel_grid import VoxelGrid
from utilities.visualizer import Visualizer

from search_environment import InspectionEnv

logger = logging.getLogger('rosout')


def start_simulated_camera(inspection_bot, start_publisher):
    path = get_pkg_path("system")
    stl_path = path + rosparam.get_param("/stl_params/directory_path") + \
                            "/" + rosparam.get_param("/stl_params/name") + ".stl"
    sim_camera = SimCamera(inspection_bot, part_stl_path=stl_path)
    if start_publisher:
        sim_camera.publish_cloud()
    return sim_camera

def start_camera(inspection_bot,transformer, localize):
    camera_properties = rospy.get_param("/camera")
    return Camera(inspection_bot,transformer,camera_properties, localize)

def main():
    rospy.init_node("main")
    if len(sys.argv) < 3:
        print("usage: main.py localize plan")
    else:
        localize = sys.argv[1] 
        plan = sys.argv[2]
    transformer = tf.TransformListener(True, rospy.Duration(10.0))
    inspection_bot = bootstrap_system()
    
    # camera = start_camera(inspection_bot,transformer=transformer, localize=localize)
    # sys.exit()

    # Check if robot is at home position
    camera_home_state = [0.207, 0.933, 0.650, 3.14, 0, 0]
    sim_camera = start_simulated_camera(inspection_bot, start_publisher=False)
    inspection_bot.execute_cartesian_path([state_to_pose(tool0_from_camera(camera_home_state,sim_camera.transformer))])
        
    while not rospy.is_shutdown():
        update_cloud_live(sim_camera)
        rospy.sleep(0.1)
    sys.exit()
    inspection_env = InspectionEnv(sim_camera, camera_home_state)
    camera_path = inspection_env.greedy_search()
    exec_path = inspection_env.get_executable_path( camera_path )
    inspection_bot.execute_cartesian_path(exec_path,vel_scale=0.1,acc_scale=0.1)

    inspection_bot.wrap_up()

if __name__=='__main__':
    main()
    rospy.signal_shutdown("Task complete")
