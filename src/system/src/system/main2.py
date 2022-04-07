#! /usr/bin/python3

import rospy
import numpy
import logging
import rosparam
import sys
import open3d
import tf
import moveit_msgs
from rospy.core import is_shutdown
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
    state_to_matrix,
    update_cloud,
    generate_zigzag
)
from utilities.robot_utils import InspectionBot
from utilities.voxel_grid import VoxelGrid
from utilities.visualizer import Visualizer

from search_environment import InspectionEnv

logger = logging.getLogger('rosout')



def main():
    save_files = True
    rospy.init_node("main")
    transformer = tf.TransformListener(True, rospy.Duration(10.0))
    inspection_bot = bootstrap_system() # I need this.

    # If you have a home location    
    # inspection_bot.execute_cartesian_path([state_to_pose(home)])
    #########################################
    path = get_pkg_path("system")
    plan_path = path + "/database/" + rosparam.get_param("/stl_params/name") + "/planned_camera_path.csv"

    logger.info("Generating Global Path")
    exec_path = numpy.loadtxt(plan_path,delimiter=",")
    logger.info("Number of points in path: %d",len(exec_path))
    ############################################

    viz = Visualizer()
    viz.axes = exec_path
    viz.start_visualizer_async()

    inspection_bot.execute_cartesian_path( [state_to_pose(tool0_from_camera(waypoint,transformer)) for waypoint in exec_path],vel_scale=0.02 )
    
    logger.info("Executing the path")

if __name__=='__main__':
    main()
    logger.info("Signalling Shutdown")
    rospy.signal_shutdown("Task complete")




# camera_path = camera_path[0:-40]
# inspection_bot.execute_cartesian_path( [state_to_pose(tool0_from_camera(camera_state, transformer)) for camera_state in camera_path],vel_scale=0.01 )
# camera.heatmap_bounds = numpy.array([ [0,0,0],[1.0,1.0,3.14] ])
# camera_path = numpy.array([ [0.093, 1.072, 0.585, 3.107, -0.043, -0.099],
#                     [0.154, 1.359, 0.472, 2.350, 0.121, -0.220],
#                     [0.102, 1.330, 0.518, 2.350, 0.121, -0.220],
#                     [0.058, 1.344, 0.497, 2.497, 0.121, -0.220],
#                     [0.025, 1.353, 0.482, 2.493, -0.154, -0.014],
#                     [-0.030, 1.371, 0.461, 2.580, -0.193, 0.010],
#                     [-0.128, 1.344, 0.455, 2.430, -0.301, 0.104],
#                     [0.093, 1.072, 0.585, 3.107, -0.043, -0.099] ])
# (exec_path, joint_states) = inspection_env.get_executable_path( camera_path )
# viz = Visualizer()
# viz.axes = exec_path
# viz.start_visualizer_async()
# inspection_bot.execute_joint_path(joint_states, camera)
# addon = open3d.io.read_point_cloud( "/home/rmalhan/Documents/addon.ply" )
# camera.voxel_grid.update_grid(addon)
# rospy.sleep(0.2)
# inspection_bot.execute_cartesian_path([state_to_pose(tool0_from_camera(camera.camera_home, transformer))])