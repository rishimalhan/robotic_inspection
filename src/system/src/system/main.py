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
    update_cloud,
    generate_zigzag
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

def start_camera(inspection_bot,transformer, flags):
    camera_properties = rospy.get_param("/camera")
    return Camera(inspection_bot,transformer,camera_properties, flags)

def main():
    rospy.init_node("main")
    transformer = tf.TransformListener(True, rospy.Duration(10.0))
    inspection_bot = bootstrap_system()
    camera = start_camera(inspection_bot,transformer=transformer, flags=sys.argv)
    inspection_bot.execute_cartesian_path([state_to_pose(tool0_from_camera(camera.camera_home, transformer))])
    path = get_pkg_path("system")
    plan_path = path + "/database/planned_camera_path.csv"
    inspection_env = InspectionEnv(inspection_bot, camera, sys.argv)

    logger.info("Generating Global Path")
    if "plan" in sys.argv:
        camera_path = inspection_env.greedy_search()
        (exec_path, joint_states) = inspection_env.get_executable_path( camera_path )
        numpy.savetxt(plan_path,camera_path,delimiter=",")
    else:
        camera_path = numpy.loadtxt(plan_path,delimiter=",")
        (exec_path, joint_states) = inspection_env.get_executable_path( camera_path )
    logger.info("Number of points in path: %d",len(exec_path))
    
    viz = Visualizer()
    viz.axes = exec_path
    viz.start_visualizer_async()
    
    camera.construct_cloud()
    logger.info("Executing the path")
    if joint_states is not None:
        inspection_bot.execute_joint_path(joint_states)
        rospy.sleep(0.2)
        logger.info("Inspection complete. Writing pointcloud to file and exiting.")
        constructed_cloud_path = path + "/database/output_cloud.ply"
        open3d.io.write_point_cloud(constructed_cloud_path, camera.op_cloud)
        rospy.sleep(0.5)
        logger.info("Pointcloud written.")
    else:
        logger.info("Planning Failure.")

    # if inspection_bot.execute_cartesian_path(exec_path,vel_scale=1.0) is not None:
    #     logger.info("Inspection complete. Writing pointcloud to file and exiting.")
    #     constructed_cloud_path = path + "/database/output_cloud.ply"
    #     open3d.io.write_point_cloud(constructed_cloud_path, camera.voxel_grid.get_cloud())
    #     logger.info("Pointcloud written.")
    # else:
    #     logger.warn("Planning failure")

    # # Check if robot is at home position
    # camera_home_state = [0.207, 0.933, 0.650, 3.14, 0, 0]
    # sim_camera = start_simulated_camera(inspection_bot, start_publisher=False)
    # inspection_bot.execute_cartesian_path([state_to_pose(tool0_from_camera(camera_home_state,sim_camera.transformer))])
        
    # while not rospy.is_shutdown():
    #     update_cloud_live(sim_camera)
    #     rospy.sleep(0.1)

if __name__=='__main__':
    main()
    rospy.signal_shutdown("Task complete")
