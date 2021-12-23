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


online_path = numpy.array([ [-0.089, 1.313, 0.601, 2.651, -0.022, -0.100],
                [-0.088, 1.335, 0.643, 2.651, -0.022, -0.100],
                [-0.092, 1.267, 0.516, 2.651, -0.022, -0.100],
                [-0.092, 1.236, 0.611, 2.775, -0.170, -0.014],
                [0.050, 1.282, 0.550, 2.775, -0.170, -0.014],
                [-0.043, 1.181, 0.576, 2.899, -0.193, -0.005] ])


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
    save_files = True
    rospy.init_node("main")
    transformer = tf.TransformListener(True, rospy.Duration(10.0))
    inspection_bot = bootstrap_system()
    camera = start_camera(inspection_bot,transformer=transformer, flags=sys.argv)
    
    inspection_bot.execute_cartesian_path([state_to_pose(tool0_from_camera(camera.camera_home, transformer))])

    path = get_pkg_path("system")
    plan_path = path + "/database/" + rosparam.get_param("/stl_params/name") + "/planned_camera_path.csv"
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

    print( numpy.sum(numpy.linalg.norm(camera_path[1:,0:3]-camera_path[0:-1,0:3],axis=1)) )
    
    viz = Visualizer()
    viz.axes = exec_path
    viz.start_visualizer_async()

    camera.construct_cloud()

    # # For online cloud
    # while not rospy.is_shutdown():
    #     rospy.sleep(0.1)
    # online_cloud_path = path + "/database/" + rosparam.get_param("/stl_params/name") + "/online.ply"
    # open3d.io.write_point_cloud(online_cloud_path, camera.op_cloud)
    # sys.exit()

    logger.info("Executing the path")
    if joint_states is not None:
        # inspection_bot.execute_joint_path(joint_states, camera)
        inspection_bot.execute_cartesian_path( [state_to_pose(tool0_from_camera(camera_state, transformer)) for camera_state in camera_path],vel_scale=0.01 )
        rospy.sleep(0.2)
        inspection_bot.execute_cartesian_path([state_to_pose(tool0_from_camera(camera.camera_home, transformer))])
        logger.info("Inspection complete.")
        logger.info("Inspection complete. Writing pointcloud to file and exiting.")
        constructed_cloud_path = path + "/database/" + rosparam.get_param("/stl_params/name") + "/" + rosparam.get_param("/stl_params/name") + ".ply"
        if save_files:
            open3d.io.write_point_cloud(constructed_cloud_path, camera.op_cloud)
            rospy.sleep(0.5)
            logger.info("Pointcloud written.")
    else:
        logger.info("Planning Failure.")

if __name__=='__main__':
    main()
    logger.info("Signalling Shutdown")
    rospy.signal_shutdown("Task complete")
