#!/usr/bin/env python3

import numpy
import open3d
import rospy
import rosparam
from utilities.robot_utils import (
    bootstrap_system
)
from utilities.filesystem_utils import (
    get_pkg_path,
    load_yaml
)
from simulated_camera.simulated_camera import SimCamera

def main():
    inspection_bot = bootstrap_system()
    path = get_pkg_path("system")
    stl_path = path + rosparam.get_param("/stl_params/directory_path") + \
                            "/" + rosparam.get_param("/stl_params/name") + ".stl"
    sim_camera = SimCamera(inspection_bot, part_stl_path=stl_path)
    sim_camera.publish_cloud()

    home_config = rospy.get_param("/robot_positions/home")
    camera_home = rospy.get_param("/camera_home")
    cloud_capture_config_1 = [0.19, 0, 0.61, 0, 0.87, 0] # Tied to stl_params transform
    cloud_capture_config_2 = [0.1, 0.19, 0.45, 0, 0.87, 0] # Tied to stl_params transform

    # inspection_bot.goal_position.position = home_config
    # inspection_bot.execute()
    inspection_bot.goal_position.position = cloud_capture_config_1
    inspection_bot.execute()
    # rospy.sleep(0.5)
    # inspection_bot.goal_position.position = cloud_capture_config_2
    # inspection_bot.execute()
    
    # rospy.signal_shutdown("Task complete")


if __name__=='__main__':
    rospy.init_node("icp_test")
    main()