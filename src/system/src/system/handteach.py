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
from system.planning_utils import(
    tf_to_state
)
from utilities.robot_utils import InspectionBot

logger = logging.getLogger('rosout')

rospy.init_node("handteach")
transformer = tf.TransformListener(True, rospy.Duration(10.0))

waypoints = []
while not rospy.is_shutdown():
    input("Enter to store waypoint")
    waypoints.append( tf_to_state(transformer.lookupTransform("base", "camera_depth_optical_frame", rospy.Time(0))) )
    rospy.sleep(0.5)

path = get_pkg_path("system")
numpy.savetxt(path+"/database/partF/planned_camera_path.csv",numpy.array(waypoints),delimiter=",")