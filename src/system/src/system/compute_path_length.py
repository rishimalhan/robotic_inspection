#! /usr/bin/python3

import rospy
import rosparam
import numpy
from utilities.robot_utils import (
    bootstrap_system
)
import tf
from utilities.filesystem_utils import (
    get_pkg_path,
)
from utilities.visualizer import Visualizer

# partA : 44 points
# partC : 80 points
# partB : 110 points

rospy.init_node("main")
path = get_pkg_path("system")
plan_path = path + "/database/" + rosparam.get_param("/stl_params/name") + "/planned_camera_path.csv"
plan_path = "/home/rmalhan/extra_results/planned_camera_path.csv"
camera_path = numpy.loadtxt(plan_path,delimiter=",")

# viz = Visualizer()
# viz.axes = camera_path
# viz.start_visualizer()

print( numpy.sum(numpy.linalg.norm(camera_path[1:,0:3]-camera_path[0:-1,0:3],axis=1)) )