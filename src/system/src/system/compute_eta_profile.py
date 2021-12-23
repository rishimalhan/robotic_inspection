#! /usr/bin/python3

import rospy
import rosparam
import numpy
import sys
from utilities.robot_utils import (
    bootstrap_system
)
import tf
from utilities.filesystem_utils import (
    get_pkg_path,
)
from utilities.visualizer import Visualizer


rospy.init_node("main")
path = get_pkg_path("system")
data_path = path + "/database/camera_data.csv"

camera_data = numpy.loadtxt(data_path,delimiter=",")

print(camera_data.shape)

sys.exit()

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
fig = plt.figure()
ax = Axes3D(fig)

idx = numpy.arange(0,camera_data.shape[0],10)

ax.scatter( camera_data[idx,0], camera_data[idx,1], camera_data[idx,2], c=camera_data[idx,3] )

plt.show()
