#!/usr/bin/env python3

import numpy
import open3d
from system.planning_utils import (
    state_to_matrix
)
from camera_localization import Localizer
import rospy
import sys

rospy.init_node("main")

init_guess = numpy.array([-0.0598154,   0.0418941,   0.0639608,   -0.494072,  0.00873799,     1.57854])

pcds = []
for i in range(1,10):
    open3d_cloud = open3d.io.read_point_cloud("/home/rmalhan/robotic_inspection/src/camera/tests/abb120/pointcloud/pos_"+str(i)+".pcd")
    open3d_cloud = open3d_cloud.voxel_down_sample(voxel_size=0.01)
    open3d_cloud = open3d_cloud.remove_statistical_outlier(nb_neighbors=20,std_ratio=0.005)[0]
    pcds.append( open3d_cloud )
# open3d.visualization.draw_geometries(pcds)
# sys.exit()

transform_file = numpy.loadtxt("/home/rmalhan/robotic_inspection/src/camera/tests/abb120/BaseToFlange.csv", delimiter=",")
transforms = []
for transform in transform_file:
    transforms.append(state_to_matrix(transform))
localizer = Localizer(pcds,transforms,init_guess)
print(localizer.localize())