#!/usr/bin/env python3

import numpy
import open3d
import rospy
from utilities.filesystem_utils import get_pkg_path

def main():
    path = get_pkg_path("system")
    pcd = open3d.io.read_point_cloud(path + "/pointclouds/calibration_reference.ply")
    # print(pcd)
    # print(numpy.asarray(pcd.points))
    # open3d.visualization.draw_geometries([pcd],
    #                               zoom=0.3412,
    #                               front=[0.4257, -0.2125, -0.8795],
    #                               lookat=[2.6172, 2.0475, 1.532],
    #                               up=[-0.0694, -0.9768, 0.2024])

    pcd_tree = open3d.geometry.KDTreeFlann(pcd)
    
    

if __name__=='__main__':
    rospy.init_node("icp_test")
    main()