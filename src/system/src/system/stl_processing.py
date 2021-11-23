#! /usr/bin/python3

import rospy
import numpy
import logging
import rosparam
import sys
import open3d
import logging
from os.path import exists
from system.planning_utils import state_to_matrix
logger = logging.getLogger("rosout")
from utilities.filesystem_utils import (
    get_pkg_path,
)

gen_ref_cloud = False
gen_measured_cloud = True
use_online_cloud = True
save_clouds = True

part = "partD"
pkg_path = get_pkg_path("system")
stl_path = pkg_path + rosparam.get_param("/stl_params/directory_path") + \
                                "/" + rosparam.get_param("/stl_params/name") + ".stl"
logger.info("Reading stl. Path: {0}".format(stl_path))
mesh = open3d.io.read_triangle_mesh(stl_path)
logger.info("Stl read. Generating PointCloud from stl")

