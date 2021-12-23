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

cloud_name = "bad"

part = "partB"

pkg_path = get_pkg_path("system")
part_tf_file = pkg_path + "/database/" + part + "/" + "part_tf.csv"
part_tf = numpy.loadtxt(part_tf_file,delimiter=",")
path = "/home/rmalhan/extra_results/"

# Read the reference cloud
reference = open3d.io.read_point_cloud(path+"reference.ply")
reference = reference.voxel_down_sample(voxel_size=0.01)
print("Reference cloud # points: ", reference)

cloud = open3d.io.read_point_cloud(path+cloud_name+".ply")
print("Cloud # points: ", cloud)
# ICP
reg_p2p = open3d.pipelines.registration.registration_icp(
    source=cloud.voxel_down_sample(voxel_size=0.005), target=reference, max_correspondence_distance=1e-3, init=numpy.identity(4))
logger.info("ICP transform: \n{0}".format(reg_p2p.transformation))
print("Cloud after down sampling # points: ", cloud)
cloud = cloud.transform(reg_p2p.transformation)

# Apply uncertainty average to the points
cloud.estimate_normals()
cloud.normalize_normals()
points = numpy.asarray(reference.points)
normals = numpy.asarray(reference.normals)
averaged_points = []
logger.info("Generating the measured cloud")
for i in range(points.shape[0]):
    matrix = numpy.identity(4)
    matrix[0:3,2] = normals[i]
    matrix[0:3,0] = [1,0,0]
    matrix[0:3,1] = numpy.cross(matrix[0:3,2],matrix[0:3,0])
    bbox = open3d.geometry.OrientedBoundingBox(center=points[i], R=matrix[0:3,0:3], extent=[0.002,0.002,0.04])
    nugget = numpy.asarray(cloud.crop(bbox).points)
    if nugget.shape[0] > 0:
        averaged_points.append(numpy.average(nugget,axis=0))
averaged_points = numpy.array(averaged_points)
measured_cloud = open3d.geometry.PointCloud()
measured_cloud.points = open3d.utility.Vector3dVector(averaged_points)
measured_cloud.colors = open3d.utility.Vector3dVector( 
                    numpy.ones(averaged_points.shape)*[0.447,0.62,0.811])

logger.info("Computing the metrics")
# Get distance map
distances = measured_cloud.compute_point_cloud_distance(reference)
avg = numpy.average(distances)
stdev = numpy.std(distances)
logger.info("Max: {0}. Average: {1}".format(avg+2*stdev, numpy.average(distances)))