#! /usr/bin/python3

import rospy
import numpy
import logging
import rosparam
import sys
import open3d
import math
import logging
from os.path import exists
from system.planning_utils import state_to_matrix
logger = logging.getLogger("rosout")
from utilities.filesystem_utils import (
    get_pkg_path,
)

N = 0.05

def reject_outliers(data, m = 1.5):
    std_dev = numpy.std(data,axis=0)
    avg = numpy.average(data,axis=0)
    indices = numpy.unique(numpy.hstack((
                numpy.where( (data[:,0]>avg[0]-m*std_dev[0]) &
                (data[:,1]>avg[1]-m*std_dev[1]) &
                (data[:,2]>avg[2]-m*std_dev[2]) ),
                numpy.where( (data[:,0]<avg[0]+m*std_dev[0]) &
                (data[:,1]<avg[1]+m*std_dev[1]) &
                (data[:,2]<avg[2]+m*std_dev[2]) )
    )))
    return data[indices]

part = "partB"

pkg_path = get_pkg_path("system")
part_tf_file = pkg_path + "/database/" + part + "/" + "part_tf.csv"
part_tf = numpy.loadtxt(part_tf_file,delimiter=",")
path = "/home/rmalhan/extra_results/"

# Read the reference cloud
reference = open3d.io.read_point_cloud(path+"reference.ply")
reference = reference.voxel_down_sample(voxel_size=0.001)
print("Reference cloud # points: ", reference)

# cloud = open3d.io.read_point_cloud(path+"n10.ply")

cloud_names = [ 'cloud1', 'cloud2', 'cloud3', 'cloud4', 'cloud5' ]
points = []
for cloud_name in cloud_names:
    cloud = open3d.io.read_point_cloud(path+cloud_name+".ply")
    points.extend( numpy.asarray(cloud.points) )
points = numpy.array(points)
cloud = open3d.geometry.PointCloud()
cloud.points = open3d.utility.Vector3dVector(points)
print("Cloud # points: ", cloud)

ref_cloud_indices = numpy.random.choice(numpy.asarray(reference.points).shape[0],size=1000,replace=False)

# Apply uncertainty average to the points
cloud.estimate_normals()
cloud.normalize_normals()
points = numpy.asarray(reference.points)
normals = numpy.asarray(reference.normals)
averaged_points = []
logger.info("Generating the measured cloud")
for i in ref_cloud_indices:
    matrix = numpy.identity(3)
    matrix[0:3,2] = normals[i]
    matrix[0:3,0] = [1,0,0]
    matrix[0:3,1] = numpy.cross(matrix[0:3,2],matrix[0:3,0])
    matrix /= numpy.linalg.norm(matrix,axis=0)
    bbox = open3d.geometry.OrientedBoundingBox(center=points[i], R=matrix, extent=[0.0015,0.0015,0.02])
    nugget = numpy.asarray(cloud.crop(bbox).points)
    if nugget.shape[0] > 30:
        number_of_rows = nugget.shape[0]
        random_indices = numpy.random.choice(number_of_rows, size=math.ceil(N*number_of_rows), replace=False)
        nugget = nugget[random_indices]
        nugget = reject_outliers(nugget)
        averaged_points.append(numpy.average(nugget,axis=0))
averaged_points = numpy.array(averaged_points)
measured_cloud = open3d.geometry.PointCloud()
measured_cloud.points = open3d.utility.Vector3dVector(averaged_points)
measured_cloud.colors = open3d.utility.Vector3dVector( 
                    numpy.ones(averaged_points.shape)*[0.447,0.62,0.811])

print(measured_cloud)

logger.info("Computing the metrics")
# Get distance map
distances = measured_cloud.compute_point_cloud_distance(reference)
avg = numpy.average(distances)
stdev = numpy.std(distances)
logger.info("Max: {0}. Average: {1}".format(numpy.max(distances), numpy.average(distances)))