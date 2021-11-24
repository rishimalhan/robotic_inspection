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

# part = "partA"
# part = "partB"
part = "partC"
# part = "partD"

pkg_path = get_pkg_path("system")
cloud_path = pkg_path + "/database/" + part + "/" + part + ".ply"
online_cloud_path = pkg_path + "/database/" + part + "/" + "online.ply"
part_tf_file = pkg_path + "/database/" + part + "/" + "part_tf.csv"
part_tf = numpy.loadtxt(part_tf_file,delimiter=",")

# Read the reference cloud
reference = open3d.io.read_point_cloud(pkg_path+"/database/"+part+"/"+"reference.ply")
reference = reference.voxel_down_sample(voxel_size=0.005)
print("Reference cloud # points: ", reference)

if gen_measured_cloud:
    cloud = open3d.io.read_point_cloud(cloud_path)
    print("Cloud # points: ", cloud)
    if use_online_cloud:
        online_cloud = open3d.io.read_point_cloud(online_cloud_path)
        points = numpy.append(numpy.asarray(cloud.points),numpy.asarray(online_cloud.points),axis=0)
        cloud = open3d.geometry.PointCloud()
        cloud.points = open3d.utility.Vector3dVector(points)
        cloud.colors = open3d.utility.Vector3dVector( 
                        numpy.ones(numpy.asarray(cloud.points).shape)*[0.447,0.62,0.811])
        print("Cloud after online merging # points: ", cloud)
    # ICP
    reg_p2p = open3d.pipelines.registration.registration_icp(
        source=cloud.voxel_down_sample(voxel_size=0.005), target=reference, max_correspondence_distance=1e-3, init=numpy.identity(4))
    logger.info("ICP transform: \n{0}".format(reg_p2p.transformation))
    cloud = cloud.voxel_down_sample(voxel_size=0.0005)
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
        bbox = open3d.geometry.OrientedBoundingBox(center=points[i], R=matrix[0:3,0:3], extent=[0.0015,0.0015,0.02])
        nugget = numpy.asarray(cloud.crop(bbox).points)
        if nugget.shape[0] > 0:
            averaged_points.append(numpy.average(nugget,axis=0))
    averaged_points = numpy.array(averaged_points)
    measured_cloud = open3d.geometry.PointCloud()
    measured_cloud.points = open3d.utility.Vector3dVector(averaged_points)
    measured_cloud.colors = open3d.utility.Vector3dVector( 
                        numpy.ones(averaged_points.shape)*[0.447,0.62,0.811])
    if save_clouds:
        logger.info("Generated. Writing the measured cloud to file")
        if use_online_cloud:
            open3d.io.write_point_cloud(pkg_path + "/database/" + part + "/" + "constructed_output.ply", measured_cloud)
        else:
            open3d.io.write_point_cloud(pkg_path + "/database/" + part + "/" + "measured_cloud.ply", measured_cloud)

# Read the measured cloud
logger.info("Reading the measured cloud")
if use_online_cloud:
    measured_cloud = open3d.io.read_point_cloud(pkg_path+"/database/"+part+"/"+"constructed_output.ply")
else:
    measured_cloud = open3d.io.read_point_cloud(pkg_path+"/database/"+part+"/"+"measured_cloud.ply")
print("Measured cloud # points: ", measured_cloud)

logger.info("ICP")
# Generate color map
# ICP
reg_p2p = open3d.pipelines.registration.registration_icp(
    source=measured_cloud, target=reference, max_correspondence_distance=1e-3, init=numpy.identity(4))
logger.info("ICP transform: \n{0}".format(reg_p2p.transformation))
measured_cloud = measured_cloud.transform(reg_p2p.transformation)

logger.info("Computing the metrics")
# Get distance map
distances = measured_cloud.compute_point_cloud_distance(reference)
avg = numpy.average(distances)
stdev = numpy.std(distances)
logger.info("Max: {0}. Average: {1}".format(avg+2*stdev, numpy.average(distances)))

from matplotlib import cm
distances = numpy.subtract(distances,numpy.min(distances))/ (numpy.max(distances-numpy.min(distances)))
colormap = cm.jet( distances )
measured_cloud.colors = open3d.utility.Vector3dVector( colormap[:,0:3] )
open3d.visualization.draw_geometries( [measured_cloud] )