#! /usr/bin/python3

import rospy
import numpy
import logging
import rosparam
import sys
import open3d
import logging
from system.planning_utils import state_to_matrix
logger = logging.getLogger("rosout")
from utilities.filesystem_utils import (
    get_pkg_path,
)

gen_ref_cloud = True
gen_measured_cloud = True

# part = "partA"
part = "partB"

pkg_path = get_pkg_path("system")
cloud_path = pkg_path + "/database/" + part + "/" + part + ".ply"
part_tf_file = pkg_path + "/database/" + part + "/" + "part_tf.csv"
part_tf = numpy.loadtxt(part_tf_file,delimiter=",")

if gen_ref_cloud:
    # Generate a reference:
    logger.info("Reading stl for the part")
    stl_path = pkg_path + rosparam.get_param("/stl_params/directory_path") + \
                                "/" + rosparam.get_param("/stl_params/name") + ".stl"
    logger.info("Reading stl. Path: {0}".format(stl_path))
    mesh = open3d.io.read_triangle_mesh(stl_path)
    logger.info("Stl read. Generating PointCloud from stl")
    mesh = mesh.transform(state_to_matrix(part_tf))
    # Point cloud of STL surface only
    filters = rospy.get_param("/stl_params").get("filters")
    stl_cloud = mesh.sample_points_poisson_disk(number_of_points=20000)
    logger.info("Writing reference pointcloud to file")
    open3d.io.write_point_cloud(pkg_path+"/database/"+part+"/"+"reference.ply", stl_cloud)
    logger.info("Written")

# Read the reference cloud
reference = open3d.io.read_point_cloud(pkg_path+"/database/"+part+"/"+"reference.ply")
reference = reference.voxel_down_sample(voxel_size=0.005)
print("Reference cloud # points: ", reference)

if gen_measured_cloud:
    # Apply uncertainty average to the points
    cloud = open3d.io.read_point_cloud(cloud_path)
    print("Measured cloud # points: ", cloud)
    cloud.estimate_normals()
    cloud.normalize_normals()
    points = numpy.asarray(reference.points)
    normals = numpy.asarray(reference.normals)
    final_cloud = []
    logger.info("Generating the measured cloud")
    for i in range(points.shape[0]):
        matrix = numpy.identity(4)
        matrix[0:3,2] = normals[i]
        matrix[0:3,0] = [1,0,0]
        matrix[0:3,1] = numpy.cross(matrix[0:3,2],matrix[0:3,0])
        bbox = open3d.geometry.OrientedBoundingBox(center=points[i], R=matrix[0:3,0:3], extent=[0.002,0.002,0.01])
        nugget = numpy.asarray(cloud.crop(bbox).points)
        if nugget.shape[0] > 0:
            final_cloud.append(numpy.average(nugget,axis=0))
    final_cloud = numpy.array(final_cloud)
    measured_cloud = open3d.geometry.PointCloud()
    measured_cloud.points = open3d.utility.Vector3dVector(final_cloud)
    measured_cloud.colors = open3d.utility.Vector3dVector( 
                        numpy.ones(final_cloud.shape)*[0.447,0.62,0.811])
    logger.info("Generated. Writing the measured cloud to file")
    open3d.io.write_point_cloud(pkg_path + "/database/" + part + "/" + "measured_cloud.ply", measured_cloud)

# Read the measured cloud
logger.info("Reading the measured cloud")
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