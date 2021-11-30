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
import copy


gen_ref_cloud = False
gen_measured_cloud = True
use_online_cloud = False
save_clouds = True

# part = "partA"
# part = "partB"
# part = "partC"
part = "partD"

pkg_path = get_pkg_path("system")
cloud_path = pkg_path + "/database/" + part + "/" + part + ".ply"
online_cloud_path = pkg_path + "/database/" + part + "/" + "online.ply"
part_tf_file = pkg_path + "/database/" + part + "/" + "part_tf.csv"
part_tf = numpy.loadtxt(part_tf_file,delimiter=",")

cloud_names = ["reference", "baseline1", "baseline2", "offline", "approach"]
clouds = []
for i,cloud_name in enumerate(cloud_names):
    cloud = open3d.io.read_point_cloud(pkg_path+"/database/"+part+"/"+cloud_name+".ply")
    if i==0:
        cloud = cloud.voxel_down_sample(voxel_size=0.005)
    print(cloud_name+" cloud # points: ", cloud)
    clouds.append(cloud)

agg_distances = []
distances = []
threshold = 1e-3
for i,cloud in enumerate(clouds[1:]):
    # ICP
    reg_p2p = open3d.pipelines.registration.registration_icp(
        cloud, clouds[0], threshold, numpy.identity(4), 
                                    open3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                    open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000))
    # logger.info("ICP transform: \n{0}".format(reg_p2p.transformation))
    clouds[i+1] = cloud.transform(reg_p2p.transformation)
    dist = numpy.asarray(cloud.compute_point_cloud_distance(clouds[0]))
    if i+1==1:
        dist *= 1.6
    if i+1==2:
        dist *= 1.4
    distances.append(dist)
    agg_distances.extend(dist)
    avg = numpy.average(dist)
    stdev = numpy.std(dist)
    logger.info(cloud_names[i+1]+". Max: {0}. Average: {1}".format(avg+2*stdev, numpy.average(dist)))



from matplotlib import cm
max_dist = numpy.max(agg_distances)
for i,dist in enumerate(distances):
    dist /= max_dist
    colormap = cm.jet( dist )
    clouds[i+1].colors = open3d.utility.Vector3dVector( colormap[:,0:3] )
    clouds[i+1].translate( numpy.array([-(i+1)*1.0, 0.0, 0.0]), relative=True)

import matplotlib.pyplot as plt
img = plt.imshow(numpy.array([agg_distances]), cmap="jet")
img.set_visible(False)
colorbar = plt.colorbar(orientation="horizontal")
# plt.show()
open3d.visualization.draw_geometries( clouds[1:] )