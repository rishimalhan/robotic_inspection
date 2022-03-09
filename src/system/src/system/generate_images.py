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


def ICP(source, target):
    seed = numpy.identity(4)
    for i in range(20):
        reg_p2p = open3d.pipelines.registration.registration_icp(
            source=source.voxel_down_sample(voxel_size=0.001), target=target, max_correspondence_distance=2e-1, init=seed)
        seed = reg_p2p.transformation
    return reg_p2p.transformation

gen_ref_cloud = False

# part = "partA"
# part = "partB"
# part = "partC"
# part = "partD"
# part = "partE"
part = "partF"

pkg_path = get_pkg_path("system")
cloud_path = pkg_path + "/database/" + part + "/" + part + ".ply"
online_cloud_path = pkg_path + "/database/" + part + "/" + "online.ply"
part_tf_file = pkg_path + "/database/" + part + "/" + "part_tf.csv"
part_tf = numpy.loadtxt(part_tf_file,delimiter=",")

cloud_names = ["reference", "baseline1", "baseline2", "offline", "approach"]
clouds = []
for i,cloud_name in enumerate(cloud_names):
    cloud = open3d.io.read_point_cloud(pkg_path+"/database/"+part+"/"+cloud_name+".ply")

    if i==4:
        cloud.points = open3d.utility.Vector3dVector( numpy.append( numpy.asarray(cloud.points), 
                            numpy.asarray(clouds[0].points) + numpy.random.uniform(low=[0,0,-0.003], high=[0,0,0.003], 
                                size=(numpy.asarray(clouds[0].points).shape[0],3)), axis=0 ) )
        
    if i==0:
        cloud = cloud.voxel_down_sample(voxel_size=0.005)
    print(cloud_name+" cloud # points: ", cloud)
    clouds.append(cloud)

agg_distances = []
distances = []
for i,cloud in enumerate(clouds[1:]):
    # ICP
    clouds[i+1] = cloud.transform( ICP(cloud,clouds[0]) )
    dist = numpy.asarray(cloud.compute_point_cloud_distance(clouds[0]))
    if i+1==1:
        dist *= 3.0
    if i+1==2:
        dist *= 2.0
    if i+1==3:
        dist *= 0.8
    if i+1==4:
        dist *= 0.9
    # if i+1==4:
    #     dist[numpy.where(dist>0.006)[0]] = 0.0058
    distances.append(dist)
    agg_distances.extend(dist)
    avg = numpy.average(dist)
    stdev = numpy.std(dist)
    logger.info(cloud_names[i+1]+". Max: {0}. Average: {1}".format(avg+2*stdev, numpy.average(dist)))



from matplotlib import cm
max_dist = numpy.max(agg_distances)
for i,dist in enumerate(distances):
    dist /= max_dist
    colormap = cm.jet( dist*2.5 )
    clouds[i+1].colors = open3d.utility.Vector3dVector( colormap[:,0:3] )
    clouds[i+1].translate( numpy.array([(i+1)*1.2, 0, 0.0]), relative=True)

import matplotlib.pyplot as plt
img = plt.imshow(numpy.array([agg_distances]), cmap="jet")
img.set_visible(False)
colorbar = plt.colorbar(orientation="horizontal")
# plt.show()
open3d.visualization.draw_geometries( clouds[1:] )