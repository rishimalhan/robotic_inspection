#!/usr/bin/env python3

import open3d
import numpy
import rosparam
import logging
import threading
import rospy
import tf
import sys
import copy
from sensor_msgs.msg import PointCloud2
from tf.transformations import quaternion_matrix
from utilities.open3d_and_ros import convertCloudFromOpen3dToRos
from utilities.voxel_grid import VoxelGrid
from system.perception_utils import (
    get_heatmap
)
logger = logging.getLogger("rosout")

class CameraModel:
    def __init__(self):
        self.z_sigma = 0.0 # Error 0.0 +- 3*sigma in meters
        self.accuracy_sigma = 0.003
        self.clearing_distance = 0.15
        self.max_distance = 0.3

    def predict(self,vision_parameters):
        # We multiply error by z_sigma to bring it to max 1 cm. Error is max 1.0 before that
        return (numpy.mean( numpy.column_stack(
                    (31.1*numpy.power(vision_parameters[:,0],2) - 9.33*vision_parameters[:,0] + 1.0,
                    0.3*numpy.exp(4.01*vision_parameters[:,1]),
                    0.3*numpy.exp(0.38*vision_parameters[:,2]))
                ),
                axis=1 ), numpy.ones(vision_parameters.shape[0],) * self.accuracy_sigma )

class SimCamera:
    def __init__(self, inspection_bot, part_stl_path=None, num_sample_points=20000):
        self.camera_model = CameraModel()
        self.overlay_error_map = True
        self.transformer = tf.TransformListener(True, rospy.Duration(10.0))
        self.latest_cloud = None
        self.default_mesh = None
        self.part_stl_path = part_stl_path
        self.num_sample_points = num_sample_points
        self.inspection_bot = inspection_bot
        self.empty_cloud = open3d.geometry.PointCloud()
        if part_stl_path:
            logger.info("Reading stl. Path: {0}".format(self.part_stl_path))
            self.default_mesh = open3d.io.read_triangle_mesh(self.part_stl_path)
            logger.info("Stl read. Generating PointCloud")
            transform = rosparam.get_param("/stl_params/transform")
            self.default_mesh = self.default_mesh.translate(transform[0:3])
            R = self.default_mesh.get_rotation_matrix_from_xyz((transform[3], transform[4], transform[5]))
            self.default_mesh = self.default_mesh.rotate(R, center=(0, 0, 0))
            self.stl_cloud = self.default_mesh.sample_points_poisson_disk(number_of_points=num_sample_points)
            bottom_less_indices = numpy.where( numpy.asarray(self.stl_cloud.points)[:,2] > transform[2]-0.08 )[0]
            self.stl_cloud = self.stl_cloud.select_by_index(bottom_less_indices)
            self.stl_cloud.estimate_normals()
            self.stl_cloud.normalize_normals()
            surface_indices = numpy.where( numpy.asarray(self.stl_cloud.normals)[:,2] > 0.4 )[0]
            self.stl_cloud = self.stl_cloud.select_by_index(surface_indices)
            logger.info("PointCloud Generated")
            self.stl_kdtree = open3d.geometry.KDTreeFlann(self.stl_cloud)
        self.max_bounds = numpy.hstack(( self.stl_cloud.get_max_bound(), [0.8,0.8,0.8] ))
        self.min_bounds = numpy.hstack(( self.stl_cloud.get_min_bound(), [-0.8,-0.8,-0.8] ))
        self.max_bounds[2] += 0.5
        self.voxel_grid = VoxelGrid(numpy.asarray(self.stl_cloud.points), [self.min_bounds, self.max_bounds])
    
    def publish_cloud(self):
        self.stl_cloud_pub = rospy.Publisher("stl_cloud", PointCloud2, queue_size=1)
        self.visible_cloud_pub = rospy.Publisher("visible_cloud", PointCloud2, queue_size=1)
        th = threading.Thread(target=self.start_publisher)
        th.start()

    def start_publisher(self):
        # -- Convert open3d_cloud to ros_cloud, and publish. Until the subscribe receives it.
        while not self.stl_cloud.is_empty() and not rospy.is_shutdown():
            self.stl_cloud_pub.publish(convertCloudFromOpen3dToRos(self.stl_cloud, frame_id="base"))
            (visible_cloud, base_T_camera) = self.capture_point_cloud()
            if not visible_cloud.is_empty():
                if self.overlay_error_map:
                    (heatmap,stddev) = get_heatmap(visible_cloud, base_T_camera, self.camera_model)
                    from matplotlib import cm
                    try:
                        colormap = cm.jet( numpy.subtract(heatmap,numpy.min(heatmap))/(numpy.max(heatmap-numpy.min(heatmap))))
                        visible_cloud.colors = open3d.utility.Vector3dVector( colormap[:,0:3] )
                    except:
                        logger.warn("Heat map failed to generate.")
                        visible_cloud.clear()
                self.visible_cloud_pub.publish(convertCloudFromOpen3dToRos(visible_cloud, frame_id="base"))
            rospy.sleep(0.1)

    def get_current_transform(self):
        base_T_tool0 = self.inspection_bot.get_current_forward_kinematics()
        tool0_T_camera_vec = self.transformer.lookupTransform("tool0", "camera_depth_frame", rospy.Time(0))
        tool0_T_camera = quaternion_matrix( [tool0_T_camera_vec[1][0], tool0_T_camera_vec[1][1],
                                            tool0_T_camera_vec[1][2], tool0_T_camera_vec[1][3]] )
        tool0_T_camera[0:3,3] = tool0_T_camera_vec[0]
        return numpy.matmul(base_T_tool0,tool0_T_camera)

    def capture_point_cloud(self, base_T_camera=None):
        if self.part_stl_path:
            visible_cloud = open3d.geometry.PointCloud()
            if base_T_camera is None:
                base_T_camera = self.get_current_transform()
            # Find the neighbors on the part within camera view
            [k, idx, distance] = self.stl_kdtree.search_radius_vector_3d( base_T_camera[0:3,3], radius=0.3 )
            if len(idx)>0:
                stl_points = numpy.asarray(self.stl_cloud.points)
                knn_points = stl_points[idx,:]
                knn_point_vectors = numpy.subtract( knn_points, base_T_camera[0:3,3])
                knn_point_vectors /= numpy.linalg.norm(knn_point_vectors,axis=1)[:,None]
                visible_points = knn_points[numpy.where(numpy.dot( knn_point_vectors, base_T_camera[0:3,0] )>=0.9)[0],:]
                if visible_points.shape[0]>0:
                    visible_points[:,2] += numpy.random.normal(0.0, 1.0, 
                                            (visible_points.shape[0],))*self.camera_model.z_sigma
                    visible_cloud.points = open3d.utility.Vector3dVector( visible_points )
                    visible_cloud.estimate_normals()
                    visible_cloud.orient_normals_towards_camera_location(camera_location=base_T_camera[0:3,3])
                    visible_cloud.normalize_normals()
                    visible_cloud.colors = open3d.utility.Vector3dVector( 
                                            numpy.ones(visible_points.shape)*[0.447,0.62,0.811])
                return (visible_cloud,base_T_camera)
        return (self.empty_cloud,None)

            
        