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
logger = logging.getLogger("rosout")


class SimCamera:
    def __init__(self, inspection_bot, part_stl_path=None, num_sample_points=20000):
        self.transformer = tf.TransformListener(True, rospy.Duration(10.0))
        self.noise_stddev = numpy.array([ 0.001, 0.001, 0.004 ]) # xyz error in m
        self.latest_cloud = None
        self.ros_cloud = None
        self.default_mesh = None
        self.part_stl_path = part_stl_path
        self.num_sample_points = num_sample_points
        self.inspection_bot = inspection_bot
        if part_stl_path:
            logger.info("Reading stl")
            self.default_mesh = open3d.io.read_triangle_mesh(self.part_stl_path)
            logger.info("Stl read. Generating PointCloud")
            transform = rosparam.get_param("/stl_params/transform")
            self.default_mesh = self.default_mesh.translate(transform[0:3])
            R = self.default_mesh.get_rotation_matrix_from_xyz((transform[0], transform[1], transform[2]))
            # self.default_mesh = self.default_mesh.rotate(R, center=(0, 0, 0))
            self.default_cloud = self.default_mesh.sample_points_poisson_disk(number_of_points=num_sample_points)
            logger.info("PointCloud Generated")
            self.stl_kdtree = open3d.geometry.KDTreeFlann(self.default_cloud)

    def publish_cloud(self):
        th = threading.Thread(target=self.start_publisher)
        th.start()

    def start_publisher(self):
        cloud = self.default_cloud
        stl_cloud_pub = rospy.Publisher("stl_cloud", PointCloud2, queue_size=1)
        visible_cloud_pub = rospy.Publisher("visible_cloud", PointCloud2, queue_size=1)
        # -- Convert open3d_cloud to ros_cloud, and publish. Until the subscribe receives it.
        while not cloud.is_empty() and not rospy.is_shutdown():
            (visible_cloud,_) = self.capture_point_cloud()
            self.ros_cloud = convertCloudFromOpen3dToRos(cloud, frame_id="base")
            stl_cloud_pub.publish(self.ros_cloud)
            self.visible_ros_cloud = convertCloudFromOpen3dToRos(visible_cloud, frame_id="base")
            visible_cloud_pub.publish(self.visible_ros_cloud)
            rospy.sleep(0.1)
        
    def capture_point_cloud(self, fit_kdtree=False):
        visible_cloud = open3d.geometry.PointCloud()
        if self.part_stl_path:
            base_T_tool0 = self.inspection_bot.get_current_forward_kinematics()
            tool0_T_camera_vec = self.transformer.lookupTransform("tool0", "camera_depth_frame", rospy.Time(0))
            tool0_T_camera = quaternion_matrix( [tool0_T_camera_vec[1][0], tool0_T_camera_vec[1][1],
                                                tool0_T_camera_vec[1][2], tool0_T_camera_vec[1][3]] )
            tool0_T_camera[0:3,3] = tool0_T_camera_vec[0]
            base_T_camera = numpy.matmul(base_T_tool0,tool0_T_camera)
            # Find the neighbors on the part within camera view
            [k, idx, distance] = self.stl_kdtree.search_radius_vector_3d( base_T_camera[0:3,3], radius=0.3 )
            if idx:
                stl_points = numpy.asarray(self.default_cloud.points)
                knn_points = stl_points[idx,:]
                knn_point_vectors = numpy.subtract( knn_points, base_T_camera[0:3,3])
                knn_point_vectors /= numpy.linalg.norm(knn_point_vectors,axis=1)[:,None]
                visible_points = knn_points[numpy.where(numpy.dot( knn_point_vectors, base_T_camera[0:3,0] )>=0.9)[0],:]
                visible_points = numpy.vstack(( visible_points,base_T_camera[0:3,3] ))
                if visible_points.shape[0]>0:
                    # visible_cloud.points = open3d.utility.Vector3dVector( visible_points + 
                    #                         numpy.random.normal(0, 1.0, (visible_points.shape[0],3))*self.noise_stddev )
                    visible_cloud.points = open3d.utility.Vector3dVector( visible_points )
                    visible_cloud.estimate_normals()
                    visible_cloud.orient_normals_towards_camera_location(camera_location=base_T_camera[0:3,3])
        return (visible_cloud,base_T_camera)
            
    def get_vision_coordinates(self):
        (visible_cloud, base_T_camera) = self.capture_point_cloud()
        camera_T_tool = numpy.linalg.inv(base_T_camera)
        if not visible_cloud.is_empty() and not rospy.is_shutdown():
            # Get points and normals in the base frame
            points = numpy.asarray(visible_cloud.points)
            normals = numpy.asarray(visible_cloud.normals)
            # Transform everything to camera_depth_frame
            transformed_points = numpy.matmul(camera_T_tool[0:3,0:3],points.T).T + camera_T_tool[0:3,3]
            transformed_normals = numpy.matmul(camera_T_tool[0:3,0:3],normals.T).T

            
        