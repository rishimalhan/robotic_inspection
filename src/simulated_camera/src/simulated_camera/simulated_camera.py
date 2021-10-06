#!/usr/bin/env python3

import open3d
import numpy
import rosparam
import logging
import threading
import rospy
from sensor_msgs.msg import PointCloud2
from utilities.open3d_and_ros import convertCloudFromOpen3dToRos
logger = logging.getLogger("rosout")


class SimCamera:
    def __init__(self, inspection_bot, part_stl_path=None, num_sample_points=20000):
        self.latest_cloud = None
        self.ros_cloud = None
        self.default_mesh = None
        self.visible_cloud = open3d.geometry.PointCloud()
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
            self.capture_point_cloud()
            self.ros_cloud = convertCloudFromOpen3dToRos(cloud, frame_id="base")
            stl_cloud_pub.publish(self.ros_cloud)
            if not self.visible_cloud.is_empty():
                self.visible_ros_cloud = convertCloudFromOpen3dToRos(self.visible_cloud, frame_id="base")
                visible_cloud_pub.publish(self.visible_ros_cloud)
            rospy.sleep(0.2)
        
    def capture_point_cloud(self, fit_kdtree=False):
        if self.part_stl_path:
            fk = self.inspection_bot.get_forward_kinematics()
            # Find the neighbors on the part within camera view
            [k, idx, _] = self.stl_kdtree.search_hybrid_vector_3d( fk[0:3,3],
                                        max_nn=int(self.num_sample_points/20), radius=0.05 )
            stl_points = numpy.asarray(self.default_cloud.points)
            visible_points = stl_points[idx,:]
            self.visible_cloud.points = open3d.utility.Vector3dVector( visible_points )
            

    def get_vision_coordinates(self, camera_transform):
        pass