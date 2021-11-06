#!/usr/bin/env python3

import open3d
import numpy
import rosparam
import logging
import threading
import rospy
from system.planning_utils import matrix_to_state
import tf
import sys
import copy
from sensor_msgs.msg import PointCloud2
from tf.transformations import quaternion_matrix

from utilities.open3d_and_ros import (
    convertCloudFromOpen3dToRos,
    convertCloudFromRosToOpen3d
)
from utilities.voxel_grid import VoxelGrid
from system.planning_utils import (
    tf_to_state,
    state_to_pose,
    tool0_from_camera,
)
from camera.camera_localization import Localizer
logger = logging.getLogger("rosout")

class Camera:
    def __init__(self, inspection_bot, transformer, camera_properties):
        self.camera_properties = camera_properties
        self.inspection_bot = inspection_bot
        self.transformer = transformer
        self.bbox = None
        self.camera_frame = camera_properties.get("camera_frame")
        self.localizer_tf = tf_to_state(self.transformer.lookupTransform("tool0", self.camera_frame, rospy.Time(0)))
        filters = camera_properties.get("filters")
        if filters is not None:
            if filters.get("bbox"):
                if len(filters["bbox"])!=0:
                    self.bbox = filters["bbox"]
    
    def get_current_transform(self):
        base_T_tool0 = self.inspection_bot.get_current_forward_kinematics()
        tool0_T_camera_vec = self.transformer.lookupTransform("tool0", self.camera_frame, rospy.Time(0))
        tool0_T_camera = quaternion_matrix( [tool0_T_camera_vec[1][0], tool0_T_camera_vec[1][1],
                                            tool0_T_camera_vec[1][2], tool0_T_camera_vec[1][3]] )
        tool0_T_camera[0:3,3] = tool0_T_camera_vec[0]
        return (numpy.matmul(base_T_tool0,tool0_T_camera), base_T_tool0, tool0_T_camera)

    def trigger_camera(self):
        # Get the cloud with respect to the base frame
        ros_cloud = rospy.wait_for_message("/camera/depth/color/points",
                                                    PointCloud2,timeout=None)
        open3d_cloud = convertCloudFromRosToOpen3d(ros_cloud)
        (transform,_,_) = self.get_current_transform()
        return (open3d_cloud, transform)

    def localize(self):
        self.plate_tf = tf_to_state(self.transformer.lookupTransform("base", "fiducial_7", rospy.Time(0)))
        # Generate points around the aruco point to execute robot motions
        plate_position = self.plate_tf[0:3] + numpy.array([0,0,0.3])
        (base_T_camera,_,_) = self.get_current_transform()
        camera_orientation = matrix_to_state(base_T_camera)[3:6]
        frames = [numpy.hstack(( plate_position,camera_orientation ))]
        # Randomly sample positions
        for sample in numpy.random.uniform(low=-1, high=1, size=(20,6)):
            frames.append(frames[0] + numpy.multiply(sample,numpy.array([0.1,0.1,0.0,0.3,0.3,0.3])))

        # Determine the bounding box
        # bbox = open3d.geometry.AxisAlignedBoundingBox()
        # localizer_bbox = numpy.array(self.camera_properties.get("filters").get("localizer_bbox"))
        # bbox.min_bound = localizer_bbox[0:3]
        # bbox.max_bound = localizer_bbox[3:6]

        clouds = []
        transforms = []
        logger.info("Moving robot to localizer frames and capturing data")
        for frame in frames:
            self.inspection_bot.execute_cartesian_path([state_to_pose(tool0_from_camera(frame,self.transformer))])
            (base_T_camera,base_T_tool0,_) = self.get_current_transform()
            transforms.append(base_T_tool0)
            ros_cloud = rospy.wait_for_message("/camera/depth/color/points",
                                                    PointCloud2,timeout=None)
            open3d_cloud = convertCloudFromRosToOpen3d(ros_cloud).transform(base_T_camera)
            # open3d_cloud = open3d_cloud.crop(bbox)
            open3d_cloud = open3d_cloud.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)[0]
            open3d_cloud = open3d_cloud.voxel_down_sample(voxel_size=0.01)
            open3d_cloud = open3d_cloud.transform(numpy.linalg.inv(base_T_camera))
            clouds.append(open3d_cloud)
        localizer = Localizer(clouds,transforms,self.localizer_tf)
        self.localizer_tf = localizer.localize()
        return self.localizer_tf
        

    def publish_cloud(self):
        pass

    def start_publisher(self):
        pass

    def capture_point_cloud(self, base_T_camera=None):
        pass

            
        