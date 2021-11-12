#!/usr/bin/env python3

import open3d
import numpy
import rosparam
import logging
import threading
import rospy
from system.planning_utils import matrix_to_state
from system.planning_utils import tf_to_matrix
import tf
import sys
import copy
from sensor_msgs.msg import PointCloud2
from tf.transformations import quaternion_matrix
from utilities.filesystem_utils import (
    get_pkg_path,
)
from utilities.open3d_and_ros import (
    convertCloudFromOpen3dToRos,
    convertCloudFromRosToOpen3d
)
from utilities.voxel_grid import VoxelGrid
from system.planning_utils import (
    tf_to_state,
    state_to_pose,
    tool0_from_camera,
    state_to_matrix,
)
from camera.camera_localization import Localizer
logger = logging.getLogger("rosout")

class Camera:
    def __init__(self, inspection_bot, transformer, camera_properties, flags):
        self.camera_properties = camera_properties
        self.inspection_bot = inspection_bot
        self.transformer = transformer
        path = get_pkg_path("system")
        # Scan the workspace boundary
        self.bbox_path = path + "/database/bbox.csv"
        self.bbox = open3d.geometry.AxisAlignedBoundingBox()
        if "bbox" in flags:
            boundary_frames = numpy.array(self.camera_properties.get("aruco_frames"))[0:4]
            frame_names = numpy.array(self.camera_properties.get("frame_names"))
            boundary_points = []
            for i,frame in enumerate(boundary_frames):
                self.inspection_bot.execute_cartesian_path([state_to_pose(frame)],avoid_collisions=False)
                boundary_points.append(self.transformer.lookupTransform("base", frame_names[i], rospy.Time(0))[0])
                
            boundary_points = numpy.array(boundary_points)
            self.bbox.min_bound = numpy.min(boundary_points,axis=0) - numpy.array([0,0,0.05])
            self.bbox.max_bound = numpy.max(boundary_points,axis=0) + numpy.array([0,0,0.5])
            numpy.savetxt(self.bbox_path,numpy.vstack(( self.bbox.min_bound,self.bbox.max_bound )),delimiter=",")
        else:
            bbox = numpy.loadtxt(self.bbox_path,delimiter=",")
            self.bbox.min_bound = bbox[0]
            self.bbox.max_bound = bbox[1]
        logger.info("Workspace bounds determined wrt base frame. Min: {0}. Max: {1}".format(self.bbox.min_bound,self.bbox.max_bound))
        
        self.tf_path = path + "/database/tool0_T_camera.csv"
        self.robot_home = rospy.get_param("/robot_positions/home")
        self.inspection_bot.execute_cartesian_path([state_to_pose(self.robot_home)], avoid_collisions=True)
        self.camera_frame = camera_properties.get("camera_frame")
        if "localize" in flags:
            self.localizer_tf = tf_to_state(self.transformer.lookupTransform("tool0", self.camera_frame, rospy.Time(0)))
            self.localize()
        else:
            self.localizer_tf = numpy.loadtxt(self.tf_path,delimiter=",")
        logger.info("Localizer transform: {0}.".format(self.localizer_tf))
        
        # Scan the part for coarse approximation
        part_frame = numpy.array(self.camera_properties.get("aruco_frames"))[4]
        frame_name = numpy.array(self.camera_properties.get("frame_names"))[4]
        self.inspection_bot.execute_cartesian_path([state_to_pose(part_frame)])
        part_transform = tf_to_matrix(self.transformer.lookupTransform("base", frame_name, rospy.Time(0)))
        rosparam.set_param("/stl_params/transform", matrix_to_state(part_transform))
        stl_path = path + rosparam.get_param("/stl_params/directory_path") + \
                            "/" + rosparam.get_param("/stl_params/name") + ".stl"
        logger.info("Reading stl. Path: {0}".format(stl_path))
        self.mesh = open3d.io.read_triangle_mesh(stl_path)
        logger.info("Stl read. Generating PointCloud from stl")
        self.mesh = self.mesh.transform(part_transform)
        self.stl_cloud = self.mesh.sample_points_poisson_disk(number_of_points=10000)
        self.stl_cloud.estimate_normals()
        self.stl_cloud.normalize_normals()
        max_bounds = numpy.hstack(( self.stl_cloud.get_max_bound(), [0.8,0.8,0.8] ))
        min_bounds = numpy.hstack(( self.stl_cloud.get_min_bound(), [-0.8,-0.8,-0.8] ))
        self.voxel_grid = VoxelGrid(numpy.asarray(self.stl_cloud.points), [min_bounds, max_bounds])
        self.voxel_grid_sim = VoxelGrid(numpy.asarray(self.stl_cloud.points), [min_bounds, max_bounds])
        self.stl_kdtree = open3d.geometry.KDTreeFlann(self.stl_cloud)
        (base_T_camera,_,_) = self.get_current_transform()
        current_orientation = matrix_to_state(base_T_camera)[3:6]
        self.camera_home = numpy.hstack(( matrix_to_state(part_transform)[0:3] + numpy.array([0,0,0.3]),
                                        current_orientation
                                        ))
        
    def get_current_transform(self):
        base_T_tool0 = self.inspection_bot.get_current_forward_kinematics()
        tool0_T_camera = state_to_matrix(self.localizer_tf)
        return (numpy.matmul(base_T_tool0,tool0_T_camera), base_T_tool0, tool0_T_camera)

    def trigger_camera(self):
        # Get the cloud with respect to the base frame
        ros_cloud = rospy.wait_for_message("/camera/depth/color/points",
                                                    PointCloud2,timeout=None)
        open3d_cloud = convertCloudFromRosToOpen3d(ros_cloud)
        (transform,_,_) = self.get_current_transform()
        open3d_cloud = open3d_cloud.transform(transform)
        open3d_cloud = open3d_cloud.crop(self.bbox)
        points = numpy.asarray(open3d_cloud.points)
        distances = numpy.linalg.norm(points-transform[0:3,3],axis=1)
        open3d_cloud = open3d_cloud.select_by_index( numpy.where( distances < 0.35 and distances > 0.25 )[0] )
        return (open3d_cloud, transform)

    def localize(self):
        logger.info("Intial value of camera to tool transform: {0}".format(self.localizer_tf))
        self.plate_tf = tf_to_state(self.transformer.lookupTransform("base", "fiducial_6", rospy.Time(0)))
        # Generate points around the aruco point to execute robot motions
        plate_position = self.plate_tf[0:3] + numpy.array([0,0,0.3])
        (base_T_camera,_,_) = self.get_current_transform()
        camera_orientation = matrix_to_state(base_T_camera)[3:6]
        frames = [numpy.hstack(( plate_position,camera_orientation ))]
        # Randomly sample positions
        for sample in numpy.random.uniform(low=-1, high=1, size=(10,6)):
            frames.append(frames[0] + numpy.multiply(sample,numpy.array([0.2,0.2,0.0,0.2,0.2,0.2])))

        clouds = []
        transforms = []
        logger.info("Moving robot to localizer frames and capturing data")
        for frame in frames:
            self.inspection_bot.execute_cartesian_path([state_to_pose(tool0_from_camera(frame,self.transformer))])
            (open3d_cloud, base_T_camera) = self.trigger_camera()
            open3d_cloud = open3d_cloud.voxel_down_sample(voxel_size=0.01)
            open3d_cloud = open3d_cloud.remove_statistical_outlier(nb_neighbors=10,std_ratio=5.0)[0]
            open3d_cloud = open3d_cloud.transform(numpy.linalg.inv(base_T_camera))
            clouds.append(open3d_cloud)
            transforms.append(base_T_camera)
        localizer = Localizer(clouds,transforms,self.localizer_tf)
        # Correct the tf and store it in a file
        self.localizer_tf = localizer.localize()
        numpy.savetxt(self.tf_path,self.localizer_tf,delimiter=",")
        return self.localizer_tf
    
    def get_simulated_cloud(self, base_T_camera=None):
        visible_cloud = open3d.geometry.PointCloud()
        if base_T_camera is None:
            base_T_camera = self.get_current_transform()
        # Find the neighbors on the part within camera view
        [k, idx, distance] = self.stl_kdtree.search_radius_vector_3d( base_T_camera[0:3,3], radius=0.3 )
        if len(idx)>0:
            stl_points = numpy.asarray(self.stl_cloud.points)
            knn_points = stl_points[idx]
            knn_point_vectors = numpy.subtract( knn_points, base_T_camera[0:3,3])
            knn_point_vectors /= numpy.linalg.norm(knn_point_vectors,axis=1)[:,None]
            visible_points = knn_points[numpy.where(numpy.dot( knn_point_vectors, base_T_camera[0:3,2] )>=0.9)[0]]
            if visible_points.shape[0]>0:
                visible_cloud.points = open3d.utility.Vector3dVector( visible_points )
                visible_cloud.estimate_normals()
                visible_cloud.orient_normals_towards_camera_location(camera_location=base_T_camera[0:3,3])
                visible_cloud.normalize_normals()
                visible_cloud.colors = open3d.utility.Vector3dVector( 
                                        numpy.ones(visible_points.shape)*[0.447,0.62,0.811])
            return (visible_cloud,base_T_camera)
        return (self.empty_cloud,None)


    def publish_cloud(self):
        pass

    def start_publisher(self):
        pass
            
        