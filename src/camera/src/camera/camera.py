#!/usr/bin/env python3

from telnetlib import IP
from numpy.lib.financial import ipmt
import open3d
import numpy
import rosparam
import logging
import threading
import rospy
from os.path import exists
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
from system.perception_utils import (
    get_heatmap,
    get_vision_coordinates
)
logger = logging.getLogger("rosout")


class Camera:
    def __init__(self, inspection_bot, transformer, camera_properties, flags):
        if "sim" in flags:
            self.simulated_camera = True
        else:
            self.simulated_camera = False
        self.empty_cloud = open3d.geometry.PointCloud()
        self.camera_properties = camera_properties
        self.inspection_bot = inspection_bot
        self.transformer = transformer
        self.heatmap_bounds = numpy.array([ [0.27,0,0],[0.39,0.2,1.0] ]) # Conservative
        # self.heatmap_bounds = numpy.array([ [0,0,0],[10,10,3.14] ]) # Bad
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

        if "camera_data" in flags:
            self.get_camera_parameter_data()

        if "localize" in flags:
            self.localizer_tf = tf_to_state(self.transformer.lookupTransform("tool0", self.camera_frame, rospy.Time(0)))
            self.localize()
        else:
            self.localizer_tf = numpy.loadtxt(self.tf_path,delimiter=",")
        logger.info("Localizer transform: {0}.".format(self.localizer_tf))
        
        # Scan the part for coarse approximation
        part_tf_path = path + "/database/" + rosparam.get_param("/stl_params/name") + "/part_tf.csv"
        if "part_tf" in flags:
            stl_param = rospy.get_param("/stl_params")
            part_frame = numpy.array(stl_param.get("aruco_frame"))
            frame_name = stl_param.get("aruco_frame_name")
            self.inspection_bot.execute_cartesian_path([state_to_pose(part_frame)])
            part_transform = []
            for i in range(100):
                part_transform.append(tf_to_state(self.transformer.lookupTransform("base", frame_name, rospy.Time(0))))
            part_transform = numpy.average(part_transform,axis=0)
            # Hard coding!
            part_transform[3:6] = [0,0,1.57]
            part_transform = state_to_matrix( part_transform )
            numpy.savetxt(part_tf_path,matrix_to_state(part_transform),delimiter=",")
            part_transform[0:3,3] += rosparam.get_param("/stl_params/compensation")
        else:
            logger.info("Reading the stored part transform")
            transform = numpy.loadtxt(part_tf_path,delimiter=",")
            part_transform = state_to_matrix( transform )
            part_transform[0:3,3] += rosparam.get_param("/stl_params/compensation")
            
        ref_path = path + "/database/" + rosparam.get_param("/stl_params/name") + "/reference.ply"
        if exists(ref_path):
            logger.info("Reference cloud found. Reading existing file.")
            self.stl_cloud = open3d.io.read_point_cloud(ref_path)
        else:
            stl_path = path + rosparam.get_param("/stl_params/directory_path") + \
                                "/" + rosparam.get_param("/stl_params/name") + ".stl"
            logger.info("Reading stl. Path: {0}".format(stl_path))
            self.mesh = open3d.io.read_triangle_mesh(stl_path)
            # Point cloud of STL surface only
            logger.info("Stl read. Generating PointCloud from stl")
            self.stl_cloud = self.mesh.sample_points_poisson_disk(number_of_points=100000)
            self.stl_cloud = self.stl_cloud.voxel_down_sample(voxel_size=0.001)
            logger.info("Applying transform: \n{0}.".format(part_transform))
            self.stl_cloud = self.stl_cloud.transform(part_transform)
            filters = rospy.get_param("/stl_params").get("filters")
            if filters is not None:
                logger.info("Applying filters")
                dot_products = numpy.asarray(self.stl_cloud.normals)[:,2]
                dot_products[numpy.where(dot_products>1)[0]] = 1
                dot_products[numpy.where(dot_products<-1)[0]] = -1    
                if filters.get("max_angle_with_normal") is not None:
                    logger.info("Normal orientation filter")
                    surface_indices = numpy.where( numpy.arccos(dot_products) < filters.get("max_angle_with_normal") )[0]
                    self.stl_cloud = self.stl_cloud.select_by_index(surface_indices)
                if filters.get("truncate_below_z") is not None:
                    logger.info("Allowed Z values filter")
                    surface_indices = numpy.where( numpy.asarray(self.stl_cloud.points)[:,2] >= part_transform[2,3]+filters.get("truncate_below_z") )[0]
                    self.stl_cloud = self.stl_cloud.select_by_index(surface_indices)
            logger.info("Writing reference pointcloud to file")
            open3d.io.write_point_cloud(ref_path, self.stl_cloud)
            
        # Cloud to be used for simulations
        self.sim_cloud = self.stl_cloud.voxel_down_sample(voxel_size=0.003)
        self.voxel_grid_sim = VoxelGrid(numpy.asarray(self.sim_cloud.points), sim=True)
        self.voxel_grid = VoxelGrid(numpy.asarray(self.stl_cloud.points),create_from_bounds=True)
        self.stl_kdtree = open3d.geometry.KDTreeFlann(self.stl_cloud)
        (base_T_camera,_,_) = self.get_current_transform()
        current_orientation = matrix_to_state(base_T_camera)[3:6]
        self.camera_home = numpy.hstack(( matrix_to_state(part_transform)[0:3] + numpy.array([0,0,0.3]),
                                        current_orientation
                                        ))
        self.publish_cloud()        

    def get_camera_parameter_data(self):
        self.localizer_tf = numpy.loadtxt(self.tf_path,delimiter=",")
        self.plate_tf = []
        for i in range(100):
            self.plate_tf.append(tf_to_state(self.transformer.lookupTransform("base", "fiducial_6", rospy.Time(0))))
        self.plate_tf = numpy.average(self.plate_tf,axis=0)
        # Generate points around the aruco point to execute robot motions
        plate_position = self.plate_tf[0:3] + numpy.array([0,0,0.35])
        (base_T_camera,_,_) = self.get_current_transform()
        camera_orientation = matrix_to_state(base_T_camera)[3:6]
        frames = [numpy.hstack(( plate_position,camera_orientation ))]
        # Randomly sample positions
        for sample in numpy.random.uniform(low=-1, high=1, size=(40,6)):
            frames.append(frames[0] + numpy.multiply(sample,numpy.array([0.1,0.1,0.1,0.4,0.4,0.4])))

        data = []
        clouds = []
        logger.info("Moving robot to localizer frames and capturing data")
        for i,frame in enumerate(frames):
            self.inspection_bot.execute_cartesian_path([state_to_pose(tool0_from_camera(frame,self.transformer))])
            (open3d_cloud, base_T_camera) = self.trigger_camera()
            open3d_cloud = open3d_cloud.voxel_down_sample(voxel_size=0.001)
            open3d_cloud = open3d_cloud.remove_statistical_outlier(nb_neighbors=10,std_ratio=5.0)[0]
            if i==0:
                first_cloud = open3d_cloud
                clouds.append(first_cloud)
                points = numpy.asarray(first_cloud.points)
                points = numpy.append(points,points+[0,0,0.02],axis=0)
                points = numpy.append(points,points-[0,0,0.02],axis=0)
                cloud = open3d.geometry.PointCloud()
                cloud.points = open3d.utility.Vector3dVector(points)
                bbox = cloud.get_axis_aligned_bounding_box()
            if i>0:
                open3d_cloud = open3d_cloud.crop(bbox)
                clouds.append(open3d_cloud)
                if open3d_cloud.is_empty():
                    continue
                open3d_cloud.estimate_normals()
                open3d_cloud.orient_normals_towards_camera_location(camera_location=base_T_camera[0:3,3])
                open3d_cloud.normalize_normals()
                distances = numpy.asarray(open3d_cloud.compute_point_cloud_distance(first_cloud))
                parameters = get_vision_coordinates(open3d_cloud, base_T_camera)
                data.extend(numpy.column_stack(( parameters, distances )))
        # open3d.visualization.draw_geometries(clouds)
        path = get_pkg_path("system")
        numpy.savetxt(path+"/database/camera_data.csv",data,delimiter=",")

    # def get_error_map(self):
    #     (open3d_cloud, transform) = self.trigger_camera()
    #     open3d_cloud.estimate_normals()
    #     open3d_cloud.orient_normals_towards_camera_location(camera_location=transform[0:3,3])
    #     open3d_cloud.normalize_normals()
    #     (heatmap,_) = get_heatmap(open3d_cloud, transform, None, vision_parameters=None)
    #     for i in [0,1,2]:
    #         heatmap[:,i] = numpy.subtract(heatmap[:,i],numpy.min(heatmap[:,i])) / (numpy.max(heatmap[:,i]-numpy.min(heatmap[:,i])))
    #     color_code = numpy.average( heatmap, axis=1 )
    #     val = numpy.average(color_code)
    #     open3d_cloud = open3d_cloud.select_by_index( numpy.where(color_code<val)[0] )
    #     color_code = color_code[numpy.where(color_code<val)[0]]
    #     from matplotlib import cm
    #     colormap = cm.jet( color_code*2 )
    #     open3d_cloud.colors = open3d.utility.Vector3dVector( colormap[:,0:3] )
    #     self.visible_cloud_pub.publish(convertCloudFromOpen3dToRos(open3d_cloud, frame_id="base"))
        
    def get_current_transform(self):
        base_T_tool0 = self.inspection_bot.get_current_forward_kinematics()
        tool0_T_camera = state_to_matrix(self.localizer_tf)
        return (numpy.matmul(base_T_tool0,tool0_T_camera), base_T_tool0, tool0_T_camera)

    def trigger_camera(self):
        if self.simulated_camera:
            return self.get_simulated_cloud()
        # Get the cloud with respect to the base frame
        ros_cloud = rospy.wait_for_message("/camera/depth/color/points",
                                                    PointCloud2,timeout=None)
        (transform,_,_) = self.get_current_transform()
        open3d_cloud = convertCloudFromRosToOpen3d(ros_cloud)
        open3d_cloud = open3d_cloud.transform(transform)
        if open3d_cloud.is_empty():
            self.trigger_camera()
        return (open3d_cloud, transform)

    def filter_cloud(self,cloud,base_T_camera):
        cloud.estimate_normals()
        cloud.orient_normals_towards_camera_location(camera_location=base_T_camera[0:3,3])
        cloud.normalize_normals()
        (heatmap,_) = get_heatmap(cloud, base_T_camera, None, vision_parameters=None)
        # logger.info("Min: {0}".format(numpy.min(heatmap,axis=0)))
        # logger.info("Max: {0}\n".format(numpy.max(heatmap,axis=0)))
        indices = numpy.where( (heatmap[:,0]>self.heatmap_bounds[0,0]) &
                                (heatmap[:,1]>self.heatmap_bounds[0,1]) &
                                (heatmap[:,2]>self.heatmap_bounds[0,2]) &
                                (heatmap[:,0]<self.heatmap_bounds[1,0]) &
                                (heatmap[:,1]<self.heatmap_bounds[1,1]) &
                                (heatmap[:,2]<self.heatmap_bounds[1,2]) )[0]
        cloud = cloud.select_by_index(indices)
        return cloud

    def localize(self):
        logger.info("Intial value of camera to tool transform: {0}".format(self.localizer_tf))
        self.plate_tf = []
        for i in range(100):
            self.plate_tf.append(tf_to_state(self.transformer.lookupTransform("base", "fiducial_6", rospy.Time(0))))
        self.plate_tf = numpy.average(self.plate_tf,axis=0)
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
            open3d_cloud = self.admittance_func(open3d_cloud,base_T_camera)
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
        
        # Make the cube bounding box as the field of view
        # Bounding box to crop pointcloud that the camera sees wrt depth optical frame
        axbbox = open3d.geometry.AxisAlignedBoundingBox()
        axbbox.min_bound = numpy.array([ -0.05, -0.05, 0.25 ])
        axbbox.max_bound = numpy.array([ 0.05, 0.05, 0.45 ])
        fov = open3d.geometry.OrientedBoundingBox().create_from_axis_aligned_bounding_box(axbbox)
        if base_T_camera is None:
            (base_T_camera,_,_) = self.get_current_transform()
        fov.rotate(base_T_camera[0:3,0:3])
        fov.center = base_T_camera[0:3,3] + 0.3*base_T_camera[0:3,2]
        visible_cloud = self.sim_cloud.crop(fov)
        if not visible_cloud.is_empty():
            visible_cloud = self.filter_cloud(visible_cloud,base_T_camera)
        return (visible_cloud,base_T_camera)

    def publish_cloud(self):
        self.stl_cloud_pub = rospy.Publisher("stl_cloud", PointCloud2, queue_size=10)
        self.visible_cloud_pub = rospy.Publisher("visible_cloud", PointCloud2, queue_size=10)
        self.constructed_cloud = rospy.Publisher("constructed_cloud", PointCloud2, queue_size=10)
        th = threading.Thread(target=self.start_publisher)
        th.start()

    def start_publisher(self):
        while not rospy.is_shutdown():
            # -- Convert open3d_cloud to ros_cloud, and publish. Until the subscribe receives it.
            self.stl_cloud_pub.publish(convertCloudFromOpen3dToRos(self.stl_cloud, frame_id="base"))
            self.op_cloud = self.voxel_grid.get_cloud()
            self.constructed_cloud.publish(convertCloudFromOpen3dToRos(self.op_cloud, frame_id="base"))
            # (cloud,base_T_camera) = self.trigger_camera() # Cloud wrt robot base
            # self.visible_cloud_pub.publish(convertCloudFromOpen3dToRos(cloud, frame_id="base"))
            rospy.sleep(0.1)
    
    def construct_cloud(self):
        logger.info("Starting cloud construction")
        self.th = threading.Thread(target=self.update_cloud)
        self.th.start()

    def stop_construct(self):
        self.th.join()

    def update_cloud_once(self):
        (cloud,base_T_camera) = self.trigger_camera()
        if not cloud.is_empty():
            cloud = self.filter_cloud(cloud,base_T_camera)
            if not cloud.is_empty():
                self.voxel_grid.update_grid(cloud)

    def update_cloud(self):
        while not rospy.is_shutdown():
            (cloud,base_T_camera) = self.trigger_camera()
            # self.visible_cloud_pub.publish(convertCloudFromOpen3dToRos(cloud, frame_id="base"))
            if not cloud.is_empty():
                cloud = self.filter_cloud(cloud,base_T_camera)
                if not cloud.is_empty():
                    self.voxel_grid.update_grid(cloud)
            rospy.sleep(0.0001)            
        