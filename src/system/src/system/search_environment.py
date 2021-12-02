from numpy.lib.financial import rate
from planning_utils import (
    generate_state_space,
    update_cloud,
    state_to_matrix,
    state_to_pose,
    tool0_from_camera
)

from scipy.spatial.kdtree import KDTree
import copy
import logging
import numpy
import random
import math
import sys
from sensor_msgs.msg import PointCloud2
import rosparam
from geometry_msgs.msg import PoseStamped
from utilities.filesystem_utils import (
    get_pkg_path,
)
logger = logging.getLogger("rosout")

import rospy
from os.path import exists
from sensor_msgs.msg import PointCloud2
from utilities.open3d_and_ros import (
    convertCloudFromOpen3dToRos,
    convertCloudFromRosToOpen3d
)

class InspectionEnv:
    def __init__(self, inspection_bot, camera, flags):
        self.coverage_threshold = 0.98
        check_ik = True
        self.inspection_bot = inspection_bot
        self.camera = camera
        self.current_config = self.inspection_bot.move_group.get_current_state().joint_state.position
        path = get_pkg_path("system")
        state_space_path = path + "/database/" + rosparam.get_param("/stl_params/name") + "/state_space.csv"
        if exists(state_space_path):
            self.state_space = numpy.loadtxt(state_space_path,delimiter=",")
        else:
            self.state_space = generate_state_space(self.camera.stl_cloud, self.camera.camera_home)
            if check_ik:
                # Clear out the unreachable states
                logger.info("State space created. Number of Points: {0}. Removing invalid points.".format(self.state_space.shape[0]))
                invalid_indices = []
                for i,state in enumerate(self.state_space):
                    tool0_pose = state_to_pose( tool0_from_camera(state,camera.transformer) )
                    ik_pose = PoseStamped()
                    ik_pose.header.frame_id = "base_link"
                    ik_pose.pose = tool0_pose
                    response = self.inspection_bot.get_ik.get_ik(ik_pose)
                    if response.error_code.val==1:
                        if not self.check_ik_validity(response.solution.joint_state.position):
                            invalid_indices.append(i)
                    else:
                        invalid_indices.append(i)
                self.state_space = numpy.delete(self.state_space,invalid_indices,axis=0)
            numpy.savetxt(state_space_path,self.state_space,delimiter=",")
        self.ss_tree = KDTree(self.state_space)
        self.step = 10
        logger.info("Number of states after filtering. {0}".format(self.state_space.shape[0]))
        self.state_zero = numpy.array(self.camera.camera_home)
        (cloud,_) = self.camera.get_simulated_cloud(base_T_camera=state_to_matrix(self.state_zero))
        self.camera.voxel_grid_sim.update_grid(cloud)
        logger.info("Inspection Environment Initialized. Initial coverage: {0}".format(self.camera.voxel_grid_sim.get_coverage()))
        # self.visible_cloud_pub = rospy.Publisher("visible_cloud", PointCloud2, queue_size=10)
    
    def get_children(self, state, query_number):
        (distance,indices) = self.ss_tree.query( state,query_number )
        valid_indices = []
        for idx in indices[-self.step:]:
            if idx not in self.visited_states:
                valid_indices.append(idx)
        return valid_indices
    
    def take_action(self, children, parent):
        if len(children)==0:
            return None
        child_state = None
        rewards = numpy.array([])
        initial_observations = self.camera.voxel_grid_sim.number_observations
        initial_coverage = self.camera.voxel_grid_sim.get_coverage()
        for child in children:
            # Get the observations that can be made from that specific view point
            if child in self.stored_obs:
                new_obs = self.stored_obs[child]
            else:
                (cloud,_) = self.camera.get_simulated_cloud(base_T_camera=state_to_matrix(self.state_space[child]))
                # print(cloud)
                self.camera.voxel_grid_sim.update_grid(cloud,update_observations=False) # Update to get only the new observations
                new_obs = self.camera.voxel_grid_sim.new_obs
                self.stored_obs[child] = new_obs
            coverage = self.camera.voxel_grid_sim.get_coverage(observations=(initial_observations + new_obs)) - initial_coverage
            rate_gain = coverage / numpy.linalg.norm( self.state_space[child]-parent )
            rewards = numpy.append( rewards, rate_gain )
        if numpy.any(rewards>0):
            child = children[numpy.argmax(rewards)]
            child_state = self.state_space[child]
            (cloud,_) = self.camera.get_simulated_cloud(base_T_camera=state_to_matrix(self.state_space[child]))
            self.camera.voxel_grid_sim.update_grid(cloud)
            self.visited_states.append(child)
        if numpy.any(rewards<0):
            logger.warn("Rewards are negative. Potential bug")
        return child_state

    def greedy_search(self):
        self.visited_states = []
        self.stored_obs = {}
        path = [self.state_zero]
        stopping_condition = False
        logger.info("Planning using greedy search")
        while not stopping_condition:
            query_number = self.step
            child_state = None
            while child_state is None:
                children = self.get_children(path[-1], query_number)
                child_state = self.take_action(children, path[-1])
                if child_state is None and query_number != self.state_space.shape[0]:
                    query_number += self.step
                    if query_number > self.state_space.shape[0]:
                        query_number = self.state_space.shape[0]
                else:
                    break
                        
            if child_state is not None:
                coverage = self.camera.voxel_grid_sim.get_score()
                path.append(child_state)
                if coverage > self.coverage_threshold:
                    stopping_condition = True
                logger.info("Coverage: {0:.2f}. States: Visited: {1}. Remaining: {2}".format(coverage, len(self.visited_states),
                                                            self.state_space.shape[0]-len(self.visited_states)))
            else:
                stopping_condition = True
        if coverage > self.coverage_threshold:
            logger.info("Solution found with {0} points and {1} coverage"
                            .format(len(path),self.camera.voxel_grid_sim.get_score()))
        else:
            logger.info("Solution found without complete coverage. Remaining States")
            logger.info("Coverage: {0:.2f}. States: Visited: {1}. Remaining: {2}. QueryNumber: {3}".format(coverage, len(self.visited_states),
                                            self.state_space.shape[0]-len(self.visited_states), query_number))
        return numpy.array(path)
    
    def check_ik_validity(self,joints):
        inf_norm = numpy.linalg.norm(numpy.array(joints) - numpy.array(self.current_config), ord=numpy.inf)
        if inf_norm < 2.0:
            return True
        else:
            False

    def get_joint_path(self,waypoints):
        logger.info("Generating Cspace Path")
        number_skips = 0
        max_skips = int(len(waypoints)*0.1)
        joint_states = []
        for i,pose in enumerate(waypoints):
            ik_pose = PoseStamped()
            ik_pose.header.frame_id = "base_link"
            ik_pose.pose = pose
            response = self.inspection_bot.get_ik.get_ik(ik_pose)
            if response.error_code.val==1:
                rob_joint_state = response.solution.joint_state
                if self.check_ik_validity(rob_joint_state.position):
                    joint_states.append( rob_joint_state )
                else:
                    number_skips += 1
                    if number_skips > max_skips:
                        return None
            else:
                number_skips += 1
                if number_skips > max_skips:
                    return None
        return joint_states

    def get_executable_path(self, path, increase_density=False):
        flange_path = numpy.array([tool0_from_camera(pose, self.camera.transformer) for pose in path])
        if increase_density:
            # Increase path density
            exec_path = []
            for i in range(1,flange_path.shape[0]):
                exec_path.append( flange_path[i-1] )
                dist = numpy.linalg.norm( flange_path[i]-flange_path[i-1] )
                if dist < 0.34:
                    continue
                number_points = math.ceil(dist / 0.17)
                dp = (flange_path[i]-flange_path[i-1]) / number_points
                for j in range(1,number_points-1):
                    exec_path.append( flange_path[i-1]+dp*j )
            exec_path = numpy.array(exec_path)
            logger.info("Number of points generated post density increase: %d",exec_path.shape[0])
        else:
            exec_path = flange_path.tolist()

        waypoints = [state_to_pose(pose) for pose in exec_path]
        for attempts in range(10):
            joint_states = self.get_joint_path(waypoints)
            if joint_states is not None:
                break
        return (waypoints, joint_states)