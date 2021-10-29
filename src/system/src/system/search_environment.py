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
import sys
logger = logging.getLogger("rosout")


class InspectionEnv:
    def __init__(self, sim_camera, camera_home_state):
        self.sim_camera = sim_camera
        self.state_space = generate_state_space(self.sim_camera.stl_cloud)
        self.ss_tree = KDTree(self.state_space)
        self.state_zero = numpy.array(camera_home_state)
        self.step = 20
        update_cloud([self.state_zero], self.sim_camera)
        logger.info("Inspection Environment Initialized. Number of states: %d", self.state_space.shape[0])

    def get_children(self, state, query_number):
        (distance,indices) = self.ss_tree.query( state,query_number[1] )
        valid_indices = []
        for idx in indices[query_number[0]:query_number[1]]:
            if idx not in self.visited_states:
                valid_indices.append(idx)
        return valid_indices
    
    def take_action(self, children, parent):
        if len(children)==0:
            return None
        child_state = None
        rewards = numpy.array([])
        for child in children:
            if child in self.stored_obs:
                new_obs = self.stored_obs[child]
            else:
                (cloud,_) = self.sim_camera.capture_point_cloud(base_T_camera=state_to_matrix(self.state_space[child]))
                self.sim_camera.voxel_grid.update_grid(cloud)
                new_obs = self.sim_camera.voxel_grid.new_obs
                self.stored_obs[child] = new_obs
                self.sim_camera.voxel_grid.devolve_grid(cloud)
            rate_gain = numpy.sum(new_obs) / numpy.linalg.norm( self.state_space[child]-parent )
            rewards = numpy.append( rewards, rate_gain )
        if numpy.any(rewards>0):
            child = children[numpy.argmax(rewards)]
            child_state = self.state_space[child]
            (cloud,_) = self.sim_camera.capture_point_cloud(base_T_camera=state_to_matrix(child_state))
            self.sim_camera.voxel_grid.update_grid(cloud)
        if numpy.any(rewards<0):
            logger.warn("Rewards are negative. Potential bug")
        self.visited_states.append(child)
        return child_state

    def greedy_search(self):
        self.visited_states = []
        self.stored_obs = {}
        path = [self.state_zero]
        stopping_condition = False
        while not stopping_condition:
            query_number = [0,self.step]
            child_state = None
            while child_state is None:
                children = self.get_children(path[-1], query_number)
                child_state = self.take_action(children, path[-1])
                if child_state is None and query_number[1] != self.state_space.shape[0]:
                    query_number[0] += self.step
                    query_number[1] += self.step
                    if query_number[1] > self.state_space.shape[0]:
                        query_number[1] = self.state_space.shape[0]
                else:
                    break
                        
            if child_state is not None:
                coverage = self.sim_camera.voxel_grid.get_coverage()
                path.append(child_state)
                if coverage > 0.98:
                    stopping_condition = True
                logger.info("Coverage: {0}".format(coverage))
            else:
                stopping_condition = True
        if coverage > 0.98:
            logger.info("Solution found with {0} points and {1} coverage"
                            .format(len(path),self.sim_camera.voxel_grid.get_coverage()))
        else:
            logger.info("Solution found without complete coverage")
        return numpy.array(path)
    
    def get_executable_path(self, path):
        flange_path = numpy.array([tool0_from_camera(pose, self.sim_camera.transformer) for pose in path])
        # Increase path density
        # if numpy.max( numpy.linalg.norm((flange_path[1:]-flange_path[0:-1])[:,0:3], axis=1) ) > 0.01:
        if False:
            exec_path = []
            for i,state in enumerate(flange_path[1:]):
                number_points = int(numpy.linalg.norm( (state-flange_path[i-1])[0:3] ) / 0.005)
                if number_points==0:
                    exec_path.append(state[i-1])
                else:
                    dp = (state - flange_path[i-1]) / number_points
                    for j in range(number_points-1):
                        exec_path.append( flange_path[i-1] + j*dp )
        else:
            exec_path = flange_path.tolist()
        return [state_to_pose(pose) for pose in exec_path]