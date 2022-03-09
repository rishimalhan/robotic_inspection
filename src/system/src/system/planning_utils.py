from os import wait
import copy
import numpy
from numpy.core.defchararray import asarray
import rospy
import math
import open3d
from geometry_msgs.msg import Pose
from tf.transformations import (
    quaternion_from_euler,
    quaternion_matrix,
    euler_matrix,
    euler_from_matrix,
    euler_from_quaternion
)
from system.perception_utils import (
    get_heatmap
)
from sensor_msgs.msg import PointCloud2
from utilities.open3d_and_ros import (
    convertCloudFromOpen3dToRos,
)
import sys

def tf_to_state(tf):
    return numpy.hstack(( tf[0],euler_from_quaternion(tf[1],'rxyz') ))

def tf_to_matrix(tf):
    matrix = quaternion_matrix( [tf[1][0], tf[1][1],
                                            tf[1][2], tf[1][3]] )
    matrix[0:3,3] = tf[0]
    return matrix

def tool0_from_camera(state, transformer):
    base_T_camera = state_to_matrix(state)
    tool0_T_camera_vec = transformer.lookupTransform("tool0", "camera_depth_optical_frame", rospy.Time(0))
    tool0_T_camera = quaternion_matrix( [tool0_T_camera_vec[1][0], tool0_T_camera_vec[1][1],
                                        tool0_T_camera_vec[1][2], tool0_T_camera_vec[1][3]] )
    tool0_T_camera[0:3,3] = tool0_T_camera_vec[0]
    return matrix_to_state( numpy.matmul(base_T_camera,numpy.linalg.inv(tool0_T_camera)) )

def pose_to_state(pose):
    return numpy.hstack(( [pose.position.x,pose.position.y,pose.position.z],
                         euler_from_quaternion([pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w])))

def state_to_pose(state):
    pose = Pose()
    pose.position.x = state[0]
    pose.position.y = state[1]
    pose.position.z = state[2]
    quaternion = quaternion_from_euler(state[3],state[4],state[5],'rxyz')
    pose.orientation.x = quaternion[0]
    pose.orientation.y = quaternion[1]
    pose.orientation.z = quaternion[2]
    pose.orientation.w = quaternion[3]
    return pose

def state_to_matrix(state):
    matrix = euler_matrix(state[3],state[4],state[5],'rxyz')
    matrix[0:3,3] = state[0:3]
    return matrix

def matrix_to_state(matrix):
    return numpy.hstack(( matrix[0:3,3],euler_from_matrix(matrix,'rxyz') ))

def generate_zigzag(camera_home_,transformer):
    camera_home = copy.deepcopy(camera_home_)
    # Increase in Y then X
    exec_path = []
    sign = 1
    vector = [(0.25/20)]*20
    vector[0] = 0.0
    for i in vector:
        camera_home[0] -= i
        for j in vector:
            camera_home[1] += (sign*j)
            exec_path.append( camera_home.tolist() )
        sign *= -1
    
    for i in [(0.05/10)]*10:
        camera_home[2] -= i
        exec_path.append( camera_home.tolist() )

    sign = 1
    vector = [(0.25/20)]*20
    vector[0] = 0.0
    for i in vector:
        camera_home[0] += i
        for j in vector:
            camera_home[1] += (sign*j)
            exec_path.append( camera_home.tolist() )
        sign *= -1

    exec_path = numpy.array(exec_path)
    flange_path = numpy.array([tool0_from_camera(pose, transformer) for pose in exec_path])
    return [state_to_pose(state) for state in flange_path]

def generate_state_space(_cloud, camera_home):
    _points = numpy.asarray(_cloud.points)
    _normals = numpy.asarray(_cloud.normals)
    _points = _points + _normals*0.3
    _cloud = open3d.geometry.PointCloud()
    _cloud.points = open3d.utility.Vector3dVector(_points)
    _cloud.normals = open3d.utility.Vector3dVector(_normals)
    voxelized_cloud = copy.deepcopy(_cloud.voxel_down_sample(voxel_size=0.06))
    voxelized_cloud.normalize_normals()
    cloud_normals = numpy.asarray(voxelized_cloud.normals)
    points = []
    normals = []
    points.extend(numpy.asarray(voxelized_cloud.points))
    normals.extend(cloud_normals)
    points = numpy.array(points)
    normals = numpy.array(normals)
    base_T_camera = state_to_matrix(camera_home)
    states = []
    
    x_rotations = [-0.3, 0.0, 0.3]
    for i,point in enumerate(points):
        matrix = numpy.identity(4)
        matrix[0:3,0] = base_T_camera[0:3,0] # Keep x axis of state same as camera
        # Keep one state with vertical camera orientation
        states.append( numpy.hstack(( point, matrix_to_state(base_T_camera)[3:6] )) )
        matrix[0:3,2] = -normals[i]
        matrix[0:3,1] = numpy.cross(matrix[0:3,2],matrix[0:3,0])
        point_orientation = matrix_to_state(matrix)[3:6]
        for rotx in x_rotations:
            states.append( numpy.hstack(( point, (point_orientation + [rotx,0,0]) )) )
    states = numpy.array(states)

    min_bound = voxelized_cloud.get_min_bound() - [0.3,0.3,0.0]
    max_bound = voxelized_cloud.get_max_bound() + [0.3,0.3,0.0]
    invalid_idx = numpy.hstack(( numpy.where( (states[:,0] < min_bound[0]) |  (states[:,1] < min_bound[1]) )[0],
                                numpy.where( (states[:,0] > max_bound[0]) |  (states[:,1] > max_bound[1]) )[0] ))
    states = numpy.delete(states,invalid_idx,axis=0)
    return states


def generate_path_between_states(states):
    return states
    # Linear interpolation between states maintaining the Z distance from the part
    # ToDo: Fix the Z distance from the part
    state1 = states[0]
    state2 = states[1]
    constant_z = state1[2]
    linear_step = 0.001
    number_points = int(numpy.linalg.norm(state2[0:3]-state1[0:3]) / linear_step)
    if number_points==0:
        return states
    dp = (state2 - state1) / number_points
    states = []
    for i in range(number_points):
        states.append( state1 + i*dp )
    return states

def update_cloud(path, camera):
    # states are the transformation matrix for camera frame
    for state in path:
        base_T_camera = state_to_matrix(state)
        (cloud,_) = camera.get_simulated_cloud(base_T_camera=base_T_camera)
        (heatmap,_) = get_heatmap(cloud, base_T_camera, camera.camera_model, vision_parameters=None)
        heatmap = (heatmap - numpy.min(heatmap)) / (numpy.max(heatmap) - numpy.min(heatmap))
        cloud = cloud.select_by_index(numpy.where(heatmap < 0.6)[0])
        camera.voxel_grid_sim.update_grid(cloud)
    return