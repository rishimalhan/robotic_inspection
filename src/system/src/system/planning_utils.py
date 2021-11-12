from os import wait
import copy
import numpy
from numpy.core.defchararray import asarray
import rospy
import math
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

def generate_waypoints_on_cloud(_cloud,transformer,get_pose):
    cloud = copy.deepcopy(_cloud.voxel_down_sample(voxel_size=0.03))
    points = numpy.asarray(cloud.points)
    # Arrange points in increasing X value
    costs = points[:,0] + points[:,1]*10
    points = points[numpy.argsort(costs),:]

    batches = []
    batch = []
    for i in range(1,points.shape[0]):
        batch.append(i-1)
        if points[i-1,0]-points[i,0] > 0.15:
            batches.append(batch)
            batch = []
    batch.append(points.shape[0]-1)
    batches.append(batch)
    
    sorted_points = []
    for i in range(len(batches)):
        batch = copy.deepcopy(batches[i])
        if i%2!=0:
            # Reverse the batch
            sorted_points.extend(points[batch[::-1],:])
        else:
            sorted_points.extend(points[batch,:])
    points = numpy.array(sorted_points)
        
    waypoints = []
    tool0_T_camera_vec = transformer.lookupTransform("tool0", "camera_depth_optical_frame", rospy.Time(0))
    tool0_T_camera = quaternion_matrix( [tool0_T_camera_vec[1][0], tool0_T_camera_vec[1][1],
                                        tool0_T_camera_vec[1][2], tool0_T_camera_vec[1][3]] )
    tool0_T_camera[0:3,3] = tool0_T_camera_vec[0]
    camera_T_tool0 = numpy.linalg.inv(tool0_T_camera)
    for point in points:
        base_T_camera = euler_matrix(-3.14,0,-1.57,'rxyz')
        base_T_camera[0:3,3] = point[0:3]
        base_T_camera[2,3] += 0.15
        base_T_tool0 = numpy.matmul( base_T_camera,camera_T_tool0 )
        waypoints.append(get_pose(base_T_tool0))
    return waypoints

def generate_state_space(_cloud, camera_home):
    voxelized_cloud = copy.deepcopy(_cloud.voxel_down_sample(voxel_size=0.1))
    voxelized_cloud.normalize_normals()
    points = []
    # points.extend(numpy.asarray(voxelized_cloud.points) + numpy.array([0,0,0.15]))
    points.extend(numpy.asarray(voxelized_cloud.points) + numpy.array([0,0,0.2]))
    points = numpy.array(points)
    # normals = numpy.vstack(( numpy.asarray(voxelized_cloud.normals),numpy.asarray(voxelized_cloud.normals) ))
    states = []
    camera_orientation = camera_home[3:6]
    x_rotations = [-3.82, -3.48, -3.14, -2.8, -2.46]
    y_rotations = [-0.69, -0.34, 0, 0.34, 0.69]
    for point in points:
        for rotx in x_rotations:
            states.append( numpy.hstack(( point,[rotx,0,-1.57] )) )
        for roty in y_rotations:
            states.append( numpy.hstack(( point,[-3.14,roty,-1.57] )) )
    return numpy.array(states)

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

def update_cloud_live(camera):
    (cloud,base_T_camera) = camera.capture_point_cloud(base_T_camera=None)
    (heatmap,_) = get_heatmap(cloud, base_T_camera, camera.camera_model, vision_parameters=None)
    heatmap = (heatmap - numpy.min(heatmap)) / (numpy.max(heatmap) - numpy.min(heatmap))
    cloud = cloud.select_by_index(numpy.where(heatmap < 0.6)[0])
    camera.voxel_grid.update_grid(cloud)

def update_cloud(path, sim_camera):
    # states are the transformation matrix for camera frame
    for state in path:
        base_T_camera = state_to_matrix(state)
        (cloud,_) = sim_camera.capture_point_cloud(base_T_camera=base_T_camera)
        (heatmap,_) = get_heatmap(cloud, base_T_camera, sim_camera.camera_model, vision_parameters=None)
        heatmap = (heatmap - numpy.min(heatmap)) / (numpy.max(heatmap) - numpy.min(heatmap))
        cloud = cloud.select_by_index(numpy.where(heatmap < 0.6)[0])
        sim_camera.voxel_grid.update_grid(cloud)
    return