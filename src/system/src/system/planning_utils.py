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
    euler_from_matrix
)

def generate_waypoints_on_cloud(_cloud,transformer,get_pose):
    cloud = copy.deepcopy(_cloud.voxel_down_sample(voxel_size=0.05))
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
    tool0_T_camera_vec = transformer.lookupTransform("tool0", "camera_depth_frame", rospy.Time(0))
    tool0_T_camera = quaternion_matrix( [tool0_T_camera_vec[1][0], tool0_T_camera_vec[1][1],
                                        tool0_T_camera_vec[1][2], tool0_T_camera_vec[1][3]] )
    tool0_T_camera[0:3,3] = tool0_T_camera_vec[0]
    camera_T_tool0 = numpy.linalg.inv(tool0_T_camera)
    for point in points:
        base_T_camera = euler_matrix(0,1.57,0,'rxyz')
        base_T_camera[0:3,3] = point[0:3]
        base_T_camera[2,3] += 0.15
        base_T_tool0 = numpy.matmul( base_T_camera,camera_T_tool0 )
        waypoints.append(get_pose(base_T_tool0))
    return waypoints

def generate_path_between_states(states):
    pass

def generate_state_space(_cloud):
    voxelized_cloud = copy.deepcopy(_cloud.voxel_down_sample(voxel_size=0.03))
    voxelized_cloud.normalize_normals()
    points = []
    points.extend(numpy.asarray(voxelized_cloud.points) + numpy.array([0,0,0.15]))
    points.extend(numpy.asarray(voxelized_cloud.points) + numpy.array([0,0,0.2]))
    points = numpy.array(points)
    normals = numpy.vstack(( numpy.asarray(voxelized_cloud.normals),numpy.asarray(voxelized_cloud.normals) ))
    states = []
    for i,point in enumerate(points):
        matrix = numpy.identity(4)
        matrix[0:3,1] = [0,1,0]
        matrix[0:3,2] = -normals[i,:]
        matrix[0:3,0] = numpy.cross(matrix[0:3,1],matrix[0:3,2])
        euler_angles = euler_from_matrix( matrix,'sxyz' )
        states.append( numpy.hstack(( point,euler_angles )) )
    return numpy.array(states)

def generate_path_between_states(states):
    import IPython
    IPython.embed()
    # Linear interpolation between states maintaining the Z distance from the part
    state1 = [states[0][0:3,3],euler_from_matrix(states[0],'sxyz')]
    state2 = [states[1][0:3,3],euler_from_matrix(states[1],'sxyz')]
    constant_z = state1[2]
    linear_step = 0.01
    number_points = int(numpy.linalg.norm(state2[0:3]-state1[0:3]) / linear_step)
    dp = (state2 - state1) / number_points
    points = []
    for i in range(number_points):
        point = state1 + i*dp
        matrix = euler_matrix(point[3], point[4], point[5], 'sxyz')
        matrix[0:3,3] = point[0:3]
        points.append( matrix )

def capture_clouds_between_states(inspection_bot, sim_camera):
    # states are the transformation matrix for camera frame
    
    pass