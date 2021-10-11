from os import wait
import copy
import numpy
import rospy
from geometry_msgs.msg import Pose
from tf.transformations import (
    quaternion_from_euler,
    quaternion_matrix,
    euler_matrix
)

def generate_path_on_cloud(_cloud,transformer,get_pose):
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