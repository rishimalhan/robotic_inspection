import numpy
import rospy
import logging
import copy
import open3d
logger = logging.getLogger("rosout")


def get_vision_coordinates(cloud, base_T_camera, angle_points_xaxis=None):
    if cloud.is_empty():
        return (cloud,None)
    camera_T_base = numpy.linalg.inv(base_T_camera)
    if not cloud.is_empty() and not rospy.is_shutdown():
        # Get points and normals in the base frame
        points = numpy.asarray(cloud.points)
        normals = numpy.asarray(cloud.normals)
        # Transform everything to camera_depth_frame
        transformed_points = numpy.matmul(camera_T_base[0:3,0:3],points.T).T + camera_T_base[0:3,3]
        transformed_normals = numpy.matmul(camera_T_base[0:3,0:3],normals.T).T

        # Points are vectors to 0,0,0 this is camera frame
        distance_to_points = numpy.linalg.norm(transformed_points,axis=1)
        if angle_points_xaxis is None:
            normalized_points = transformed_points / distance_to_points[:,None]
            angle_points_xaxis = numpy.arccos(normalized_points[:,2])
        transformed_normals[numpy.where(transformed_normals[:,2]<-1)[0],2] = -1
        transformed_normals[numpy.where(transformed_normals[:,2]>1)[0],2] = 1
        return numpy.column_stack(( numpy.multiply(distance_to_points,numpy.cos(angle_points_xaxis)),
                                numpy.multiply(distance_to_points,numpy.sin(angle_points_xaxis)),
                                numpy.arccos(-transformed_normals[:,2])    ))

def get_heatmap(cloud, base_T_camera, model, vision_parameters=None):
    # Average error values for each function is 0.3-1.0. A factor 0.01 is multiplied to convert it to max 5 mm
    if vision_parameters is None:
        vision_parameters = get_vision_coordinates(cloud,base_T_camera)
    if cloud.is_empty() or vision_parameters is None:
        return (cloud,vision_parameters)
    return model.predict(vision_parameters)