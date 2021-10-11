import numpy
import rospy

def get_vision_coordinates(cloud, base_T_camera):
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
        normalized_points = transformed_points / distance_to_points[:,None]
        angle_points_xaxis = numpy.arccos(normalized_points[:,0])
        return numpy.column_stack(( numpy.multiply(distance_to_points,numpy.cos(angle_points_xaxis)),
                                numpy.multiply(distance_to_points,numpy.sin(angle_points_xaxis)),
                                numpy.arccos(-transformed_normals[:,0])    ))

def get_heatmap(cloud, base_T_camera):
    vision_parameters = get_vision_coordinates(cloud,base_T_camera)
    if cloud.is_empty() or vision_parameters is None:
        return (cloud,vision_parameters)
    return numpy.multiply( numpy.multiply( 
                -26.67*numpy.power(vision_parameters[:,0],2) + 8*vision_parameters[:,0] + 0.4,
                numpy.exp(-3.03*vision_parameters[:,1])),
                numpy.exp(-0.29*vision_parameters[:,2]))