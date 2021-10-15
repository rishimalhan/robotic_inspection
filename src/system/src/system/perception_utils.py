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
            angle_points_xaxis = numpy.arccos(normalized_points[:,0])
        return numpy.column_stack(( numpy.multiply(distance_to_points,numpy.cos(angle_points_xaxis)),
                                numpy.multiply(distance_to_points,numpy.sin(angle_points_xaxis)),
                                numpy.arccos(-transformed_normals[:,0])    ))

def get_heatmap(cloud, base_T_camera, model, vision_parameters=None):
    # Average error values for each function is 0.3-1.0. A factor 0.01 is multiplied to convert it to max 5 mm
    if vision_parameters is None:
        vision_parameters = get_vision_coordinates(cloud,base_T_camera)
    if cloud.is_empty() or vision_parameters is None:
        return (cloud,vision_parameters)
    return model.predict(vision_parameters)

def get_error_state(_cloud, base_T_camera, model):
    N = 100
    if _cloud.is_empty():
        logger.warn("Invalid request to get accuracy state. Pointcloud is empty")
        return [None]*2
    cloud = copy.deepcopy(_cloud)
    cloud.estimate_normals()
    cloud.orient_normals_towards_camera_location(camera_location=base_T_camera[0:3,3])
    cloud.normalize_normals()

    convergence = False
    number_points = numpy.asarray(cloud.points).shape[0]
    max_itr = 100
    itr = 0
    previous_errors = None
    vision_parameters = get_vision_coordinates(cloud,base_T_camera)
    (error_mean, stddev) = get_heatmap(cloud,base_T_camera,vision_parameters=vision_parameters,
                                model=model) 
    point_errors_set = numpy.array([error_mean,stddev])
    while not convergence and itr < max_itr:
        # Phase two: Get the normal distribution by using point error distribution
        samples = numpy.random.normal(point_errors_set[0],point_errors_set[1],size=(N,number_points))
        nomal_angles_set = []
        for z_deviation in samples:
            points = copy.deepcopy(numpy.asarray(_cloud.points))
            points[:,2] += z_deviation
            cloud.points = open3d.utility.Vector3dVector(points)
            cloud.estimate_normals()
            cloud.orient_normals_towards_camera_location(camera_location=base_T_camera[0:3,3])
            cloud.normalize_normals()
            nomal_angles_set.append( numpy.arccos(numpy.asarray(cloud.normals)[:,0]) )
        nomal_angles = numpy.mean(nomal_angles_set,axis=0)
        normals_stddev = numpy.std(nomal_angles_set,axis=0)

        # Phase one: Estimate point error distrubution using normal distrubution
        samples = numpy.random.normal(nomal_angles,normals_stddev,size=(N,nomal_angles.shape[0]))
        for i,angles in enumerate(samples):
            vision_parameters = get_vision_coordinates(cloud,base_T_camera,angle_points_xaxis=angles)
            (new_means, new_stddev) = get_heatmap(cloud,base_T_camera,vision_parameters=vision_parameters,
                                        model=model) 
            if i==0:
                point_errors_set = numpy.array([new_means, new_stddev])
            else:
                old_means = point_errors_set[0]
                old_stddev = point_errors_set[1]
                point_errors_set[0] = numpy.add(old_means,new_means) / 2
                point_errors_set[1] = numpy.power( numpy.abs((numpy.power(old_stddev,2) + numpy.power(new_stddev,2) +
                                    numpy.power( old_means+new_means,2)) / 4 -
                                    numpy.power(point_errors_set[0],2)), 0.5 )

        if previous_errors is None:
            previous_errors = point_errors_set[0,:]
        else:
            difference = numpy.linalg.norm(point_errors_set[0,:] - previous_errors)
            logger.info("Difference: {0}.".format(difference))
            previous_errors = point_errors_set[0,:]

        itr += 1
    return (point_errors_set[0,:],None)
