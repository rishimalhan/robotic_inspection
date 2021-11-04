import numpy
import sys
from scipy.linalg import lstsq
from scipy.optimize import minimize
from system.planning_utils import (
    tf_to_state,
    state_to_matrix
)
import logging
logger = logging.getLogger("rosout")
from scipy.optimize import Bounds
import open3d

class Localizer:
    def __init__(self, clouds, transforms, init_guess):
        # The clouds are downsampled and filtered and wrt camera frame
        self.clouds = clouds
        self.transforms = transforms
        self.x0 = init_guess
        return
    
    def cloud_error(self,x):
        tool0_T_camera = state_to_matrix(x)
        points = []
        for i,cloud in enumerate(self.clouds):
            base_T_camera = numpy.matmul(self.transforms[i],tool0_T_camera)
            cloud_points = numpy.asarray(cloud.points)
            cloud_points = numpy.matmul(base_T_camera[0:3,0:3],cloud_points.T).T + base_T_camera[0:3,3]
            points.extend(cloud_points)
        points = numpy.array(points)
        A = numpy.column_stack(( points[:,0:2],numpy.ones(points.shape[0],) ))
        b = points[:,2]
        fit, residual, rnk, s = lstsq(A, b)
        plane = numpy.array([ fit[0],fit[1],-1,fit[2] ]) / fit[2]
        error = ( abs(plane[0]*points[:,0]+plane[1]*points[:,1]+plane[2]*points[:,2]+plane[3]) 
                        / numpy.linalg.norm(plane[0:3]) )
        return numpy.average(error)

    def localize(self):
        logger.info("Initial transform for camera: {0}. Error: {1}".format(self.x0, self.cloud_error(self.x0)))
        logger.info("Localizing camera using optimization routine")

        lb = self.x0 - numpy.array([ 0.01,0.01,0.01,0.1,0.1,0.1 ])
        ub = self.x0 + numpy.array([ 0.01,0.01,0.01,0.1,0.1,0.1 ])
        # Optimize wrt fixed plane
        opt_res = minimize(self.cloud_error, self.x0, method='SLSQP', 
                    bounds=Bounds(lb, ub, keep_feasible=False), options={'disp':True})
        if opt_res.success:
            logger.info("Optimized transform for camera: {0}. Error: {1}"
                            .format(opt_res.x, self.cloud_error(opt_res.x)))
            return opt_res.x
        else:
            logger.info("Localization failed")
            return self.x0

