import numpy
from scipy.linalg import lstsq
from scipy.optimize import minimize
from system.planning_utils import (
    tf_to_state,
    state_to_matrix
)

class Localizer:
    def __init__(self, clouds, transforms, init_guess):
        # The clouds are downsampled and filtered
        self.clouds = clouds
        self.transforms = transforms
        self.x0 = numpy.hstack((tf_to_state(init_guess), self.get_best_fit_plane(self.clouds) ))
    
    def get_best_fit_plane(self,clouds):
        points = []
        for cloud in clouds:
            points.extend(numpy.asarray(cloud.points))
        points = numpy.array(points)
        A = numpy.column_stack(( points[:,0:2],numpy.ones(points.shape[0],) ))
        b = points[:,2]
        fit, residual, rnk, s = lstsq(A, b)
        return fit

    def cloud_error(self,x):
        tool0_T_camera = state_to_matrix(x)
        points = []
        for i,transform in enumerate(self.transforms):
            base_T_camera = numpy.matmul(transform,tool0_T_camera)
            points.append( numpy.asarray(self.clouds[i].transform(transform).points) )
        points = numpy.array(points)


    def localize(self):
        # Optimize wrt fixed plane
        
        return self.x0

