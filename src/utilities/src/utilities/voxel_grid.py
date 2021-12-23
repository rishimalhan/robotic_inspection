import numpy
import open3d
import sys
import logging
logger = logging.getLogger("rosout")

class Voxel:
    def __init__(self):
        self.points = []
        self.cg = None
        return

    def reject_outliers(self,data, m = 1.0):
        d = numpy.abs(data - numpy.median(data))
        mdev = numpy.median(d)
        s = d/mdev if mdev else 0.
        return data[s<m]

    def add_point(self,point):
        self.points.append(point)
        if len(self.points)==1:
            self.cg = self.points[0].tolist()
            return
        if len(self.points)>10:
            self.points = self.points[-10:]

        # self.points = self.reject_outliers(self.points)
        self.cg = numpy.average(self.points,axis=0).tolist()
        return

class VoxelGrid(open3d.geometry.VoxelGrid):
    def __init__(self,points,create_from_bounds=False,sim=False):
        self.sim = sim
        self.output_points = []
        cloud = open3d.geometry.PointCloud()
        cloud.points = open3d.utility.Vector3dVector(points)
        cloud.colors = open3d.utility.Vector3dVector(numpy.ones(points.shape)*[0,0,1])
        if self.sim:
            self.voxel_grid = self.create_from_point_cloud(cloud, voxel_size=0.0015)
        else:
            limits = numpy.array([0.02,0.02,0.02])
            _points = numpy.asarray(cloud.points)
            cloud.estimate_normals()
            cloud.normalize_normals()
            _normals = numpy.asarray(cloud.normals)
            _new_points = numpy.asarray(cloud.points)
            for i in range(10):
                _new_points = numpy.append( _new_points, _points + _normals * numpy.random.uniform(low=-limits,
                                                                            high=limits, size=(_points.shape[0],3)), axis=0 )
            cloud = open3d.geometry.PointCloud()
            cloud.points = open3d.utility.Vector3dVector(_new_points)
            cloud.colors = open3d.utility.Vector3dVector(numpy.ones(_new_points.shape)*[0,0,1])

            self.voxel_grid = self.create_from_point_cloud(cloud, voxel_size=0.001)
        self.grid_indices = []
        for voxel in self.get_all_voxels():
            self.grid_indices.append(voxel.grid_index)
        self.number_voxels = len(self.grid_indices)
        logger.info("Number of points: {0}. Number of Voxels: {1}.".format(numpy.asarray(cloud.points).shape[0], self.number_voxels))
        self.threshold_obs = 1
        self.max_points = self.number_voxels*self.threshold_obs
        self.max_indices = numpy.max(self.grid_indices,axis=0)+1
        self.cg_array = []
        self.cloud = open3d.geometry.PointCloud()
        self.reset()

    def get_valid_points(self,cloud):
        valid_indices = numpy.where(self.voxel_grid.check_if_included(cloud.points))[0]
        return numpy.asarray(cloud.points)[valid_indices]

    def devolve_grid(self,cloud):
        points = self.get_valid_points(cloud)
        for point in points:
            index = self.voxel_grid.get_voxel(point)
            self.number_observations[index[0],index[1],index[2]] -= 1

    def update_grid(self,cloud,update_observations=True):
        self.new_obs = numpy.zeros(shape=(self.max_indices[0],
                                    self.max_indices[1],self.max_indices[2]),dtype=int)
        points = self.get_valid_points(cloud)
        for point in points:
            index = self.voxel_grid.get_voxel(point)
            
            if not self.sim:
                # if self.number_observations[index[0],index[1],index[2]]==0:
                #     self.voxels[index[0],index[1],index[2]] = Voxel()
                # else:
                #     self.cg_array.remove(self.voxels[index[0],index[1],index[2]].cg)
                # self.voxels[index[0],index[1],index[2]].add_point(point)
                # self.cg_array.append(self.voxels[index[0],index[1],index[2]].cg)
                self.cg_array.append(point)
            else:
                # If this voxel is being newly discovered it's a new observation
                if self.number_observations[index[0],index[1],index[2]]<self.threshold_obs:
                    self.new_obs[index[0],index[1],index[2]] += 1
                if update_observations:
                    self.number_observations[index[0],index[1],index[2]] += 1
    
    def get_cloud(self):
        if len(self.cg_array)>0:
            cg_coordinates = numpy.array(self.cg_array)
            self.cloud.points = open3d.utility.Vector3dVector(cg_coordinates)
            self.cloud.colors = open3d.utility.Vector3dVector( 
                                            numpy.ones(cg_coordinates.shape)*[0.447,0.62,0.811])
        return self.cloud

    def within_limits(self,state):
        return (numpy.any( state > self.sim_camera.max_bounds ) or numpy.any( state < self.sim_camera.min_bounds ))

    def get_all_voxels(self):
        return self.voxel_grid.get_voxels()
    
    def voxel_threshold_met(self):
        return numpy.all(self.number_observations==10)

    def get_coverage(self, observations=None):
        if observations is None:
            return (numpy.sum(numpy.where(self.number_observations>=1, 1, 0)) / 
                        self.number_voxels)
        else:
            return (numpy.sum(numpy.where(observations>=1, 1, 0)) / 
                        self.number_voxels)
    
    def get_score(self):
        return (numpy.sum(numpy.where(self.number_observations>=self.threshold_obs, 1, 0)) / 
                    self.number_voxels)

    def reset(self):
        self.number_observations = numpy.zeros(shape=(self.max_indices[0],
                                    self.max_indices[1],self.max_indices[2]),dtype=int)
        self.voxels = numpy.empty(shape=(self.max_indices[0],
                                    self.max_indices[1],self.max_indices[2]),dtype=object)