import numpy
import open3d
import sys

class VoxelGrid(open3d.geometry.VoxelGrid):
    def __init__(self,points,bounds):
        self.min_bounds = bounds[0]
        self.max_bounds = bounds[1]
        cloud = open3d.geometry.PointCloud()
        cloud.points = open3d.utility.Vector3dVector(points)
        cloud.colors = open3d.utility.Vector3dVector(numpy.ones(points.shape)*[0,0,1])
        self.voxel_grid = self.create_from_point_cloud(cloud, voxel_size=0.0015)
        self.grid_indices = []
        for voxel in self.get_all_voxels():
            self.grid_indices.append(voxel.grid_index)
        self.number_voxels = len(self.grid_indices)
        self.threshold_obs = 10
        self.max_points = self.number_voxels*self.threshold_obs
        self.max_indices = numpy.max(self.grid_indices,axis=0)+1
        self.reset()

    def devolve_grid(self,cloud):
        points = numpy.asarray(cloud.points)
        for point in points:
            index = self.voxel_grid.get_voxel(point)
            self.number_observations[index[0],index[1],index[2]] -= 1

    def update_grid(self,cloud):
        self.new_obs = numpy.zeros(shape=(self.max_indices[0],
                                    self.max_indices[1],self.max_indices[2]),dtype=int)
        points = numpy.asarray(cloud.points)
        for point in points:
            index = self.voxel_grid.get_voxel(point)
            self.number_observations[index[0],index[1],index[2]] += 1
            if self.number_observations[index[0],index[1],index[2]]<=self.threshold_obs:
                self.new_obs[index[0],index[1],index[2]] += 1

    def within_limits(self,state):
        return (numpy.any( state > self.sim_camera.max_bounds ) or numpy.any( state < self.sim_camera.min_bounds ))

    def get_all_voxels(self):
        return self.voxel_grid.get_voxels()
    
    def voxel_threshold_met(self):
        return numpy.all(self.number_observations==10)

    def get_coverage(self):
        return (numpy.sum(numpy.where(self.number_observations>=self.threshold_obs, 1, 0)) / 
                            self.number_voxels)

    def reset(self):
       self.number_observations = numpy.zeros(shape=(self.max_indices[0],
                                    self.max_indices[1],self.max_indices[2]),dtype=int)