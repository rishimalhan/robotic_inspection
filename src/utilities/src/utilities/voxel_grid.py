import numpy
import open3d
import sys

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
        self.points = self.reject_outliers(self.points)
        self.cg = numpy.average(self.points,axis=0).tolist()
        return

class VoxelGrid(open3d.geometry.VoxelGrid):
    def __init__(self,points,create_from_bounds=False,sim=False):
        self.sim = sim
        cloud = open3d.geometry.PointCloud()
        cloud.points = open3d.utility.Vector3dVector(points)
        cloud.colors = open3d.utility.Vector3dVector(numpy.ones(points.shape)*[0,0,1])
        self.voxel_grid = self.create_from_point_cloud(cloud, voxel_size=0.01)
        self.grid_indices = []
        for voxel in self.get_all_voxels():
            self.grid_indices.append(voxel.grid_index)
        self.number_voxels = len(self.grid_indices)
        self.threshold_obs = 4
        self.max_points = self.number_voxels*self.threshold_obs
        self.max_indices = numpy.max(self.grid_indices,axis=0)+1
        self.cg_array = []
        self.cloud = open3d.geometry.PointCloud()
        self.reset()

    def devolve_grid(self,cloud):
        points = numpy.asarray(cloud.points)
        for point in points:
            index = self.voxel_grid.get_voxel(point)
            self.number_observations[index[0],index[1],index[2]] -= 1

    def update_grid(self,cloud):
        valid_indices = numpy.where(self.voxel_grid.check_if_included(cloud.points))[0]
        points = numpy.asarray(cloud.points)[valid_indices]
        for point in points:
            index = self.voxel_grid.get_voxel(point)
            
            if not self.sim:
                if self.number_observations[index[0],index[1],index[2]]==0:
                    self.voxels[index[0],index[1],index[2]] = Voxel()
                else:
                    self.cg_array.remove(self.voxels[index[0],index[1],index[2]].cg)
                self.voxels[index[0],index[1],index[2]].add_point(point)
                self.cg_array.append(self.voxels[index[0],index[1],index[2]].cg)
            else:
                self.new_obs = numpy.zeros(shape=(self.max_indices[0],
                                    self.max_indices[1],self.max_indices[2]),dtype=int)
                self.number_observations[index[0],index[1],index[2]] += 1
                if self.number_observations[index[0],index[1],index[2]]<=self.threshold_obs:
                    self.new_obs[index[0],index[1],index[2]] += 1
    
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

    def get_coverage(self):
        return (numpy.sum(numpy.where(self.number_observations>=self.threshold_obs, 1, 0)) / 
                            self.number_voxels)

    def reset(self):
        self.number_observations = numpy.zeros(shape=(self.max_indices[0],
                                    self.max_indices[1],self.max_indices[2]),dtype=int)
        self.voxels = numpy.empty(shape=(self.max_indices[0],
                                    self.max_indices[1],self.max_indices[2]),dtype=object)