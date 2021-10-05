import open3d
import numpy

class SimCamera:
    def __init__(self,stl_path=None):
        self.latest_cloud = None
        self.stl_path = stl_path
        if stl_path:
            self.default_cloud = pcd = open3d.io.read_point_cloud(self.stl_path)
        pass
    def capture_point_cloud(self):
        self.latest_cloud = self.default_cloud
        pass
    def get_vision_coordinates(self, camera_transform):
        pass