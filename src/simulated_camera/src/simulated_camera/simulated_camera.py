import open3d
import numpy

class SimCamera:
    def __init__(self,part_stl=None):
        self.latest_cloud = None
        self.part_stl = part_stl
        if part_stl:
            self.default_cloud = open3d.io.read_point_cloud(self.part_stl)
        pass
    def capture_point_cloud(self):
        if self.part_stl:
            self.latest_cloud = self.default_cloud
        pass
    def get_vision_coordinates(self, camera_transform):
        pass