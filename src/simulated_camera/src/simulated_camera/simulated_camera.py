

class SimCamera:
    def __init__(self,default_cloud):
        self.latest_cloud = None
        self.default_cloud = default_cloud
        pass
    def capture_point_cloud(self):
        self.latest_cloud = self.default_cloud
        pass
    def get_vision_coordinates(self, camera_transform):
        pass