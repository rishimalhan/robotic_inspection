import rospy
from camera.srv import data_capture

class DataCapture:
    def __init__(self,camera):
        self.camera = camera
        self.capture_srv = rospy.Service('data_capture',
                                               data_capture,
                                               self.start_capture)

    def start_capture(self):
        
        return