import rospy
from rviz_tools_py.rviz_tools import RvizMarkers
import logging
from tf import transformations
from geometry_msgs.msg import (
    Point,
    Pose
)
from system.planning_utils import pose_to_state
import threading
logger = logging.getLogger("rosout")

class Visualizer:
    def __init__(self):
        # Initialize the ROS Node
        self.markers = RvizMarkers('base_link', 'visualization_marker')
        self.axes = None

    # Define exit handler
    def __del__(self):
        logger.info("Cleaning up visualizer")
        self.markers.deleteAllMarkers()

    def start_visualizer_async(self):
        logger.info("Starting cloud construction")
        th = threading.Thread(target=self.start_visualizer)
        th.start()

    def start_visualizer(self):
        logger.info("Starting up visualizer.")
        while not rospy.is_shutdown():
            if self.axes is not None:
                path = []
                for axis in self.axes:
                    if isinstance(axis,Pose):
                        axis = pose_to_state(axis)
                    point = Point()
                    point.x = axis[0]
                    point.y = axis[1]
                    point.z = axis[2]
                    path.append(point)
                    # Publish an axis using a numpy transform matrix
                    T = transformations.euler_matrix(axis[3],axis[4],axis[5],'rxyz')
                    T[0:3,3] = axis[0:3]
                    axis_length = 0.02
                    axis_radius = 0.003
                    self.markers.publishAxis(T, axis_length, axis_radius, 2.0) # pose, axis length, radius, lifetime
                self.markers.publishPath(path, width=0.003, color='green')