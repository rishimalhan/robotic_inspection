import rospy
from rviz_tools_py.rviz_tools import RvizMarkers
import logging
from tf import transformations
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

    def start_visualizer(self):
        logger.info("Starting up visualizer")
        while not rospy.is_shutdown():
            if self.axes is not None:
                for axis in self.axes:
                    # Publish an axis using a numpy transform matrix
                    T = transformations.euler_matrix(axis[3],axis[4],axis[5],'sxyz')
                    T[0:3,3] = axis[0:3]
                    axis_length = 0.01
                    axis_radius = 0.001
                    self.markers.publishAxis(T, axis_length, axis_radius, 2.0) # pose, axis length, radius, lifetime