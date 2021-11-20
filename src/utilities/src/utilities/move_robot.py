import rospy
from moveit_msgs.srv import ExecuteKnownTrajectory
from moveit_msgs.srv import ExecuteKnownTrajectoryRequest
from moveit_msgs.srv import ExecuteKnownTrajectoryResponse
from trajectory_msgs.msg import JointTrajectoryPoint
from industrial_msgs.srv import (
    CmdJointTrajectory, 
    CmdJointTrajectoryResponse, 
    CmdJointTrajectoryRequest
)

"""
Give it a list of links and its wanted positions
and it will send a trajectory to get there
using MoveIt!
Author: Sammy Pfeiffer <Sammy.Pfeiffer at student.uts.edu.au>
"""


class GoToConfiguration(object):
    def __init__(self, time=3.0, wait_for_execution=True):
        """
        Class to do calls to MoveIt! /execute_kinematic_path
        :param float time: default goal time
        :param bool wait_for_execution: use blocking calls or not.
        """
        rospy.loginfo("Initalizing GoToPose...")
        self.time = time
        self.wait_for_execution = wait_for_execution
        rospy.loginfo("Default goal time of: " + str(self.time))
        rospy.loginfo("Blocking call: " + str(self.wait_for_execution))
        self.ekt_srv = rospy.ServiceProxy('/joint_path_command',
                                          CmdJointTrajectory)
        rospy.loginfo("Waiting for /joint_path_command service...")
        self.ekt_srv.wait_for_service()
        rospy.loginfo("Connected!")

    def go_to_configuration(self, joint_names, positions,
                            time=None, wait_for_execution=None):
        """
        Execute a trajectory from current configuration to
        the given configuration.
        :param list of str joint_names: List of joint names.
        :param list of float positions: List of positions.
        """
        if len(joint_names) != len(positions):
            rospy.logerr("joint_names and positions have different lengths!")
            resp = CmdJointTrajectoryResponse()
            resp.error_code = 9999  # Failure
            return resp
        if time is None:
            time = self.time
        if wait_for_execution is None:
            wait_for_execution = self.wait_for_execution
        req = CmdJointTrajectoryRequest()
        req.trajectory.joint_names = joint_names
        p = JointTrajectoryPoint()
        p.positions = positions
        p.time_from_start = rospy.Duration(time)
        req.trajectory.points.append(p)
        # req.wait_for_execution = wait_for_execution
        try:
            resp = self.ekt_srv.call(req)
            return resp
        except rospy.ServiceException as e:
            rospy.logerr("Service exception: " + str(e))
            resp = CmdJointTrajectoryResponse()
            resp.error_code = 9999  # Failure
            return resp
