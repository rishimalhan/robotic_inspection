#! /usr/bin/python3

import rospy
import moveit_msgs
from moveit_commander.robot import RobotCommander
from moveit_commander.planning_scene_interface import PlanningSceneInterface
from moveit_commander.move_group import MoveGroupCommander
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import numpy

def main():
    rospy.init_node("plan_p2p")
    robot = RobotCommander()
    scene = PlanningSceneInterface()
    group_name = "manipulator"
    move_group = MoveGroupCommander(group_name)

    display_trajectory_publisher = rospy.Publisher(
        "/move_group/display_planned_path",
        moveit_msgs.msg.DisplayTrajectory,
        queue_size=20,
    )
    
    # We get the joint values from the group and change some of the values:
    joint_goal = move_group.get_current_joint_values()
    joint_goal[0] = 0
    joint_goal[1] = -1 / 8
    joint_goal[2] = 0
    joint_goal[3] = -1 / 4
    joint_goal[4] = 0
    joint_goal[5] = 1 / 6  # 1/6 of a turn

    # The go command can be called with joint values, poses, or without any
    # parameters if you have already set the pose or joint target for the group
    move_group.go(joint_goal, wait=True)

    # Calling ``stop()`` ensures that there is no residual movement
    move_group.stop()


if __name__=='__main__':
    main()