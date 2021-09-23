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

    home_config = JointState()
    home_config.position = numpy.radians([0,-16,-30,0,60,0])
    (flag, trajectory, planning_time, err_code) = move_group.plan(home_config)
    print("Trajectory planning to home status: ", flag)
    print(trajectory)
    move_group.execute(trajectory, wait=True)
    move_group.stop()

    # print("Preplanned config: ", move_group.get_current_joint_values())
    # current_config = move_group.get_current_joint_values()
    # goal_config = JointState()
    # goal_config.position = current_config + [0.1,0.1,-0.2,-0.2,0,0]
    # (flag, trajectory, planning_time, err_code) = move_group.plan(goal_config)
    # move_group.execute(trajectory, wait=True)
    # move_group.stop()
    # print("Postplanned config: ", move_group.get_current_joint_values())

    wpose = move_group.get_current_pose().pose
    waypoints = []
    for i in range(10):
        wpose.position.y += (i+1)*0.01
        waypoints.append( wpose )

    (path, fraction) = move_group.compute_cartesian_path(
                                    waypoints,
                                    eef_step=0.001,
                                    jump_threshold=0.05,
                                    avoid_collisions=True,
                                    path_constraints=None,
                                )
    print("Fraction of path feasible: ",fraction)
    move_group.execute(path, wait=True)
    move_group.stop()


if __name__=='__main__':
    main()