#! /usr/bin/python3

import rospy
import logging
from utilities.filesystem_utils import load_yaml
from moveit_commander.planning_scene_interface import PlanningSceneInterface
from geometry_msgs.msg import PoseStamped

logger = logging.getLogger('rosout')

rospy.init_node("add_collision_objects",anonymous=True)

def box_added(name,scene):
    start = rospy.get_time()
    seconds = rospy.get_time()
    while (seconds - start < 2.0) and not rospy.is_shutdown():
        # Test if the box is in the scene.
        is_known = name in scene.get_known_object_names()
        if is_known:
            return True
        # Sleep so that we give other threads time on the processor
        rospy.sleep(0.1)
        seconds = rospy.get_time()
    # If we exited the while loop without returning then we timed out
    return False

# Get collision boxes
load_yaml("system", "system")
collision_boxes = rospy.get_param("/collision_boxes")
scene = PlanningSceneInterface(synchronous=True)
for key in collision_boxes.keys():
    pose = PoseStamped()
    pose.header.frame_id = collision_boxes[key]['frame_id']
    pose.pose.position.x = collision_boxes[key]['position'][0]
    pose.pose.position.y = collision_boxes[key]['position'][1]
    pose.pose.position.z = collision_boxes[key]['position'][2]
    pose.pose.orientation.x = collision_boxes[key]['orientation'][0]
    pose.pose.orientation.y = collision_boxes[key]['orientation'][1]
    pose.pose.orientation.z = collision_boxes[key]['orientation'][2]
    pose.pose.orientation.w = collision_boxes[key]['orientation'][3]
    scene.add_box( collision_boxes[key]['name'], pose, 
                        size=collision_boxes[key]['dimension'])
    if box_added(key,scene):
        logger.info("Collision object {0} added".format(key))
    else:
        raise Exception("Unable to add collision object: ", key)
logger.info("All collision objects added")