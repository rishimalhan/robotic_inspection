cmake_minimum_required(VERSION 3.1.3)
project(camera)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    message_generation
    )

catkin_python_setup()

add_service_files(
    FILES
    data_capture.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
 LIBRARIES ${PROJECT_NAME}
 CATKIN_DEPENDS roscpp rospy std_msgs
)
