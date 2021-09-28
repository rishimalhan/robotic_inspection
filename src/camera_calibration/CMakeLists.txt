cmake_minimum_required(VERSION 3.1.3)
project(camera_calibration)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

catkin_install_python(
  PROGRAMS src/camera_calibration/bootstrap_camera.py
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
