cmake_minimum_required(VERSION 3.1.3)
project(utilities)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

catkin_install_python(
  PROGRAMS src/utilities/filesystem_utils.py
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})