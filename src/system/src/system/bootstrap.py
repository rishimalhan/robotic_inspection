#! /usr/bin/python3

from utilities.filesystem_utils import load_yaml
from camera_calibration.bootstrap_camera import bootstrap_camera

def bootstrap_system():
    # Bootstrap the robot parameters
    load_yaml("utilities","system")
    bootstrap_camera()

if __name__=='__main__':
    bootstrap_system()