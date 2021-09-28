#! /usr/bin/python3

from utilities.filesystem_utils import load_yaml

def bootstrap_camera():
    load_yaml("camera_calibration","system")