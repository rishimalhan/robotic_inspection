#! /usr/bin/python3

from os import name
import rospkg
import roslib
import rosparam

def get_pkg_path(pkg_name):
    # get an instance of RosPack with the default search paths
    rospack = rospkg.RosPack()
    # get the file path for rospy_tutorials
    return rospack.get_path(pkg_name)

def load_yaml(pkg_name, yaml_file, namespace=""):
    if namespace:
        namespace = "/"+namespace
    path = get_pkg_path(pkg_name)
    roslib.load_manifest("rosparam")
    paramlist = rosparam.load_file(path + "/config/" + yaml_file + ".yaml" + namespace)
    for params, ns in paramlist:
        rosparam.upload_params(ns,params)