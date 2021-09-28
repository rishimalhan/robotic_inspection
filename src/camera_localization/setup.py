#! /usr/bin/python3

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup()
d['packages'] = ['camera_localization']
d['package_dir'] = {'': 'src'}

setup(**d)
