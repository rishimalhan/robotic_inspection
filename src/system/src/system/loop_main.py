#! /usr/bin/python3

import os
import time

for i in range(100):
    os.system("rosrun system main.py")
    time.sleep(1)