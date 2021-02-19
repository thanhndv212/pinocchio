#
# In this short script, we show how to use RobotWrapper
# integrating different kinds of viewers
#

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import (GepettoVisualizer, MeshcatVisualizer)
from sys import argv
import time 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import norm, solve
from pinocchio.utils import *
import os
from os.path import dirname, join, abspath

# If you want to visualize the robot in this example,
# you can choose which visualizer to employ
# by specifying an option from the command line:
# GepettoVisualizer: -g
# MeshcatVisualizer: -m
VISUALIZER = None
if len(argv)>1:
    opt = argv[1]
    if opt == '-g':
        VISUALIZER = GepettoVisualizer
    elif opt == '-m':
        VISUALIZER = MeshcatVisualizer
    else:
        raise ValueError("Unrecognized option: " + opt)

# Load the URDF model with RobotWrapper
# Conversion with str seems to be necessary when executing this file with ipython
pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))),"models")
model_path = join(pinocchio_model_dir,"others/robots")
mesh_dir = model_path
urdf_filename = "2DOF_description.urdf"
urdf_model_path = join(join(model_path,"2DOF_description/urdf"),urdf_filename)
robot = RobotWrapper.BuildFromURDF(urdf_model_path, mesh_dir, pin.JointModelFreeFlyer())

# alias
model = robot.model
data = robot.data

# do whatever, e.g. compute the center of mass position expressed in the world frame
#q0 = robot.q0

#q1 = np.array([0,0,0, 0,0,0, 0,0.5,0.5])
#generate waypoints
q1 = np.array([[0, 0 , 0 , 0 , 0 , 0 , 0 , 0, 0]])
t = 0
eps = 1e-4
a = 0
b = 3
c = 2
d = 2
q_l2 = q_l1 = 0.0
Q = np.array([[0,0,0]])
for i in range(1000):   
    q_l1 =  a + b*t + c*t*t + d*t*t*t
    q_l2 =  a + 2*b*t + c*t*t + d*t*t*t
    t = t+ eps
    q1 = np.append(q1, np.array([[0,0,0, 0,0,0, 0,q_l1,q_l2]]), axis = 0)
    
    
q1 = np.delete(q1, 0, axis=0)  
q1 = q1.T

# Show model with a visualizer of your choice
dt = 1e-2
if VISUALIZER:
    robot.setVisualizer(VISUALIZER())
    robot.initViewer()
    robot.loadViewerModel("pinocchio")
    for k in range(q1.shape[1]):
        t0 = time.time()
        robot.display(q1[:, k])
        t1 = time.time()
        elapsed_time = t1 - t0
        if elapsed_time < dt:
            time.sleep(dt - elapsed_time)
#    robot.setVisualizer(VISUALIZER())
#    robot.initViewer()
#    robot.loadViewerModel("pinocchio")
#    robot.display(q1)
