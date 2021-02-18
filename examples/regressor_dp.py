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
    #else:
    #    raise ValueError("Unrecognized option: " + opt)


# Load the URDF model with RobotWrapper
# Conversion with str seems to be necessary when executing this file with ipython
pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))),"models")
model_path = join(pinocchio_model_dir,"others/robots")
mesh_dir = model_path
#urdf_filename = "2DOF_description.urdf"
#urdf_model_path = join(join(model_path,"2DOF_description/urdf"),urdf_filename)

urdf_filename = "double_pendulum.urdf"
urdf_model_path = join(join(model_path,"double_pendulum_description/urdf"),urdf_filename)
robot = RobotWrapper.BuildFromURDF(urdf_model_path, mesh_dir, pin.JointModelFreeFlyer())

model = robot.model
data = robot.data	
nq, nv = model.nq, model.nv

#numbers of samples
N = 999 

#test case
q = pin.neutral(model)
v = pin.utils.rand(model.nv)
a = pin.utils.zero(model.nv)
#print('q = ', q)
#print('v = ', v)
#print('a = ', a)
tau = pin.rnea(model, data, q, v, a)
#print('tau = ', tau.T)

Y = pin.computeJointTorqueRegressor(model, data, q, v, a)
#print('Y = ', Y)
J = pin.crba(model, data, q)
print('J = ', J)
#generate waypoints
q = np.array([[0, 0 , 0 , 0 , 0 , 0 , 0 , 0, 0]])
qd = np.array([[0, 0 , 0 , 0 , 0 , 0 , 0 , 0, 0]])
qdd = np.array([[0, 0 , 0 , 0 , 0 , 0 , 0 , 0, 0]])

t = 0
eps = 1e-4
a = 0
b = 3
c = 2
d = 2
q_l2 = q_l1 = 0.0
for i in range(N):	
	q_l1 =  a + b*t + c*t*t + d*t*t*t
	q_l2 =  a + b*t + c*t*t + d*t*t*t
    
	qd_l1 = b + 2*c*t + 3*d*t*t
	qd_l2 = b + 2*c*t + 3*d*t*t
    
	qdd_l1 = 2*c + 6*d*t
	qdd_l2 = 2*c + 6*d*t
    
	t = t+ eps
#q = np.append(q, np.array([[0,0,0, 0,0,0, 0,q_l1,q_l2]]), axis = 0)
#qd = np.append(qd, np.array([[0,0,0, 0,0,0, 0,qd_l1,qd_l2]]), axis = 0)
#qdd = np.append(qdd, np.array([[0,0,0, 0,0,0, 0,qd_l1,qd_l2]]), axis = 0)

#compute Joint Torque regressor

    #w = pin.computeJointTorqueRegressor(model, data, q, v, a)
    #todo: append to matrix
#todo: naming columns

#joint torques by rnea()
    #tau = pin.rnea(model, data, q, v, a)   
#decomposition 

#todo: print out table of combined values after decomposition 


#display 
dt = 1e-2
if VISUALIZER:
	
    robot.setVisualizer(VISUALIZER())
    robot.initViewer()
    robot.loadViewerModel("pinocchio")
    for k in range(q.shape[1]):
        t0 = time.time()
        robot.display(q[:, k])
        t1 = time.time()
        elapsed_time = t1 - t0
        if elapsed_time < dt:
            time.sleep(dt - elapsed_time)
    