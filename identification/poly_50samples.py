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
VISUALIZER = GepettoVisualizer
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

#motion
q0 = np.array([0, 0 , 0 , 0 , 0 , 0 , 0 , 0, 0])
v = zero(6)
#velocity = robot.velocity(q0,v,3)

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
	q_l2 =  a + b*t + c*t*t + d*t*t*t
	v_l1 = b + 2*c*t + 3*d*t*t
	v_l2 = b + 2*c*t + 3*d*t*t
	a_l1 = 2*c + 6*d*t
	a_l2 = 2*c + 6*d*t
	t = t+ eps
	q1 = np.append(q1, np.array([[0,0,0, 0,0,0, 0,q_l1,q_l2]]), axis = 0)
	Q_z = np.array([[1,-v_l1*np.sin(q_l1),-(v_l1+v_l2)*np.sin(q_l1+q_l2)]])
	Q = np.append(Q, Q_z, axis = 0)
	
	
q1 = np.delete(q1, 0, axis=0)  
q1 = q1.T
for j in range(2):
	Q = np.delete(Q, j, axis= 0)




V = np.array([0])
for k in range(q1.shape[1]):
        p_ee = robot.framePlacement(q1[:,k],7).translation
        if k >0:
        	v_est = (p_ee - p_prev)/eps
        	V = np.append(V, np.array([v_est[2]]), axis=0)
        p_prev = p_ee
V = np.delete(V,0,0)
V = V.T


#solve least square to get L1 and L2
L = np.dot(np.dot(np.linalg.inv(np.dot(Q.T,Q)),Q.T),V)
print(L)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(Q[:,1],Q[:,2],V)
plt.show()

#display 
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
    