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
urdf_filename = "2DOF_description.urdf"
urdf_model_path = join(join(model_path,"2DOF_description/urdf"),urdf_filename)

#urdf_filename = "double_pendulum.urdf"
#urdf_model_path = join(join(model_path,"double_pendulum_description/urdf"),urdf_filename)
robot = RobotWrapper.BuildFromURDF(urdf_model_path, mesh_dir, pin.JointModelFreeFlyer())

model = robot.model
data = robot.data	
nq, nv = model.nq, model.nv

#numbers of samples
N = 999 

#test case
q = pin.neutral(model)
#q = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1])



v = pin.utils.rand(model.nv)
#v = np.array([1, 1, 1, 1, 1, 1, 1, 1])

a = pin.utils.zero(model.nv)
#a = np.array([1, 1, 1, 1, 1, 1, 1, 1])

print('q = ', q)
print('v = ', v)
print('a = ', a)


TAU = pin.rnea(model, data, q, v, a)
#print('tau = ', tau)


W = pin.computeJointTorqueRegressor(model, data, q, v, a)
#print('Y = ', Y)

#J = pin.crba(model, data, q)
#print('J = ', J)











#generate waypoints
Q = np.array([[0, 0 , 0 , 0 , 0 , 0 , 0 , 0, 0]])
DQ = np.array([[ 0 , 0 , 0 , 0 , 0 , 0 , 0, 0]])
DDQ = np.array([[ 0 , 0 , 0 , 0 , 0 , 0 , 0, 0]])

t = 0
eps = 1e-4
a = 0
b = 3
c = 2
d = 2
q_l2 = q_l1 = 0.0
q_temp = q
v_temp = v
a_temp = a
for i in range(N):	
	q_l1 =  a + b*t + c*t*t + d*t*t*t
	q_l2 =  a + b*t + c*t*t + d*t*t*t
	q_temp =  np.array([0,0,0, 0,0,0, 0,q_l1,q_l2])
	qd_l1 = b + 2*c*t + 3*d*t*t
	qd_l2 = b + 2*c*t + 3*d*t*t
	v_temp = np.array([0,0,0, 0,0,0, 0,qd_l1,qd_l2])
	qdd_l1 = 2*c + 6*d*t
	qdd_l2 = 2*c + 6*d*t
	a_temp = np.array([0,0,0, 0,0,0, 0,qdd_l1,qdd_l2])
	t = t+ eps
	tau_temp = pin.rnea(model, data, q_temp, v_temp, a_temp)
	W_temp = pin.computeJointTorqueRegressor(model, data, q_temp, v_temp, a_temp)
	TAU = np.vstack((TAU,tau_temp))

	W = np.vstack((W,W_temp)) #size of 8000 x 30


TAU = TAU.T #transpose the joint torque matrix, size of 8x1000, the last 2 rows corresponds to 2 joint torque

#inertial parameters of link2 from urdf 
P = model.inertias[1].toDynamicParameters()
for i in range(2):
	p_i = model.inertias[i+2].toDynamicParameters()
	P = np.append(P, p_i)


#calculate joint torque between link1 and link2
TAU_est = np.dot(W,P)

tau2_est = np.array([TAU_est[7]])
for i in range(999):
	tau2_est = np.append(tau2_est, TAU_est[8*(i+1)-1 + 8])

# joint 2 by rnea()
tau2 = TAU[7,:]
print(tau2.shape[0])
#plot measured tau and estimated tau 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(tau2, tau2_est)
m, b = np.polyfit(tau2, tau2_est,1)
print(m, b)
plt.plot(tau2, m*tau2 + b,'r', label='y={:.2f}x+{:.2f}'.format(m,b))
plt.xlabel("Joint torque 2 by rnea() function")
plt.ylabel("Joint torque 2 estimated by regressor matrix")
plt.legend()
plt.grid()
plt.show()



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
	