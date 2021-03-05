import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import (GepettoVisualizer, MeshcatVisualizer)
from sys import argv

from tabulate import tabulate
import time 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import norm, solve
from pinocchio.utils import *
from scipy import linalg
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

# urdf_filename = "double_pendulum.urdf"
# urdf_model_path = join(join(model_path,"double_pendulum_description/urdf"),urdf_filename)
robot = RobotWrapper.BuildFromURDF(urdf_model_path, mesh_dir)

model = robot.model
data = robot.data	
nq, nv = model.nq, model.nv


#numbers of samples
N = 49

#test case
q = pin.neutral(model)
# q = np.array([ 0.5, 0.5])
v = pin.utils.zero(model.nv)

a = pin.utils.zero(model.nv)


print(model)
TAU = pin.rnea(model, data, q, v, a)
W = pin.computeJointTorqueRegressor(model, data, q, v, a)

##################
print('*******************************************')
##################

#generate waypoints

t = 0
eps = 1e-4
A1 = np.random.uniform(low = 5, high = 10, size=(3,))
A2 = np.random.uniform(low = -7, high = 10, size=(3,))
a = -0.7
b = 10
c = 2
d = 2
q_l2 = q_l1 = 0.0
q_temp = q
v_temp = v
a_temp = a
for i in range(N):	
	q_l1 =  a + A1[0]*t + A1[1]*t*t + A1[2]*t*t*t
	q_l2 =  a + A2[0]*t + A2[1]*t*t + A2[2]*t*t*t
	# q_temp =  np.array([q_l1,q_l2])
	q_temp = np.random.uniform(low = -10, high = 10, size=(2,))
	q = np.vstack((q,q_temp))
	qd_l1 = A1[0] + 2*A1[1]*t + 3*A1[2]*t*t
	qd_l2 = A2[0] + 2*A2[1]*t + 3*A2[2]*t*t
	# v_temp = np.array([qd_l1,qd_l2])
	v_temp = np.random.uniform(low = -10, high = 10, size=(2,))
	qdd_l1 = 2*A1[1] + 6*A1[2]*t
	qdd_l2 = 2*A2[1] + 6*A2[2]*t
	# a_temp = np.array([qdd_l1,qdd_l2])
	a_temp = np.random.uniform(low = -10, high = 10, size=(2,))
	t = t+ eps
	tau_temp = pin.rnea(model, data, q_temp, v_temp, a_temp)
	W_temp = pin.computeJointTorqueRegressor(model, data, q_temp, v_temp, a_temp)
	#TODO: rearrange tau and W by the order of links
	TAU = np.append(TAU,tau_temp)
	W = np.vstack((W,W_temp))
# print(dir(model))
#inertial parameters of link2 from urdf model
phi1 = model.inertias[1].toDynamicParameters()
# print(model.friction)
phi2 = model.inertias[2].toDynamicParameters()
phi = np.round(np.append(phi1, phi2),6)
print('standard paramaters: ')
#TODO: generialize listing of inertia parameters. 
params_full = ['m1', 'mx1','my1','mz1','Ixx1',
'Ixy1','Iyy1','Ixz1', 'Iyz1','Izz1',
'm2', 'mx2','my2','mz2','Ixx2',
'Ixy2','Iyy2','Ixz2', 'Iyz2','Izz2']
std_table = [params_full, phi]
print(tabulate(std_table))
# print('condition number of regressor W: ', np.linalg.cond(W))
##################
print('*******************************************')
##################
print(dir(model.inertias[1]))
print(model.inertias[2]) # inertial matrix  in  vector and matrix forms

#eliminate columns crspnd. non-contributing parameters
params_e = []
params_r = [] 

tol_e = 1e-16
diag_W2 = np.diag(np.dot(W.T,W)) 
phi_e = np.array([])
for i in range(diag_W2.shape[0]):
	if diag_W2[i] < tol_e:
		phi_e = np.append(phi_e, i)
		params_e.append(params_full[i])
	else:
		params_r.append(params_full[i])
# phi_e = np.where(diag_W2 < tol_e, diag_W2, np.delete(diag_W2,,1))
phi_e = phi_e.astype(int)
W_e = np.delete(W, phi_e, 1)
npb = len(params_r)
print('eliminated parameters: ',params_e)

#QR decompostion, rank revealing
params_rsorted = []

epsilon = np.finfo(float).eps# machine epsilon
Q, R , P= linalg.qr(W_e, pivoting=True)
# print(np.diag(R))
# print(Q.shape[0])
for ind in P: 
	params_rsorted.append(params_r[ind])
print('reduced parameters sorted:',params_rsorted)
# print(tabulate([W_e[7, :],W_e[8,:],W_e[9,:],W_e[10,:]]))
tolpal = W_e.shape[0]*abs(np.diag(R).max())*epsilon#rank revealing tolerance
for i in range(np.diag(R).shape[0]):
	if abs(np.diag(R)[i]) < tolpal:
		numrank_W = i
R1 = R[0:numrank_W,0:numrank_W]
Q1 = Q[:,0:numrank_W]
R2 = R[0:numrank_W,numrank_W:npb]
beta = np.round(np.dot(np.linalg.inv(R1),R2),6)
# beta = np.where(beta > tol_e, beta, 0)#remove small beta close to zero 
phi_b = np.round(np.dot(np.linalg.inv(R1),np.dot(Q1.T,TAU)),6)
W_b = np.dot(Q1,R1)
params_idp = params_rsorted[:numrank_W]
params_rgp = params_rsorted[numrank_W]
# print('idpnd. parameters: ', params_idp)
# print('regrouped parameters: ', params_rgp)
print('condition number of base regressor: ',np.linalg.cond(W_b))
U, S, VT = np.linalg.svd(W_b)
print('singular values of base regressor:', S)
##################
print('*******************************************')
##################

params_base = []
for i in range(numrank_W):
	if beta[i] == 0:	
		params_base.append(params_idp[i])
		
	else:
		params_base.append(params_idp[i] + ' + '+str(round(float(beta[i]),6)) + '*'+ str(params_rgp))

print('base parameters and their identified values: ')
table = [params_base, phi_b]
print(tabulate(table))
# 

# #calculate joint torque by estimating from regressor and model params
# TAU_est = np.dot(W,phi)
# tau1_est = np.array([TAU_est[6]])
# for i in range(999):
# 	tau1_est = np.append(tau1_est, TAU_est[8*(i+1)-2 + 8])
# tau2_est = np.array([TAU_est[7]])
# for j in range(999):
# 	tau2_est = np.append(tau2_est, TAU_est[8*(j+1)-1 + 8])

#plot measured tau and estimated joint2 
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(TAU, TAU_est)
# m, b = np.polyfit(tau2, tau2_est,1)
# print(m, b)
# plt.plot(tau2, m*tau2 + b,'r', label='y={:.2f}x+{:.2f}'.format(m,b))
# plt.xlabel("Joint torque 2 by rnea() function")
# plt.ylabel("Joint torque 2 estimated by regressor matrix")
# plt.legend()
# plt.grid()
# plt.show()



#display 
dt = 1e-2
if VISUALIZER:
	
	robot.setVisualizer(VISUALIZER())
	robot.initViewer()
	robot.loadViewerModel("pinocchio")
	for k in range(q.shape[0]):
		t0 = time.time()
		robot.display(q[k,:])
		t1 = time.time()
		elapsed_time = t1 - t0
		if elapsed_time < dt:
			time.sleep(dt - elapsed_time)
	