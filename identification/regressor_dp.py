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

#Load the URDF model with RobotWrapper
def loadModels(robotname, robot_urdf_file):
	# print("Load model with RobotWrapper")
	pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))),"models")
	model_path = join(pinocchio_model_dir,"others/robots")
	mesh_dir = model_path
	urdf_filename = robot_urdf_file
	urdf_dir = robotname + "/urdf"
	urdf_model_path = join(join(model_path,urdf_dir),urdf_filename)	
	robot = RobotWrapper.BuildFromURDF(urdf_model_path, mesh_dir)
	return robot
	
#inertial parameters of link2 from urdf model
def standardParameters(njoints):
	# print("Standard parameters")
	params_name = ['m', 'mx','my','mz','Ixx',
'Ixy','Iyy','Ixz', 'Iyz','Izz']
	phi = []
	params = []
	for i in range(1,njoints):
		P = model.inertias[i].toDynamicParameters()
		for k in P: 
			phi.append(k)
		for j in params_name: 
			params.append(j + str(i))
	if isFrictionincld:
		for k in range(1, njoints):
			phi.extend([fv,fc])
			params.extend(['fv' + str(k),'fc' + str(k)])
	params_std = dict(zip(params, phi)) 
	return params_std

#generate waypoints and identification model 
def generateWaypoints(N,nq,nv,mlow,mhigh): 
	# print('generate waypoints')
	q = np.empty((1,nq))
	v = np.empty((1,nv))
	a = np.empty((1,nv))
	for i in range(N):
		q = np.vstack((q,np.random.uniform(low = mlow, high = mhigh, size=(nq,))))		
		v = np.vstack((v,np.random.uniform(low = mlow, high = mhigh, size=(nq,))))
		a = np.vstack((a,np.random.uniform(low = mlow, high = mhigh, size=(nq,))))
	return q, v, a

#Building regressor
def  iden_model(N, nq, q, v, a): 
	# print("Creat identification model")
	tau = np.empty(nq*N)
	W = np.empty([N*nq, 10*nq]) 
	for i in range(N):
		tau_temp = pin.rnea(model, data, q[i,:], v[i,:], a[i,:])
		W_temp = pin.computeJointTorqueRegressor(model, data, q[i,:], v[i,:], a[i,:])
		for j in range(nq):
			tau[j*N + i] = tau_temp[j]
			W[j*N + i, :] = W_temp[j,:]
	if isFrictionincld:
		W = np.c_[W,np.zeros([N*nq,2*nq])]
		for i in range(N):
			for j in range(nq):
				tau[j*N + i] = tau[j*N + i] + v[i,j]*fv + np.sign(v[i,j])*fc
				W[j*N + i, 10*nq+2*j] = v[i,j]
				W[j*N + i, 10*nq+2*j + 1] = np.sign(v[i,j])
	return tau, W

#Eliminate columns crspnd. non-contributing parameters
def eliminateNonAffecting(W, tol_e):
	# print('Eliminate columns that has affecting to dynamics')
	col_norm = np.diag(np.dot(W_.T,W_))
	idx_e = []
	params_e = []
	params_r = []
	for i in range(col_norm.shape[0]):
		if col_norm[i] < tol_e: 
			idx_e.append(i)
			params_e.append(list(params_std.keys())[i])
		else: 
			params_r.append(list(params_std.keys())[i])
	W_e = np.delete(W_, idx_e, 1)
	return W_e, params_r

#QR decompostion, rank revealing
def QR_pivoting(W_e):
	# print('QR_pivoting')
	Q, R, P = linalg.qr(W_e, pivoting = True) #scipy has pivoting
	# sort params as decreasing order of diagonal of R 
	params_rsorted = []
	for ind in P: 
		params_rsorted.append(params_r[ind])
	#find rank of regressor
	numrank_W = 0
	epsilon = np.finfo(float).eps# machine epsilon
	tolpal = W_e.shape[0]*abs(np.diag(R).max())*epsilon#rank revealing tolerance
	for i in range(np.diag(R).shape[0]):
		if abs(np.diag(R)[i]) < tolpal:
			numrank_W = i
	#regrouping, calculating base params, base regressor
	R1 = R[0:numrank_W,0:numrank_W]
	Q1 = Q[:,0:numrank_W]
	R2 = R[0:numrank_W,numrank_W:R.shape[1]]
	beta = np.round(np.dot(np.linalg.inv(R1),R2),6)#regrouping coefficient
	phi_b = np.round(np.dot(np.linalg.inv(R1),np.dot(Q1.T,tau_)),6)#values of base params
	W_b = np.dot(Q1,R1)#base regressor
	params_idp = params_rsorted[:numrank_W]
	params_rgp = params_rsorted[numrank_W]
	params_base = []
	for i in range(numrank_W):
		if beta[i] == 0:	
			params_base.append(params_idp[i])
			
		else:
			params_base.append(params_idp[i] + ' + '+str(round(float(beta[i]),6)) + '*'+ str(params_rgp))
	print('base parameters and their identified values: ')
	table = [params_base, phi_b]
	print(tabulate(table))
	return W_b, phi_b, numrank_W, params_rsorted

# print('idpnd. parameters: ', params_idp)
# print('regrouped parameters: ', params_rgp)


#display 
# If you want to visualize the robot in this example,
# you can choose which visualizer to employ
# by specifying an option from the command line:
# GepettoVisualizer: -g
# MeshcatVisualizer: -m

def visualization(): 
	print('visualization')
	pass

VISUALIZER = None
if len(argv)>1:
	opt = argv[1]
	if opt == '-g':
		VISUALIZER = GepettoVisualizer
	elif opt == '-m':
		VISUALIZER = MeshcatVisualizer
	#else:
	#    raise ValueError("Unrecognized option: " + opt)
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

isFrictionincld = False	
fv = 0.05
fc = 0.01
robot = loadModels("2DOF_description", "2DOF_description.urdf")
model = robot.model
data = robot.data	
nq, nv , njoints = model.nq, model.nv, model.njoints
#numbers of samples
N = 1000
params_std = standardParameters(njoints)
print("Standard inertial parameters: ",params_std)
print("###########################")
q , qd, qdd = generateWaypoints(N, nq, nv, -10, 10)
tau_, W_ = iden_model(N, nq, q, qd, qdd)
W_e, params_r = eliminateNonAffecting(W_, 1e-6)
W_b, phi_b, numrank_W, params_rsorted = QR_pivoting(W_e)
print("###########################")
print('condition number of base regressor: ',np.linalg.cond(W_b))
U, S, VT = np.linalg.svd(W_b)
print('singular values of base regressor:', S)
# print(nq, nv)