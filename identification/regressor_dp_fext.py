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
	"""This function create a robot model and its data by inputting a URDF file that describes the robot.
		Input: 	robotname: directory containing model of robot
			robot_urdf_file: URDF file of robot
		Output: robot: robot model and data created by Pinocchio"""
	
	pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))),"models")
	model_path = join(pinocchio_model_dir,"others/robots")
	mesh_dir = model_path
	urdf_filename = robot_urdf_file
	urdf_dir = robotname + "/urdf"	
	urdf_model_path = join(join(model_path,urdf_dir),urdf_filename)	
	if not isFext: 
		robot = RobotWrapper.BuildFromURDF(urdf_model_path, mesh_dir)
	else: 
		robot = RobotWrapper.BuildFromURDF(urdf_model_path, mesh_dir, pin.JointModelFreeFlyer())
	return robot
	
#inertial parameters of link2 from urdf model
def standardParameters(njoints):
	"""This function prints out the standard inertial parameters obtained from 3D design.
		Note: a flag IsFrictioncld to include in standard parameters
		Input: 	njoints: number of joints
		Output: params_std: a dictionary of parameter names and their values"""
	params_name = ['m', 'mx','my','mz','Ixx','Ixy','Iyy','Ixz', 'Iyz','Izz']
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
	"""This function generates N random values for joints' position,velocity, acceleration.
		Input: 	N: number of samples
				nq: length of q, nv : length of v
				mlow and mhigh: the bound for random function
		Output: q, v, a: joint's position, velocity, acceleration"""
	q = np.empty((1,nq))
	v = np.empty((1,nv))
	a = np.empty((1,nv))
	for i in range(N):
		q = np.vstack((q,np.random.uniform(low = mlow, high = mhigh, size=(nq,))))		
		v = np.vstack((v,np.random.uniform(low = mlow, high = mhigh, size=(nv,))))
		a = np.vstack((a,np.random.uniform(low = mlow, high = mhigh, size=(nv,))))
	return q, v, a

#Building regressor
def  iden_model(model, data, N, nq, nv, njoints, q, v, a): 
	"""This function calculates joint torques and generates the joint torque regressor.
		Note: a flag IsFrictioncld to include in dynamic model
		Input: 	model, data: model and data structure of robot from Pinocchio
				q, v, a: joint's position, velocity, acceleration
				N : number of samples
				nq: length of q
		Output: tau: vector of joint torque
				W : joint torque regressor"""
	tau = np.empty(nv*N)
	W = np.empty([N*nv, 10*(njoints-1)]) 
	for i in range(N):
		tau_temp = pin.rnea(model, data, q[i,:], v[i,:], a[i,:])
		W_temp = pin.computeJointTorqueRegressor(model, data, q[i,:], v[i,:], a[i,:])
		for j in range(W_temp.shape[0]):
			tau[j*N + i] = tau_temp[j]
			W[j*N + i, :] = W_temp[j,:]
	if isFrictionincld:
		W = np.c_[W,np.zeros([N*nv,2*nv])]
		for i in range(N):
			for j in range(nv):
				tau[j*N + i] = tau[j*N + i] + v[i,j]*fv + np.sign(v[i,j])*fc
				W[j*N + i, 10*(njoints-1)+2*j] = v[i,j]
				W[j*N + i, 10*(njoints-1)+2*j + 1] = np.sign(v[i,j])
	return tau, W
def iden_model_fext():


	pass
#Eliminate columns crspnd. non-contributing parameters
def eliminateNonAffecting(W, tol_e):
	"""This function eliminates columns which has L2 norm smaller than tolerance.
		Input: 	W: joint torque regressor
				tol_e: tolerance 
		Output: W_e: reduced regressor
				params_r: corresponding parameters to columns of reduced regressor"""
	col_norm = np.diag(np.dot(W.T,W))
	idx_e = []
	params_e = []
	params_r = []
	for i in range(col_norm.shape[0]):
		if col_norm[i] < tol_e: 
			idx_e.append(i)
			params_e.append(list(params_std.keys())[i])
		else: 
			params_r.append(list(params_std.keys())[i])
	W_e = np.delete(W, idx_e, 1)
	return W_e, params_r

#QR decompostion, rank revealing
def QR_pivoting(W_e, params_r):
	"""This function calculates QR decompostion with pivoting, finds rank of regressor,
	and calculates base parameters
		Input: 	W_e: reduced regressor
				params_r: inertial parameters corresponding to W_e 
		Output: W_b: base regressor
				phi_b: values of base parameters
				numrank_W: numerical rank of regressor, determined by using a therehold
				params_rsorted: inertial parameters included in base parameters """
	Q, R, P = linalg.qr(W_e, pivoting = True) #scipy has QR pivoting
	# sort params as decreasing order of diagonal of R 
	params_rsorted = []
	for ind in P: 
		params_rsorted.append(params_r[ind])
	#find rank of regressor
	numrank_W = 0
	epsilon = np.finfo(float).eps# machine epsilon
	tolpal = W_e.shape[0]*abs(np.diag(R).max())*epsilon#rank revealing tolerance
	for i in range(np.diag(R).shape[0]):
		if abs(np.diag(R)[i]) > tolpal:
			continue
		else: 
			numrank_W = i
			break
	#regrouping, calculating base params, base regressor
	R1 = R[0:numrank_W,0:numrank_W]
	Q1 = Q[:,0:numrank_W]
	R2 = R[0:numrank_W,numrank_W:R.shape[1]]
	beta = np.round(np.dot(np.linalg.inv(R1),R2),6)#regrouping coefficient

	phi_b = np.round(np.dot(np.linalg.inv(R1),np.dot(Q1.T,tau)),6)#values of base params
	W_b = np.dot(Q1,R1)#base regressor
	params_base = params_rsorted[:numrank_W]
	params_rgp = params_rsorted[numrank_W:]
	for i in range(numrank_W):
		for j in range(beta.shape[1]):
			if beta[i,j] == 0:	
				params_base[i] = params_base[i]
			elif beta[i,j] < 0:
				params_base[i] = params_base[i] + ' - '+str(abs(beta[i,j])) + '*'+ str(params_rgp[j])
			else:
				params_base[i] = params_base[i] + ' + '+str(abs(beta[i,j])) + '*'+ str(params_rgp[j])
	print('base parameters and their identified values: ')
	base_parameters = dict(zip(params_base,phi_b))
	print(base_parameters)
	# table = [params_base, phi_b]
	# print(tabulate(table))
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

isFext = False

isFrictionincld = False
if len(argv)>1:
	if argv[1] == '-f':
		isFrictionincld = True
fv = 0.05
fc = 0.01
robot = loadModels("2DOF_description", "2DOF_description.urdf")
# robot = loadModels("SC_3DOF", "3DOF.urdf")
model = robot.model
print(model)
data = robot.data	
nq, nv , njoints = model.nq, model.nv, model.njoints
#numbers of samples
N = 1000
params_std = standardParameters(njoints)
print("Standard inertial parameters: ",params_std)
print("###########################")
q , qd, qdd = generateWaypoints(N, nq, nv, -1, 1)
tau, W = iden_model(model, data, N,  nq, nv,njoints, q, qd, qdd)
W_e, params_r = eliminateNonAffecting(W, 1e-6)
W_b, phi_b, numrank_W, params_rsorted = QR_pivoting(W_e, params_r)
print("###########################")
print('condition number of base regressor: ',np.linalg.cond(W_b))
U, S, VT = np.linalg.svd(W_b)
print('singular values of base regressor:', S)
# print(nq, nv)