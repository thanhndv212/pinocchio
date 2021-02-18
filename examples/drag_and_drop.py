from os.path import join
import pinocchio as se3
from mobilerobot import MobileRobotWrapper
from pinocchio.utils import *
PKG = '/opt/openrobots/share'
URDF = join(PKG, 'ur5_description/urdf/ur5_gripper.urdf')
robot = MobileRobotWrapper(URDF, [PKG])
robot.initDisplay(loadModel=True)
# robot.viewer.gui.addFloor('world/floor')
NQ, NV = robot.model.nq, robot.model.nv
q = rand(NQ)
robot.display(q)
IDX_TOOL = 24
IDX_BASIS = 23
se3.framesKinematics(robot.model, robot.data)
Mtool = robot.data.oMf[IDX_TOOL]
Mbasis = robot.data.oMf[IDX_BASIS]
#The first task will be concerned with the end effector. First define a goal placement.
def place(name, M):
    robot.viewer.gui.applyConfiguration(name, se3ToXYZQUAT(M))
    robot.viewer.gui.refresh()
def Rquat(x, y, z, w):
    q = se3.Quaternion(x, y, z, w)
    q.normalize()
    return q.matrix()
Mgoal = se3.SE3(Rquat(.4, .02, -.5, .7), np.matrix([.2, -.4, .7]).T)
robot.viewer.gui.addXYZaxis('world/framegoal', [1., 0., 0., 1.], .015, 4)
place('world/framegoal', Mgoal)
#The current placement of the tool at configuration q is available as follows:
IDX_TOOL = 24
se3.forwardKinematics(robot.model, robot.data, q)  # Compute joint placements
se3.framesKinematics(robot.model, robot.data)      # Also compute operational frame placements
Mtool = robot.data.oMf[IDX_TOOL]  # Get placement from world frame o to frame f oMf

#The desired velocity of the tool in tool frame is given by the log:
nu = se3.log(Mtool.inverse() * Mgoal).vector

#The tool Jacobian, also in tool frame, is available as follows:
J = se3.frameJacobian(robot.model, robot.data, IDX_TOOL, q)

#Pseudoinverse operator is available in numpy.linalg toolbox.
from numpy.linalg import pinv

#The integration of joint velocity vq in configuration q can be done directly (q += vq * dt). More generically, the se3 method integrate can be used:
q = se3.integrate(robot.model, q, vq * dt)