from pinocchio.robot_wrapper import RobotWrapper
URDF = '/opt/openrobots/share/example-robot-data/robots/ur_description/urdf/ur5_gripper.urdf'
robot = RobotWrapper.BuildFromURDF(URDF)
for name, function in robot.model.__class__.__dict__.items():
    print(' **** %s: %s' % (name, function.__doc__))
    # Get index of end effector
idx = robot.index('wrist_3_joint')
# Compute and get the placement of joint number idx
placement = robot.placement(q, idx)
# Be carreful, Python always returns references to values.
# You can often .copy() the object to avoid side effects
# Only get the placement
placement = robot.data.oMi[idx].copy()
q = zero(robot.nq)
v = rand(robot.nv)
robot.com(q)  # Compute the robot center of mass.
robot.placement(q, 3)  # Compute the placement of joint 3
robot.initViewer(loadModel=True)
visualObj = robot.visual_model.geometryObjects[4]  # 3D object representing the robot forarm
visualName = visualObj.name                        # Name associated to this object
visualRef = robot.getViewerNodeName(visualObj, pin.GeometryType.VISUAL)    # Viewer reference (string) representing this object
q1 = (1, 1, 1, 1, 0, 0, 0)  # x, y, z, quaternion
robot.viewer.gui.applyConfiguration(visualRef, q1)
robot.viewer.gui.refresh()  # Refresh the window.