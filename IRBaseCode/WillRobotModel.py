import numpy as np
import swift
from roboticstoolbox import DHLink, DHRobot, jtraj
from math import pi
from ir_support import CylindricalDHRobotPlot
import os
from spatialgeometry import Mesh
from spatialmath.base import trotx, trotz, troty, transl
from spatialmath import SE3

# Z is d, X is a

#Links for the Kawasaki RS007N.                                                              Specifications from Kawasaki Documentation pg23
link1 = DHLink(d=0.36, a=0, alpha=pi/2, qlim=[-180*pi/180, 180*pi/180])                         # JT1, "Arm Rotation"
link2 = DHLink(d=0, a=0.355, alpha=0, qlim=[-135*pi/180, 135*pi/180])                            # JT2, "Arm Out-in"
link3 = DHLink(d=0, a=0, alpha=-pi/2, qlim=[-155*pi/180, 155*pi/180])                       # JT3, "Arm Up-down"
link4 = DHLink(d=0.375, a=0, alpha=pi/2, qlim=[-200*pi/180, 200*pi/180])                         # JT4, "Wrist Swivel"
link5 = DHLink(d=0, a=0, alpha=-pi/2, qlim=[-125*pi/180, 125*pi/180])                        # JT5, "Wrist Bend"
link6 = DHLink(d=0.078, a=0, alpha=0, qlim=[-360*pi/180, 360*pi/180])                            # JT6, "Wrist Twist"

robot = DHRobot([link1, link2, link3, link4, link5, link6], name='Kawasaki RS007N')
q = np.array([0, pi/2, -pi/2, 0, -pi/4, 0]) #vertical pose, same as the stls
#q = np.array([0, 0, 0, 0, 0, 0]) #default pose

#test cylinders to add to robot. comment out later. same as A1
cyl_viz = CylindricalDHRobotPlot(robot, cylinder_radius=0.025, color="#7b1d1d")
robot = cyl_viz.create_cylinders()

#sets up environment and robot.
env = swift.Swift()
env.launch(realtime=True)
env.add(robot)
#robot.base = transl(-0.15, -0.2675, 0)              #Set the base to the origins
robot.q = q
env.step()

#mesh attached to robot inspired by the pen mesh import in week 4
mesh_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Kawasaki_Robot_Files")
link_check = robot.fkine_all(q) #checks all locations of the robot and lists. used to update position each time. maybe theres a better way?



#Base link Purple: 893BFF
base_location = os.path.join(mesh_folder, "wBase.stl")
base_mesh = Mesh(filename=base_location, color="#5B5B5B8D", scale=[0.001, 0.001, 0.001]) 
base_mesh.T = link_check[0].A @ trotz(-pi/2) @ transl(-0.15, -0.2675, 0) 

#link 1
link_1_location = os.path.join(mesh_folder, "wLink1.stl")
link_1_mesh = Mesh(filename=link_1_location, color="#5B5B5B8D", scale=[0.001, 0.001, 0.001])
link_1_mesh.T = link_check[1].A @ trotx(-pi/2) @ trotz(-pi/2) @ transl(-0.15, -0.2675, -0.36) 

#link 2
link_2_location = os.path.join(mesh_folder, "wLink2.stl")
link_2_mesh = Mesh(filename=link_2_location, color="#5B5B5B8D", scale=[0.001, 0.001, 0.001])
link_2_mesh.T = link_check[2].A @ trotx(-pi/2) @ troty(pi/2) @ trotz(-pi/2) @ transl(-0.15, -0.2675, -0.715)

#link 3
link_3_location = os.path.join(mesh_folder, "wLink3.stl")
link_3_mesh = Mesh(filename=link_3_location, color="#5B5B5B8D", scale=[0.001, 0.001, 0.001])
link_3_mesh.T = link_check[3].A @ trotz(-pi/2) @ transl(-0.15, -0.2675, -0.715)

#link 4
link_4_location = os.path.join(mesh_folder, "wLink4.stl")
link_4_mesh = Mesh(filename=link_4_location, color="#5B5B5B8D", scale=[0.001, 0.001, 0.001])
link_4_mesh.T = link_check[4].A @ trotx(-pi/2) @ trotz(-pi/2) @ transl(-0.15, -0.2675, -1.090)

#link 5
link_5_location = os.path.join(mesh_folder, "wLink5.stl")
link_5_mesh = Mesh(filename=link_5_location, color="#5B5B5B8D", scale=[0.001, 0.001, 0.001])
link_5_mesh.T = link_check[5].A @ trotz(-pi/2) @ transl(-0.15, -0.2675, -1.090)

#link 6 
link_6_location = os.path.join(mesh_folder, "wLink6.stl")
link_6_mesh = Mesh(filename=link_6_location, color="#5B5B5B8D", scale=[0.001, 0.001, 0.001])
link_6_mesh.T = link_check[6].A @ trotz(-pi/2) @ transl(-0.15, -0.2675, -1.168)

env.add(base_mesh)
env.add(link_1_mesh)
env.add(link_2_mesh)
env.add(link_3_mesh)
env.add(link_4_mesh)
env.add(link_5_mesh)
env.add(link_6_mesh)
env.step()
input("Enter to continue\n")

T_target = SE3(-0.2, 0.1, 0.5)* SE3.Rx(pi)

ik_sol = robot.ikine_LM(T_target)

if not ik_sol.success:
    raise ValueError("Inverse kinematics failed to find a valid solution for the target pose!")

q_goal = ik_sol.q
print("IK Solution found (radians):", q_goal)

traj = jtraj(robot.q, q_goal, 50)

for q_step in traj.q:
    robot.q = q_step
    
    link_check = robot.fkine_all(q_step)
    
    base_mesh.T = link_check[0].A @ trotz(-pi/2) @ transl(-0.15, -0.2675, 0) 
    link_1_mesh.T = link_check[1].A @ trotx(-pi/2) @ trotz(-pi/2) @ transl(-0.15, -0.2675, -0.36) 
    link_2_mesh.T = link_check[2].A @ trotx(-pi/2) @ troty(pi/2) @ trotz(-pi/2) @ transl(-0.15, -0.2675, -0.715)
    link_3_mesh.T = link_check[3].A @ trotz(-pi/2) @ transl(-0.15, -0.2675, -0.715)
    link_4_mesh.T = link_check[4].A @ trotx(-pi/2)  @ trotz(-pi/2) @ transl(-0.15, -0.2675, -1.090)
    link_5_mesh.T = link_check[5].A @ trotz(-pi/2) @ transl(-0.15, -0.2675, -1.090)
    link_6_mesh.T = link_check[6].A @ trotz(-pi/2) @ transl(-0.15, -0.2675, -1.168)
    
    env.step()

input("Enter to continue\n")