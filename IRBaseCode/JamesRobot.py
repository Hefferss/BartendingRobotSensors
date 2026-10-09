# Required libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import swift
import roboticstoolbox as rtb
from spatialmath.base import *
from spatialmath import SE3
from roboticstoolbox import models, jtraj, trapezoidal
from spatialgeometry import Mesh
from ir_support.robots import UR3
from spatialmath.base import *
from roboticstoolbox import DHLink, DHRobot, models
from ir_support import CylindricalDHRobotPlot
from ir_support_extra_robots.robots import Turtlebot3Waffle
from ir_support_extra_parts.parts import part_names, part_mesh
from math import pi

# -----------------------------------------------------------------------------------#
class Assignment2():
    def __init__(self):
        print("ROBOT INITIATE")


    # def meshIN(self):

  

    def Arm(self):
        """
        ABB IRB 140 DH MATRIX 
        """

        link1 = DHLink(d=0.352, a=0.0000, alpha=0, qlim=[-pi, pi]) # DH perameters d = offset a = link length A = relative rotation theta = varibale angle 
        link2 = DHLink(d=0.0, a=-0.0700, alpha=-pi/2, qlim=[-pi, pi]) 
        link3 = DHLink(d=0.0, a=0.3600, alpha=0, qlim=[-pi, pi])
        link4 = DHLink(d=0.380, a=0.0000, alpha=-pi/2, qlim=[-pi, pi])
        link5 = DHLink(d=0.0780, a=0.0000, alpha=pi/2, qlim=[-pi, pi])
        link6 = DHLink(d=0.0650, a=0.0000, alpha=-pi/2, qlim=[-pi, pi])
        robot = DHRobot([link1, link2, link3, link4, link5, link6], name='ABB IRB 140')

        robot = DHRobot([link1, link2, link3, link4, link5, link6], name='ABB IRB 140')
        q = np.array([0, pi/2, -pi/2, 0, 0, 0]) #vertical pose, same as the stls

        self.env = swift.Swift()
        self.env.launch(realtime=True)
        self.env.add(robot)
        robot.q = q
        self.env.step()

        #mesh attached to robot inspired by the pen mesh import in week 4
        mesh_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IRB_Robot_Files")
        link_check = robot.fkine_all(q) #checks all locations of the robot and lists. used to update position each time. maybe theres a better way?

        #Base link
        base_location = os.path.join(mesh_folder, "base.stl")
        base_mesh = Mesh(filename=base_location, color="#39FF14", scale=[0.001, 0.001, 0.001]) #red because its the same as my last one and it was ugly as gray.
        base_mesh.T = link_check[0].A @ transl(-0.31, -0.2, 0)

        #link 1
        link_1_location = os.path.join(mesh_folder, "link 1.stl")
        link_1_mesh = Mesh(filename=link_1_location, color="#39FF14", scale=[0.001, 0.001, 0.001])
        link_1_mesh.T = link_check[1].A @ transl(-0.15, -0.13, -0.345)

        #link 2
        link_2_location = os.path.join(mesh_folder, "link 2.stl")
        link_2_mesh = Mesh(filename=link_2_location, color="#39FF14", scale=[0.001, 0.001, 0.001])
        link_2_mesh.T = link_check[2].A @ transl(-0.105, 0.35, 0.05) @ trotz(-pi/2) @ troty(pi/2)

        #link 3
        link_3_location = os.path.join(mesh_folder, "link 3.stl")
        link_3_mesh = Mesh(filename=link_3_location, color="#39FF14", scale=[0.001, 0.001, 0.001])
        link_3_mesh.T = link_check[3].A @ transl(-0.7, 0.17, -0.2) @ troty(pi/2) @ trotz(pi)

        #link 4
        link_4_location = os.path.join(mesh_folder, "link 4.stl")
        link_4_mesh = Mesh(filename=link_4_location, color="#39FF14", scale=[0.001, 0.001, 0.001])
        link_4_mesh.T = link_check[4].A @ transl(0.725, 0.5, 0.025) @ trotx(pi/2) @ troty(-pi/2)

        #link 5
        link_5_location = os.path.join(mesh_folder, "link 5.stl")
        link_5_mesh = Mesh(filename=link_5_location, color="#39FF14", scale=[0.001, 0.001, 0.001])
        link_5_mesh.T = link_check[5].A @ transl(-0.43, -0.005, -0.76) 

        #link 6 
        link_6_location = os.path.join(mesh_folder, "link 6.stl")
        link_6_mesh = Mesh(filename=link_6_location, color="#39FF14", scale=[0.001, 0.001, 0.001])
        link_6_mesh.T = link_check[6].A @transl(-0.5, -0.5, -0.72)


        self.env.add(base_mesh)
        self.env.add(link_1_mesh)
        self.env.add(link_2_mesh)
        self.env.add(link_3_mesh)
        self.env.add(link_4_mesh)
        self.env.add(link_5_mesh)
        self.env.add(link_6_mesh)
        self.env.step()




        cyl_viz = CylindricalDHRobotPlot(robot, cylinder_radius=0.05, color="#702963")
        robot = cyl_viz.create_cylinders()
        self.env.add(robot)


        

        # goal_pose = SE3(0.4, 0, 0) #hadnerover point to the ur3 for the turtle base
        # ur3goal_pose = SE3(0.44, 0, 0.84) #handoever location for the beer bottle
        # robotgoal_pose = robot.base * SE3(0, 0, 0.7) #turtle base location hadnover + vertical offset for ur3 handover
        # final_pose = SE3(0, -0.7, 0.6) #table position final placement for the beer 

        # steps = 100


        # q_pickup = robot.ikine_LM(SE3(1.3, 1, 0.2)).q #ikine given the end effector pose calculate joint angles 
        # traj1 = rtb.jtraj(robot.q, q_pickup, steps).q #IK has multiple solutions and joint states or none 

        # base_traj = rtb.ctraj(start_pose, goal_pose, steps) #traj run interpolation algorithm 

        # q_dropoff = robot.ikine_LM(robotgoal_pose).q
        # traj2 = rtb.jtraj(q_pickup, q_dropoff, steps).q #dont use robot.q teleports back

        # q_ur3pickup = self.ur3.ikine_LM(ur3goal_pose).q
        # traj3 = rtb.jtraj(self.ur3.q, q_ur3pickup, steps).q

        # q_ur3dropoff = self.ur3.ikine_LM(final_pose).q
        # traj4 = rtb.jtraj(q_ur3pickup, q_ur3dropoff, steps).q # had to change start pos to pickup as the ur3 would snap to default


        
        # #arm picks up beer 
        # for q in traj1:
        #     robot.q = q
        #     self.env.step(0.02)

        # # beer, arm and bot move towards ur3
        # for q in base_traj:
        #     self.turtle.base = q
        #     robot.base = self.turtle.base * offset   # arm offset for no clipping 
        #     self.beer.T = robot.fkine(robot.q).A * beeroffest # fkine given joint angles where the end effector is just DH transforms
        #     self.env.step(0.02)

        # for i in range(steps):
        #     robot.q = traj2[i]
        #     self.beer.T = robot.fkine(robot.q).A * beeroffest
        #     self.ur3.q = traj3[i]
        #     self.env.step(0.02)

        # for q in traj4:
        #     self.ur3.q = q
        #     self.beer.T = self.ur3.fkine(self.ur3.q).A * beeroffest
        #     self.env.step(0.02)


        self.env.hold()





# ---------------------------------------------------------------------------------------#
# Main block
if __name__ == "__main__":
    soln = Assignment2()
    # soln.meshIN()
    soln.Arm()

    print("Assignment completed")
    plt.close("all")
    time.sleep(0.5)
