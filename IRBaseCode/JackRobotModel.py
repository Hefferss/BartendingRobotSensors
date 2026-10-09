import numpy as np
import swift
from roboticstoolbox import DHLink, DHRobot, jtraj
from math import pi
from ir_support import CylindricalDHRobotPlot
import os
from spatialmath.base import ellipsoid, trotx, trotz, troty, transl, rt2tr
from spatialmath import SE3
from spatialgeometry import Mesh
from CollisionEllipsoid import elipsoids_for_robot_around_links, elipsoid_parameters_for_robots

#links for the staubli. the qlims are specific. yay.
link1 = DHLink(d=0.375, a=0, alpha=pi/2, qlim=[-pi, pi]) #0.375 from sheet, lines up with stl now mesh transforms are right
link2 = DHLink(d=0, a=0.29, alpha=0, qlim=[-127.5*pi/180, 127.5*pi/180])
link3 = DHLink(d=-0.02, a=0, alpha=-pi/2, qlim=[-142.5*pi/180, 142.5*pi/180]) #negative so offset is same side as stl
link4 = DHLink(d=0.31, a=0, alpha=pi/2, qlim=[-270*pi/180, 270*pi/180])
link5 = DHLink(d=0, a=0, alpha=-pi/2, qlim=[-121*pi/180, 132.5*pi/180])
link6 = DHLink(d=0.07, a=0, alpha=0, qlim=[-270*pi/180, 270*pi/180])

robot = DHRobot([link1, link2, link3, link4, link5, link6], name='Staubli TX60')
q = np.array([0, pi/2, -pi/2, 0, 0, 0]) #vertical pose, same as the stls

#test cylinders to add to robot. comment out later. same as A1
#cyl_viz = CylindricalDHRobotPlot(robot, cylinder_radius=0.025, color="#7b1d1d")
#robot = cyl_viz.create_cylinders()

#sets up environment and robot.
env = swift.Swift()
env.launch(realtime=True)
env.add(robot)
robot.q = q
env.step()

#mesh attached to robot inspired by the pen mesh import in week 4
mesh_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Staubli Robot Files")
link_check = robot.fkine_all(q) #checks all locations of the robot and lists. used to update position each time. maybe theres a better way?

#Base link
base_location = os.path.join(mesh_folder, "base_link.stl")
base_mesh = Mesh(filename=base_location, color="#6e6e6e") #red because its the same as my last one and it was ugly as gray.
base_mesh.T = link_check[0].A

#link 1
link_1_location = os.path.join(mesh_folder, "link_1.stl")
link_1_mesh = Mesh(filename=link_1_location, color="#6e6e6e")
link_1_mesh.T = link_check[1].A @ trotx(-pi/2)

#link 2
link_2_location = os.path.join(mesh_folder, "link_2.stl")
link_2_mesh = Mesh(filename=link_2_location, color="#6e6e6e")
link_2_mesh.T = link_check[2].A @ transl(-0.29, 0, 0) @ trotz(-pi/2) @ trotx(-pi/2) #dh frame is at elbow, stl origin at shoulder

#link 3
link_3_location = os.path.join(mesh_folder, "link_3.stl")
link_3_mesh = Mesh(filename=link_3_location, color="#6e6e6e")
link_3_mesh.T = link_check[3].A

#link 4
link_4_location = os.path.join(mesh_folder, "link_4.stl")
link_4_mesh = Mesh(filename=link_4_location, color="#6e6e6e")
link_4_mesh.T = link_check[4].A @ transl(0, -0.31, 0) @ trotx(-pi/2) #according to the schematics, its at the wrist point so it needs to be translatsed and offset. 4th adjustment. 0-.25, -0.3, -0.325. -0.32.

#link 5
link_5_location = os.path.join(mesh_folder, "link_5.stl")
link_5_mesh = Mesh(filename=link_5_location, color="#6e6e6e")
link_5_mesh.T = link_check[5].A 

#link 6 
link_6_location = os.path.join(mesh_folder, "link_6.stl")
link_6_mesh = Mesh(filename=link_6_location, color="#6e6e6e")
link_6_mesh.T = link_check[6].A #no transform needed, lines up already


env.add(base_mesh)
env.add(link_1_mesh)
env.add(link_2_mesh)
env.add(link_3_mesh)
env.add(link_4_mesh)
env.add(link_5_mesh)
env.add(link_6_mesh)
env.step()
input("Enter to continue\n")

#Apply Collision detection mesh
#draw the elipsoids in swift. visual only, not needed for the collision check
elipsoid_transparency = 0.25
sphere_location = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Scene_parts", "CollisionSphere.stl")
elipsoid_meshes = [] #keeps the drawn elipsoids so they can be moved later
for elipsoid_info in elipsoids_for_robot_around_links(robot, q): #for each link....
    centre, rotation, radii = elipsoid_parameters_for_robots(elipsoid_info) #where, which way, how big. taken from the elipsoid info stuff above. makes it into a 3D mesh for swift. uses the ellipsoid method from week 4.
    elipsoid_mesh = Mesh(filename=sphere_location, scale=radii/1000, color="#41b3f9") #stretches sphere by teh radii, and also makes it blue. lab 6 2.1 and 2.2
    elipsoid_mesh.set_alpha(elipsoid_transparency) #makes it transparent so you can see the robot through it. 
    elipsoid_mesh.T = rt2tr(rotation, centre) #places it, rotation + centre in one 4x4. same stuff as lab 2
    env.add(elipsoid_mesh)
    elipsoid_meshes.append(elipsoid_mesh) #save it in the list
env.step()
input("Enter to continue\n")

#make the robot move, with the link meshes following
q_goal = np.array([-pi/2, 0, -pi/4, 0, 0, 0]) #arm out to the side. just an arbitrary pose to test the meshes. 
for q_step in jtraj(q, q_goal, 50).q: #50 steps from vertical to q_goal. 
    robot.q = q_step
    link_check = robot.fkine_all(q_step) #where every joint is at this step
    base_mesh.T = link_check[0].A
    link_1_mesh.T = link_check[1].A @ trotx(-pi/2)
    link_2_mesh.T = link_check[2].A @ transl(-0.29, 0, 0) @ trotz(-pi/2) @ trotx(-pi/2)
    link_3_mesh.T = link_check[3].A
    link_4_mesh.T = link_check[4].A @ transl(0, -0.31, 0) @ trotx(-pi/2)
    link_5_mesh.T = link_check[5].A
    link_6_mesh.T = link_check[6].A
    elipsoids = elipsoids_for_robot_around_links(robot, q_step) #where the elipsoids should be at this step
    for i in range(len(elipsoid_meshes)): #for each drawn elipsoid
        centre, rotation, radii = elipsoid_parameters_for_robots(elipsoids[i]) #where, which way, how big
        elipsoid_meshes[i].T = rt2tr(rotation, centre) #move it there
    env.step(0.05)
input("Enter to continue\n")