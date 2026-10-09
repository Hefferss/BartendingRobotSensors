import numpy as np
import swift
from roboticstoolbox import DHLink, DHRobot, jtraj
from math import pi
from ir_support import CylindricalDHRobotPlot
import os
from spatialmath.base import ellipsoid, trotx, trotz, troty, transl
from spatialmath import SE3
from spatialgeometry import Mesh

#COLLISION STUFF BELOW. based mostly off lab 5 question 1 and lab 6 q2 elipsoids. using elipsoid method

#create an elipsoid. 
elipsoid_fatness = 0.1 

def elipsoids_for_robot_around_links(robot,q): #will be called every time j traj runs for this robot. Idea here is to make an elipsoid for each link and then use this to attach them to Q's to check for collisions.
    Joint_Pos_Rotation = [T.A for T in robot.fkine_all(q)] #same way as in lab 6 q2. gets a 4x4 matrix for each joint so we can work out positoin adn rotation. get_transforms from lab 6.
    Ellipsoid_all = [] # makes an empty array for the elipsoids to be added to. same idea as week 6 q2.
    for i in range(len(Joint_Pos_Rotation)-1): #for each link (between joint i and i+1):
            Elipsoid_vector = Joint_Pos_Rotation[i+1][:3,3] - Joint_Pos_Rotation[i][:3,3] #arrow from this joint to the next, along the link.
            Elipsoid_width = np.array([-Elipsoid_vector[1], Elipsoid_vector[0], 0]) #left and right of the link
            if np.linalg.norm(Elipsoid_width) == 0: #link 5 has no length, so the width is zero. make it a flat direction instead of zero.
                Elipsoid_width = np.array([1, 0, 0]) #any flat direction is sideways to a vertical link
            Elipsoid_height = np.cross(Elipsoid_vector, Elipsoid_width) #up and down from the link
            Elipsoid_centre = (Joint_Pos_Rotation[i+1][:3,3] + Joint_Pos_Rotation[i][:3,3])/2 #middle of the link, where the elipsoid sits. halfway between the two joints. Lab 6 2.5
            
            if np.linalg.norm(Elipsoid_vector) != 0:
                Elipsoid_direction = Elipsoid_vector / np.linalg.norm(Elipsoid_vector) #along the link, length 1
            else:
                Elipsoid_direction = Elipsoid_vector #link with no length, leave it as zeros. This is for link 5, because it doesnt have a length. just a direction.
            if np.linalg.norm(Elipsoid_width) != 0:
                Elipsoid_width = Elipsoid_width / np.linalg.norm(Elipsoid_width)
            if np.linalg.norm(Elipsoid_height) != 0:
                Elipsoid_height = Elipsoid_height / np.linalg.norm(Elipsoid_height)
                
            Elipsoid_rotation_matrix = np.column_stack((Elipsoid_direction, Elipsoid_width, Elipsoid_height)) #which way the elipsoid points. lab 6 2.7. makes it so the matrix lines up the long axis with the link
            Elipsoid_size = np.diag(np.square([np.linalg.norm(Elipsoid_vector/2), elipsoid_fatness, elipsoid_fatness])) #radii squared on the diagonal. lab 6 2.1
            Elipsoid_size_rotation = Elipsoid_rotation_matrix @ Elipsoid_size @ Elipsoid_rotation_matrix.T #size and direction packed into one 3x3. lab 6 2.7
            Ellipsoid_all.append({'matrix': Elipsoid_size_rotation, 'center': Elipsoid_centre}) #save this link's elipsoid. lab 6 2.10
    return Ellipsoid_all #give the list back once every link is done. updates it to Ellispod_all every time jtraj runs.


def elipsoid_parameters_for_robots(elipsoid_info): #takes the elipsoid info and makes it into a 3D mesh for swift. uses the ellipsoid method from week 4. 
    eigen_values, Eigen_vectors = np.linalg.eigh(elipsoid_info['matrix']) #taken from IR support in week 6. splits it into eigenvalues and eigenvectors so that they can be used:
    Elipsoid_rotation = Eigen_vectors #uses the 3 vectors to rotate the elipsoid so it matches up with the link. same as in lab 6 q2.
    if np.linalg.det(Elipsoid_rotation) < 0: #if it comes out negative, flip one of the directions so it is positive. same as in lab 6 q2.
        Elipsoid_rotation[:, 0] = -Elipsoid_rotation[:, 0] #flip one direction, elipsoid is the same either way
    Elipsoid_radius_xyz = np.sqrt(eigen_values) #takes teh eigenvalues and makes them into the radii of the elipsoid. same as in lab 6 q2.
    return elipsoid_info['center'], Elipsoid_rotation, Elipsoid_radius_xyz

#Notes for me when im not sleepy:
# - Need to draw the elipsoids in swift so i can check them
# - Need to make sure they update when the robot moves, so they are always in the right place
# - Need to add an obstacle and check that this robot works
# - need to code a safety solution and brainstorm how i can make it work properly when there are obstacles
            