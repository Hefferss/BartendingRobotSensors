import os
import numpy as np
from spatialgeometry import Mesh
from spatialmath import SE3
from roboticstoolbox import models
from roboticstoolbox import jtraj

item_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Scene_parts")

def add_UR3(env):
    ur3_base_location = os.path.join(item_folder, "BasePlatForUR3.stl")
    ur3_base_scene = Mesh(filename=ur3_base_location, color="#430F98", scale=[0.001, 0.001, 0.001])
    ur3_base_scene.T = SE3(-0.08, 0, -0.1)
    env.add(ur3_base_scene)

    ur3 = models.UR3()
    ur3.base = SE3(0, 0, 0)
    ur3.q = np.zeros(ur3.n)
    env.add(ur3)

    return ur3, ur3_base_scene

def UR3_move_base(env, ur3, ur3_base_mesh, target_x, steps=30): #moves the UR3 base along the x-axis to a target position over a number of steps. should lock on rail. need to add a thing that makes sure it doesnt slide off lol.
    start_x = ur3.base.t[0]
    move_step = (target_x - start_x) / steps

    for i in range(steps):
        ur3.base = ur3.base * SE3(move_step, 0, 0)
        ur3_base_mesh.T = ur3_base_mesh.T * SE3(move_step, 0, 0)
        env.step(0.05)

def UR3_move_and_grab(env, ur3, item, item_name="item", steps=30): #move the UR3 to item position and attach it to the end-effector. uses inverse kinematics to find the joint angles needed to reach the item. also attaches it. would be a seperate command but i think concantenating it like this would be a wise decision here.
    item_pos = item.T[0:3, 3] #gets item position
    target_pose = SE3(item_pos[0], item_pos[1], item_pos[2]) #gest xyz pos
    result = ur3.ikine_LM(target_pose, q0=ur3.q, mask=[1,1,1,0,0,0]) #usual
    q_matrix = jtraj(ur3.q, result.q, steps).q #stuff for the matrix

    for q in q_matrix:
        ur3.q = q
        env.step(0.05) 

    item.T = ur3.fkine(ur3.q).A  # attach item to end-effector
    env.step()

def set_UR3_pose(ur3, ur3_base_scene, pose):
    ur3_base_scene.T = pose
    ur3.base = pose * SE3(0.08, 0, 0.1)