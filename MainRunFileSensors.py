import os
import sys
import swift
import numpy as np
from spatialgeometry import Mesh
from math import pi
from spatialmath import SE3
from spatialmath.base import trotz

scene_folder = os.path.dirname(os.path.abspath(__file__))
ir_folder = os.path.join(scene_folder, "IRBaseCode") #UR3.py and the rail live in the IR code folder
sys.path.append(ir_folder)
from UR3 import add_UR3, UR3_move_base, UR3_move_and_grab, set_UR3_pose, update_camera_mesh

env = swift.Swift()
env.launch(realtime=True)

#UR3 rail
ur3_rail_location = os.path.join(ir_folder, "Scene_parts", "RailForUR3.stl")
ur3_rail_scene = Mesh(filename=ur3_rail_location, color="#C0C0C0", scale=[0.001, 0.001, 0.001])
ur3_rail_scene.T = SE3(-1.5, 0.1, 0)
env.add(ur3_rail_scene)

#BasePlatePick.STL
#turned so the long side runs along the rail. sits on the floor. top covers x 0 to 1.5, y -0.45 to -0.05, at z = 0.01
base_plate_pick_location = os.path.join(scene_folder, "BasePlatePick.STL")
base_plate_pick_scene = Mesh(filename=base_plate_pick_location, color="#2E8B57", scale=[0.001, 0.001, 0.001])
base_plate_pick_scene.T = SE3(0, -0.05, 0) * trotz(-pi/2)
env.add(base_plate_pick_scene)

#BasePlatePlace.STL
#other side of the rail. also on the floor. top covers x 0.375 to 1.125, y 0.25 to 0.65, at z = 0.01
base_plate_place_location = os.path.join(scene_folder, "BasePlatePlace.STL")
base_plate_place_scene = Mesh(filename=base_plate_place_location, color="#1E5AA8", scale=[0.001, 0.001, 0.001])
base_plate_place_scene.T = SE3(0.375, 0.65, 0) * trotz(-pi/2)
env.add(base_plate_place_scene)

#Can.STL
#random spawning stuff
spawn_x_range = [0.025, 1.475] #pick plate top minus the can radius so the whole can sits on the plate. gets cut down by the reach check
spawn_y_range = [-0.425, -0.075]
can_offset = SE3(-0.025, -0.025, 0) #Can.STL origin is on the corner not the centre of the base. Shouldve modeled it better but oh no

def spawn_can(env, seed):
    random = np.random.default_rng(seed) #week 4 content. 
    can_x = random.uniform(spawn_x_range[0], spawn_x_range[1])
    can_y = random.uniform(spawn_y_range[0], spawn_y_range[1])
    can_pose = SE3(can_x, can_y, 0.01) #centre of the base of the can, sitting on top of the pick plate
    can_location = os.path.join(scene_folder, "Can.STL")
    can_scene = Mesh(filename=can_location, color="#C8102E", scale=[0.001, 0.001, 0.001])
    can_scene.T = can_pose * can_offset
    env.add(can_scene) 
    return can_scene, can_pose #hands back two things like add_UR3 does

random_result = 1 #change this to get a different spawn. like 1,2,3,4,5 etc
can_scene, can_pose = spawn_can(env, random_result)

#UR3 Base
ur3, ur3_base_scene = add_UR3(env)
set_UR3_pose(ur3, ur3_base_scene, SE3(0, 0.1, 0.15))

#Camera.STL
#just the model so you can see where the camera is. sits on the wrist and moves with the arm
camera_location = os.path.join(scene_folder, "Camera.STL")
camera_scene = Mesh(filename=camera_location, color="#E0E0E05C", scale=[0.001, 0.001, 0.001])
update_camera_mesh(ur3, camera_scene)
env.add(camera_scene)
env.step()

#park the UR3 between the two plates
ur3_park_x = 0.75 #middle of both plates
UR3_move_base(env, ur3, ur3_base_scene, ur3_park_x, camera_mesh=camera_scene)
input("Press Enter to continue...")