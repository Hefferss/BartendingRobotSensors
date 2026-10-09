import os
import sys
import swift
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
#sitting in the middle of the pick plate for now. gets replaced by the random spawn function
can_location = os.path.join(scene_folder, "Can.STL")
can_scene = Mesh(filename=can_location, color="#C8102E", scale=[0.001, 0.001, 0.001])
can_offset = SE3(-0.025, -0.025, 0) #Can.STL origin is on the corner not the centre of the base. delete this if the stl gets re-exported centred
can_scene.T = SE3(0.75, -0.25, 0.01) * can_offset
env.add(can_scene)

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