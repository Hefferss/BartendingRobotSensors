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
from UR3 import add_UR3, UR3_move_base, UR3_move_and_grab, set_UR3_pose, update_camera_mesh, UR3_move_joints, wrist_to_camera

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
spawn_x_range = [0.55, 0.95] #stress tested
spawn_y_range = [-0.25, -0.075] #stress tested
can_offset = SE3(-0.025, -0.025, 0) #Can.STL origin is on the corner not the centre of the base. Shouldve modeled it better but oh well

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

random_result = 1 #change this to get a different spawn
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

#look at the pick plate
ur3_look_q = [1.9267, -1.7146, -1.9752, -1.0226, 1.5708, 0.3559] #joint angles that point the camera straight down at the spawn area, 0.44 above the plate. found through moving the camera directly into the middle ofthe spawn and then doing it 0.44m above the base (which was a number i just made up), THEN USING Ikine to find it
UR3_move_joints(env, ur3, ur3_look_q, camera_mesh=camera_scene)

#camera reading
camera_matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]) #same numbers as the rgbd camera in the week 4 tutorial (f=0.005, rho=10e-6, 640 x 480)
camera_random = np.random.default_rng(random_result + 1000) #random numbers for the camera noise. seeded so a trial repeats, + 1000 so its not the same numbers the spawn used (same trick as the week 2 tutorial)
pixel_sigma = 1.0 #how many pixels the reading is out by on average. picked by me, not from a lab. will stress test and changed
 
def can_camera_location(ur3, can_pose): #works out where the can shows up in the image. can_pose is the true position, only used here to make the reading
    camera_pose = ur3.fkine(ur3.q) * wrist_to_camera #where the camera is in the world
    can_to_camera = (camera_pose.inv() * can_pose.t).flatten() #can position in the camera frame. x right, y down, z forward (week 4 section 2.2, but the oter way around because of how its been put in and orientated)
    can_position_to_camera = camera_matrix @ can_to_camera / can_to_camera[2] #pixel the can lands on. u = f/rho * X/Z + u0, v = f/rho * Y/Z + v0 from da week 4 stuff
    can_position_to_camera = can_position_to_camera[0:2] + camera_random.normal(0, 1, 2) * pixel_sigma #pixel with a bit of noise on it
    depth_true = can_to_camera[2] #d = Z, week 4 lecture
    sigma = 0.0015 + 0.0022 * depth_true ** 2 #sigma_d grows with d^2, week 4 tutorial part 2
    can_depth_noisy = depth_true + camera_random.normal(0, 1) * sigma #same line as the week 4 tutorial depth image. called noisy because im planning on using noise stuff to make it non noisy, cause right now it'll be affected by.... noise....
    return can_position_to_camera, can_depth_noisy

def can_location_estimate(ur3, can_position_to_camera, can_depth_noisy): #works the can position back out from the pixel and the depth. it never gets can_pose, only what the camera read
    camera_pose = ur3.fkine(ur3.q) * wrist_to_camera #where the camera is in the world
    pixel_number_location = np.array([can_position_to_camera[0], can_position_to_camera[1], 1]) #the pixel written as [u(x direction distance in pixels) v(y direction distance in pixels) 1]. called uv1 in the week 4 tutorial
    can_from_camera_measurement = np.linalg.inv(camera_matrix) @ pixel_number_location * can_depth_noisy #point in the camera frame, called P_C in the week 4 tutorial 2.1. inv(K) @ [u v 1] is the ray, then scaled by the depth
    can_pose_estimate = (camera_pose * can_from_camera_measurement).flatten() #moved into the world frame, x y z. same as T_RC * P_C in week 4 tutorial 2.2
    return can_pose_estimate


can_position_to_camera, can_depth_noisy = can_camera_location(ur3, can_pose)
can_pose_estimate = can_location_estimate(ur3, can_position_to_camera, can_depth_noisy)
print("can shows up at pixel", can_position_to_camera, "depth", can_depth_noisy) #shows where the can shows up, thought it might be a good iudea in case we do stress tests and it bugs out and dies on me. 
input("Press Enter to continue...")

#go to the can
#only uses can_pose_estimate. keeps the camera pointing down and stops just above the top of the can
can_height = 0.135 #measured off Can.STL
grab_pose = SE3.Rt(ur3.fkine(ur3.q).R, [can_pose_estimate[0], can_pose_estimate[1], can_pose_estimate[2] + can_height + 0.01]) #same tool direction as the look pose, 0.01 above the top of the can
result = ur3.ikine_LM(ur3.base.inv() * grab_pose, q0=ur3.q) #inverse kinematics like UR3_move_and_grab, with the same base frame fix
UR3_move_joints(env, ur3, result.q, camera_mesh=camera_scene)
input("Press Enter to continue...")