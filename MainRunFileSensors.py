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
spawn_y_range = [-0.325, -0.19] #stress tested. kept away from the rail so the arm doesnt hit the purple base plate when it reaches down
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

#camera check
#checks the camera numbers are right. works out which pixel the can should be on, then sees how far off the camera's pixel was (week 3 part C)
#uses the real can position, but only to mark the camera, not to move the robot
camera_pose = ur3.fkine(ur3.q) * wrist_to_camera #where the camera is
can_to_camera = (camera_pose.inv() * can_pose.t).flatten() #where the can is, measured from the camera
pixel_expected = (camera_matrix @ can_to_camera / can_to_camera[2])[0:2] #the pixel the can should be on
reprojection_error = np.linalg.norm(can_position_to_camera - pixel_expected) #how many pixels off the camera was. about 1 is good, big means the camera numbers are wrong
print("reprojection error", reprojection_error, "pixels")

#take lots of readings
number_of_readings = 20 #get 20 readings and then get the average of those so that i can get a more accurate reading. the camera is noisy, so this should help. also stress test the camera code
can_readings = []
for k in range(number_of_readings):
    reading_pixel, reading_depth = can_camera_location(ur3, can_pose)
    can_readings.append([reading_pixel[0], reading_pixel[1], reading_depth]) #one reading is [u, v, depth]
can_readings = np.array(can_readings)
print("Can readings:", can_readings)

#kalman filter on the can position
#one reading is a bit off, so take 20 and blend them. each reading nudges the guess, and the guess gets steadier
#this is the week 7 kalman filter (part B) with A and C left out, because the can doesnt move and the camera gives its position directly
Qd = np.eye(3) * 1e-8 #how much the can might drift between readings. basically nothing
xy_sigma = pixel_sigma * 0.44 / camera_matrix[0, 0] #1 pixel of noise is this many metres on the plate, from 0.44 up. about 0.9mm
depth_sigma = 0.0015 + 0.0022 * 0.44 ** 2 #depth noise from 0.44 up, same formula as the camera. about 2mm
Rn = np.diag([xy_sigma ** 2, xy_sigma ** 2, depth_sigma ** 2]) #how noisy one reading is in x, y, z
xh = np.array([0.75, -0.2575, 0.01]) #first guess: the middle of the spawn area
P = np.eye(3) * 0.2 ** 2 #how unsure the first guess is. about 20cm either way

for k in range(number_of_readings):
    reading_pixel, reading_depth = can_camera_location(ur3, can_pose)
    y = can_location_estimate(ur3, reading_pixel, reading_depth) #this reading turned into x y z
    P = P + Qd #a touch less sure before each reading
    K = P @ np.linalg.inv(P + Rn) #how much to trust the new reading. near 1 = believe it, near 0 = ignore it
    xh = xh + K @ (y - xh) #move the guess toward the reading by that much
    P = (np.eye(3) - K) @ P #a bit more sure afterwards
    print("reading", k + 1, "guess", xh, "unsure by", np.sqrt(np.diag(P)) * 1000, "mm, really out by", np.linalg.norm(xh - can_pose.t) * 1000, "mm") #real position only used here to score it
can_pose_estimate = xh #the filter's answer is what the robot uses from here
input("Press Enter to continue...")

#go to the can
#only uses can_pose_estimate. keeps the camera pointing down and stops just above the top of the can
can_height = 0.135 #measured off Can.STL
grab_pose = SE3.Rt(ur3.fkine(ur3.q).R, [can_pose_estimate[0], can_pose_estimate[1], can_pose_estimate[2] + can_height + 0.01]) #same tool direction as the look pose, 0.01 above the top of the can
result = ur3.ikine_LM(ur3.base.inv() * grab_pose, q0=ur3.q) #inverse kinematics like UR3_move_and_grab, with the same base frame fix
UR3_move_joints(env, ur3, result.q, camera_mesh=camera_scene)
input("Press Enter to continue...")

#pick up the can
#the can gets stuck to the gripper and lifted straight up
can_in_gripper = ur3.fkine(ur3.q).inv() * SE3(np.asarray(can_scene.T), check=False) #where the can is compared to the gripper right now. it keeps that spot while its carried
lift_pose = SE3(0, 0, 0.3) * ur3.fkine(ur3.q) #same spot, 0.3 higher. high enough to carry the can over the rail
result = ur3.ikine_LM(ur3.base.inv() * lift_pose, q0=ur3.q)
UR3_move_joints(env, ur3, result.q, camera_mesh=camera_scene, item=can_scene, item_offset=can_in_gripper)
input("Press Enter to continue...")

#put the can on the blue plate
ur3_over_place_q = [5.039, -1.7066, -1.4656, -1.5402, 1.5708, 0.3266] #joint angles for above the middle of the blue plate. found with ikine, same way as ur3_look_q
ur3_place_q = [5.0392, -2.9424, -1.7358, -0.0342, 1.5708, 0.3268] #same spot, lowered so the can sits on the plate
UR3_move_joints(env, ur3, ur3_over_place_q, camera_mesh=camera_scene, item=can_scene, item_offset=can_in_gripper) #swing over the rail with the can
UR3_move_joints(env, ur3, ur3_place_q, camera_mesh=camera_scene, item=can_scene, item_offset=can_in_gripper) #lower it onto the plate
UR3_move_joints(env, ur3, ur3_over_place_q, camera_mesh=camera_scene) #let go and back off. no item passed in this time, so the can stays where it was put
input("Press Enter to continue...")
