import time
import numpy as np
import mujoco
import mujoco.viewer
from MujocoScene import build_scene

model, data = build_scene()

#Can.STL random spawning, same ranges as the swift version
spawn_x_range = [0.55, 0.95] #stress tested in the swift version
spawn_y_range = [-0.325, -0.19] #stress tested in the swift version
random_result = int(np.random.default_rng().integers(0, 1000)) #picks a new spawn every run. swap this for a number (like 3) to repeat one exact trial
random = np.random.default_rng(random_result) #week 4 content
can_pose = np.array([random.uniform(spawn_x_range[0], spawn_x_range[1]), random.uniform(spawn_y_range[0], spawn_y_range[1]), 0.01]) #true position of the middle of the bottom of the can. only used to place it and to score
data.qpos[6:9] = can_pose #qpos[6:9] is the can, qpos[0:6] is the arm
print("trial number", random_result, "can spawned at", can_pose) #real spawn position, only printed for testing. the robot never gets this

#UR3 starting pose and the window to watch it in
ur3_start_q = [0, 0, 0, 0, 0, 0] #all six joints at zero, then it moves over to look for the can. list of joint angles, same as ur3_look_q in the swift version
data.qpos[0:6] = ur3_start_q #puts the arm at those angles. qpos from the mujoco docs (mjdata), does the same job as mj_resetDataKeyframe in raphaels move_ur3e.py
data.ctrl[:] = ur3_start_q #tells the motors to hold those angles. from raphaels move_ur3e.py, where he moves the arm by changing data.ctrl
viewer = mujoco.viewer.launch_passive(model, data) #opens the window. from raphaels simulate_ur3e.py and move_ur3e.py

def move_arm(target_q, steps=1000): #sends the arm to a set of joint angles with the ur3e's own motors. 1000 steps of 0.002s is 2 seconds. same idea as raphaels move_ur3e.py, gets swapped for the lqr later
    data.ctrl[:] = target_q #tells the motors where to go. from raphaels move_ur3e.py
    for k in range(steps):
        mujoco.mj_step(model, data) #moves the sim forward one step. from raphaels move_ur3e.py
        viewer.sync() #updates the window. from raphaels move_ur3e.py
        time.sleep(model.opt.timestep) #slows it down to about real speed

input("Press Enter to continue...") #waits at zero until you press enter in the terminal, same as the swift version

#look at the pick plate
ur3_look_q = [-1.9766, -1.1482, 0.8327, -1.2553, -1.5708, 2.7358] #camera straight down, 0.6 above the plate, over the middle of the spawn area. found with inverse kinematics so the whole spawn area is in the picture
move_arm(ur3_look_q)

#camera reading
camera_matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]) #same numbers as the rgbd camera in the week 4 tutorial. the mujoco camera is set up to match (fovy 51.28 deg on 640 x 480)
camera_random = np.random.default_rng(random_result + 1000) #random numbers for the camera noise. + 1000 so its not the same numbers the spawn used (week 2 tutorial trick)
pixel_sigma = 1.0 #how many pixels the reading is out by on average. picked by me, not from a lab
can_height = 0.135 #measured off Can.STL
renderer = mujoco.Renderer(model, 480, 640) #takes 640 x 480 photos from the wrist camera

def can_camera_location(): #takes a colour and depth photo, finds the red can in it and reads how far away it is. the robot side: it never gets can_pose
    renderer.update_scene(data, camera="wrist_cam")
    photo = renderer.render() #colour photo
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="wrist_cam")
    depth_photo = renderer.render() #depth photo, metres per pixel
    renderer.disable_depth_rendering()
    red_pixels = (photo[:, :, 0] > 120) & (photo[:, :, 1] < 80) & (photo[:, :, 2] < 100) #the can is the only red thing in the picture
    v, u = np.nonzero(red_pixels) #rows and columns of every red pixel
    can_position_to_camera = np.array([u.mean(), v.mean()]) + camera_random.normal(0, 1, 2) * pixel_sigma #middle of the red pixels, with a bit of noise on it
    depth_true = depth_photo[int(v.mean()), int(u.mean())] #depth at the middle of the can. d = Z, week 4 lecture
    sigma = 0.0015 + 0.0022 * depth_true ** 2 #sigma_d grows with d^2, week 4 tutorial part 2
    can_depth_noisy = depth_true + camera_random.normal(0, 1) * sigma #same line as the week 4 tutorial depth image
    return can_position_to_camera, can_depth_noisy

def can_location_estimate(can_position_to_camera, can_depth_noisy): #works the can position back out from the pixel and the depth. same maths as the swift version
    pixel_number_location = np.array([can_position_to_camera[0], can_position_to_camera[1], 1]) #called uv1 in the week 4 tutorial
    can_from_camera_measurement = np.linalg.inv(camera_matrix) @ pixel_number_location * can_depth_noisy #P_C in the week 4 tutorial 2.1
    can_from_camera_measurement = can_from_camera_measurement * np.array([1, -1, -1]) #mujoco cameras look down -z with y up. week 4 is z forward, y down. so flip y and z
    can_top = data.cam_xpos[0] + data.cam_xmat[0].reshape(3, 3) @ can_from_camera_measurement #moved into the world frame. same as T_RC * P_C in week 4 tutorial 2.2
    return can_top - np.array([0, 0, can_height]) #the camera sees the top of the can, take off the height to get the bottom, same point as can_pose

can_position_to_camera, can_depth_noisy = can_camera_location()
print("can shows up at pixel", can_position_to_camera, "depth", can_depth_noisy)

#kalman filter on the can position. same as the swift version: week 7 kalman filter (part B) with A and C left out, because the can doesnt move
number_of_readings = 20
Qd = np.eye(3) * 1e-8 #how much the can might drift between readings. basically nothing
xy_sigma = pixel_sigma * 0.45 / camera_matrix[0, 0] #1 pixel of noise is this many metres at the can top, about 0.45 away
depth_sigma = 0.0015 + 0.0022 * 0.45 ** 2 #depth noise at 0.45 away, same formula as the camera
Rn = np.diag([xy_sigma ** 2, xy_sigma ** 2, depth_sigma ** 2]) #how noisy one reading is in x, y, z
xh = np.array([0.75, -0.2575, 0.01]) #first guess: the middle of the spawn area
P = np.eye(3) * 0.2 ** 2 #how unsure the first guess is. about 20cm either way

for k in range(number_of_readings):
    reading_pixel, reading_depth = can_camera_location()
    y = can_location_estimate(reading_pixel, reading_depth) #this reading turned into x y z
    P = P + Qd #a touch less sure before each reading
    K = P @ np.linalg.inv(P + Rn) #how much to trust the new reading
    xh = xh + K @ (y - xh) #move the guess toward the reading by that much
    P = (np.eye(3) - K) @ P #a bit more sure afterwards
print("kalman guess", xh, "unsure by", np.sqrt(np.diag(P)) * 1000, "mm, really out by", np.linalg.norm(xh - can_pose) * 1000, "mm") #real position only used here to score it
can_pose_estimate = xh #the filter's answer is what the robot uses from here

while viewer.is_running(): #keeps the window open until you close it
    mujoco.mj_step(model, data)
    viewer.sync()