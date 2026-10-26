#The main run. spawns the can somewhere random, finds it with the camera, drives the base along the rail to it, picks it up, and puts it on the blue plate. the robot only ever uses its guesses, never the real answers
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from MujocoScene import build_scene
from LQRPart import UR3_move_joints_lqr, joints_estimate, rail_estimate, rail_log
from RailEstimation import kf_step, plot_rail_log

model, data = build_scene() #load our scene

#Can.STL random spawning, same ranges as the swift version
spawn_x_range = [0.55, 0.95] #where along the green plate the can can appear. stress tested in the swift version
spawn_y_range = [-0.325, -0.19] #how far across the green plate it can appear. stress tested in the swift version
random_result = int(np.random.default_rng().integers(0, 1000)) #picks a new trial number every run. swap this for a number (like 3) to repeat one exact trial
random = np.random.default_rng(random_result) #random numbers from that trial number. np.random.default_rng(seed) like the week 2 and week 4 tutorials
can_pose = np.array([random.uniform(spawn_x_range[0], spawn_x_range[1]), random.uniform(spawn_y_range[0], spawn_y_range[1]), 0.01]) #the REAL spot of the bottom middle of the can. only used to put it there and to mark how well we did, the robot never sees it
data.qpos[7:10] = can_pose #put the can there. qpos[0] is the rail, qpos[1:7] is the arm, qpos[7:10] is the can
print("trial number", random_result, "can spawned at", can_pose) #printed just so we can check results

#UR3 starting pose and the window to watch it in
ur3_start_q = [0, 0, 0, 0, 0, 0] #all six joint angles at zero to start. same as the swift version
data.qpos[1:7] = ur3_start_q #put the arm at those angles. same job as mj_resetDataKeyframe in raphaels move_ur3e.py
viewer = mujoco.viewer.launch_passive(model, data) #opens the 3d window. from raphaels simulate_ur3e.py and move_ur3e.py

encoder_random = np.random.default_rng(random_result + 2000) #random numbers for the sensor wobble. + 2000 so theyre different from the can spawn and camera numbers, but the same trial number always gives the same wobble. same trick as seed + 2000 for the encoders in the week 2 tutorial (part B), same as swift

input("Press Enter to continue...") #waits here until you press enter in the terminal, same as the swift version

#move to look at the pick plate
ur3_look_q = [-1.9766, -1.1482, 0.8327, -1.2553, -1.5708, 2.7358] #joint angles that point the camera straight down, 0.6m above the middle of the spawn area. found with inverse kinematics (industrial robotics, not this subject) so the whole spawn area fits in the picture
UR3_move_joints_lqr(model, data, viewer, ur3_look_q, encoder_random) #move there with the lqr in LQRPart

#camera setup
camera_matrix = np.array([[499.741, 0, 319.266], [0, 499.809, 239.465], [0, 0, 1]]) #called K in week 3. the camera's zoom (about 500) and picture centre (about 320, 240). worked out in CameraCalibration.py
camera_random = np.random.default_rng(random_result + 1000) #random numbers for the camera wobble. + 1000 so theyre different from the spawn numbers. same trick as seed + 1000 in the week 2 tutorial (part B)
pixel_sigma = 1.0 #how many pixels the camera reading wobbles by. picked by me, not from a lab. adds some randomness to the scene so the robot has to use its guesses, not just the camera. same as the swift version
can_height = 0.135 #how tall the can is, measured off Can.STL
renderer = mujoco.Renderer(model, 480, 640) #takes 640 x 480 photos from the wrist camera. mujoco docs (Renderer), not from a lab

#camera window on the side, so you can watch what the wrist camera sees. matplotlib like the tutorials. plt.ion and plt.pause for live updating are from the matplotlib docs, not a lab
plt.ion() #lets the window keep updating while the code runs
camera_window, camera_axes = plt.subplots()
camera_picture = camera_axes.imshow(np.zeros((480, 640, 3), dtype=np.uint8)) #black picture to start with
can_marker, = camera_axes.plot([], [], 'g+', markersize=25, markeredgewidth=2) #green cross that goes on the can. Taken from Week 3 content ('g+' marker in part c). starting it empty then moving it with set_data is like the week 7 c.4 animation
camera_axes.set_title("wrist camera")

def show_camera(photo): #puts a photo in the camera window
    camera_picture.set_data(photo)
    plt.pause(0.001) #gives the window a moment to redraw

def can_camera_location(): #SENSING: takes a colour photo and a depth photo, finds the red can, and reads how far away it is
    renderer.update_scene(data, camera="wrist_cam")
    photo = renderer.render() #colour photo
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="wrist_cam")
    depth_photo = renderer.render() #depth photo: each pixel is how far away that spot is, in metres
    renderer.disable_depth_rendering()
    red_pixels = (photo[:, :, 0] > 120) & (photo[:, :, 1] < 80) & (photo[:, :, 2] < 100) #find the red pixels. lots of red, not much green or blue. the can is the only red thing. NOT from a lab: colour cut-off numbers picked by me. the true/false mask idea is like valid and good in the week 4 tutorial. getting swapped for week 4 RANSAC (find_objects)
    red_rows, red_columns = np.nonzero(red_pixels) #row (red_rows) and column (red_columns) of every red pixel. like np.flatnonzero in week 4 find_objects
    can_position_to_camera = np.array([red_columns.mean(), red_rows.mean()]) + camera_random.normal(0, 1, 2) * pixel_sigma #the middle of all the red pixels = where the can is in the picture, like the centroid (mean) in week 4 find_objects. plus a bit of wobble like a real camera
    can_marker.set_data([red_columns.mean()], [red_rows.mean()]) #move the green cross onto the can
    show_camera(photo)
    depth_true = depth_photo[int(red_rows.mean()), int(red_columns.mean())] #how far away the middle of the can is. d = Z, week 4 lecture (depth camera measurement model)
    sigma = 0.0015 + 0.0022 * depth_true ** 2 #how much the depth wobbles. further away = more wobble. week 4 tutorial part 2
    can_depth_noisy = depth_true + camera_random.normal(0, 1) * sigma #depth plus that wobble. same line as the week 4 tutorial depth image
    return can_position_to_camera, can_depth_noisy

def can_location_estimate(can_position_to_camera, can_depth_noisy): #turns "pixel + how far away" into a real x y z spot in the world. same maths as the swift version
    pixel_number_location = np.array([can_position_to_camera[0], can_position_to_camera[1], 1]) #the pixel with a 1 stuck on the end, the maths needs it. called uv1 in the week 4 tutorial
    can_from_camera_measurement = np.linalg.inv(camera_matrix) @ pixel_number_location * can_depth_noisy #undo the camera zoom to get a direction, then go that far along it = where the can is from the camera. P_C in the week 4 tutorial 2.1
    can_from_camera_measurement = can_from_camera_measurement * np.array([1, -1, -1]) #mujoco cameras point a different way to the week 4 maths (looks down -z with y up, instead of z forward y down). flip y and z to match
    can_top = data.cam_xpos[0] + data.cam_xmat[0].reshape(3, 3) @ can_from_camera_measurement #turn "from the camera" into "in the world" using where the camera is and which way its facing. same as T_RC * P_C in week 4 tutorial 2.2
    return can_top - np.array([0, 0, can_height]) #the camera sees the top of the can, take off its height to get the bottom, same spot as can_pose

can_position_to_camera, can_depth_noisy = can_camera_location() #first photo, just for the camera check below

#camera check (week 3 part C). works out which pixel the can SHOULD be on, then sees how far off the camera actually was. uses the real can spot, but only to mark the camera, never to move the robot. same as the swift version
camera_pose = data.cam_xpos[0] #where the camera is. same job as ur3.fkine(ur3.q) * wrist_to_camera in swift
can_to_camera = data.cam_xmat[0].reshape(3, 3).T @ (can_pose + np.array([0, 0, can_height]) - camera_pose) * np.array([1, -1, -1]) #the real can top seen from the camera, flipped to week 4's direction. same as camera_pose.inv() * can_pose.t in swift
pixel_expected = (camera_matrix @ can_to_camera / can_to_camera[2])[0:2] #the pixel the can should land on. same line as swift
reprojection_error = np.linalg.norm(can_position_to_camera - pixel_expected) #how many pixels off the camera was. about 1 is good, big means the camera numbers are wrong
print("reprojection error", reprojection_error, "pixels")

#kalman filter on the can position (week 7 part B). same guess-then-fix idea as the rail, but the can doesnt move so the guess step does nothing
number_of_readings = 20 #how many photos to take before trusting the answer
can_model_noise = np.eye(3) * 0.002 ** 2 #called Qd in the tutorial. how much we doubt our own guess each step. the can doesnt move, but this stops the filter getting too sure of itself. averaging cant fix a camera error that stays the same every photo (week 7 B.6, when in doubt make Qd bigger)
xy_sigma = pixel_sigma * 0.45 / camera_matrix[0, 0] #1 pixel of wobble turned into metres, at the can top about 0.45m away. X = (u - u0) Z / f from week 4 tutorial 1.3, so 1 pixel = Z / f metres
depth_sigma = 0.0015 + 0.0022 * 0.45 ** 2 #depth wobble at 0.45m away, same formula as the camera
steady_sigma = 0.003 #about 3mm of error that stays the same every photo: the middle of the red pixels isnt exactly the middle of the can top. a bias, not noise (week 2 part 0). the 3mm is picked by me from the 2.7mm error in a test run, not from a lab. check over more trials
camera_noise = np.diag([xy_sigma ** 2, xy_sigma ** 2, depth_sigma ** 2]) + np.eye(3) * steady_sigma ** 2 #called Rn in the tutorial. how wobbly one photo reading is in x, y and z, all added up
can_guess = np.array([0.75, -0.2575, 0.01]) #called xh in the tutorial. first guess before any photos: the middle of the spawn area
can_unsure = np.eye(3) * 0.2 ** 2 #called P in the tutorial. how unsure that first guess is. about 20cm either way

def can_kf_update(can_guess, can_unsure): #take a photo, turn it into x y z, then do one guess-then-fix step with the same week 7 filter as the rail. A and C are just 1s (the can doesnt move, and the reading is already x y z). no motor, so B is 0
    y = can_location_estimate(*can_camera_location())
    return kf_step(can_guess, can_unsure, 0, y, np.eye(3), np.zeros((3, 1)), np.eye(3), can_model_noise, camera_noise)

for k in range(number_of_readings): #take 20 photos, each one makes the guess better
    can_guess, can_unsure = can_kf_update(can_guess, can_unsure)
print("kalman guess", can_guess, "unsure by", np.sqrt(np.diag(can_unsure)) * 1000, "mm, really out by", np.linalg.norm(can_guess - can_pose) * 1000, "mm") #the real spot is only used here to mark it

#drive the base along the rail to line up with the can. rail 0 is x = 0.75, so the rail spot is the can's guessed x minus 0.75. the arm holds still while the base drives
UR3_move_joints_lqr(model, data, viewer, ur3_look_q, encoder_random, steps=1500, rail_to=can_guess[0] - 0.75)
print("rail guess", rail_estimate[0], "really out by", (rail_estimate[0] - data.qpos[0]) * 1000, "mm") #the real rail spot is only used here to mark it

#moving the gripper. week 9 jacobian maths (q_dot = J+ * lambda * e), like rmrc in ir
gripper_id = model.site("attachment_site").id #the end of the wrist, where the gripper is. from the ur3e.xml
servo_gain = 2.0 #lambda in the week 9 slides. how fast to close the gap. picked by me

def move_gripper(target_pose, steps=300, item_offset=None): #moves the gripper to target_pose in small steps, keeping it pointing straight down. same job as ikine then move in the swift version
    for k in range(steps):
        position_error = target_pose - data.site_xpos[gripper_id] #how far the gripper is from where it should be
        lean_error = np.cross(data.site_xmat[gripper_id].reshape(3, 3)[:, 2], [0, 0, -1]) #how much its tilted away from straight down. the week 9 PBVS error uses the rotation angle (theta u), this cross product is a simpler version that matches it for small tilts. np.cross is not from a lab
        error = np.concatenate([position_error, lean_error]) #both together. e in the week 9 slides (PBVS: translation stacked on rotation)
        if np.linalg.norm(error) < 0.002: #close enough (2mm), stop
            break
        jacobian = np.zeros((6, model.nv))
        mujoco.mj_jacSite(model, data, jacobian[0:3], jacobian[3:6], gripper_id) #the jacobian: how much the gripper moves when each joint turns a little. mujoco docs, same as jacob0 in ir
        q_dot = np.linalg.pinv(jacobian[:, 1:7]) @ (servo_gain * error) #flip that around: how fast to turn each joint to close the gap. q_dot = J+ * v with v = lambda * e, week 9 slides (PBVS control law and robot joint velocities). columns 1 to 6 are the arm, column 0 is the rail which stays still here
        new_q = joints_estimate[:, 0] + q_dot * 0.2 #where the joints should be 0.2s from now, starting from our guess of where they are. 0.2s picked by me
        UR3_move_joints_lqr(model, data, viewer, new_q, encoder_random, steps=100, item_offset=item_offset) #move there over 0.2s with the lqr
        renderer.update_scene(data, camera="wrist_cam")
        show_camera(renderer.render()) #update the camera window as it moves
    return np.linalg.norm(error)

#go to the can. pbvs from week 9: take a photo, fix the guess, move a bit closer, repeat until the error is small. getting closer makes the photos better, because depth wobble grows with distance squared (week 4)
for k in range(200):
    can_guess, can_unsure = can_kf_update(can_guess, can_unsure) #new photo from where the camera is now, same filter as above
    if move_gripper(can_guess + np.array([0, 0, can_height + 0.15]), steps=1) < 0.002: #one small move toward 0.15m above the can, high enough that it stays in the picture. stop once its there
        break
can_marker.set_data([], []) #hide the green cross, the can gets picked up from here
print("pbvs done after", k, "photos, gripper above the can by", (data.site_xpos[gripper_id] - can_pose)[0:2] * 1000, "mm in x y") #the real spot is only used here to mark it
input("Press Enter to continue...")

#pick up the can
grab_pose = can_guess + np.array([0, 0, can_height + 0.01]) #1cm above the top of the can, same as the swift version
move_gripper(grab_pose) #go down to it
can_in_gripper = data.qpos[7:10] - data.site_xpos[gripper_id] #where the can sits compared to the gripper. it keeps that spot while being carried, same as the swift version
move_gripper(grab_pose + np.array([0, 0, 0.3]), item_offset=can_in_gripper) #lift it 0.3m straight up
input("Press Enter to continue...")

#put the can on the blue plate
ur3_over_place_q = joints_estimate[:, 0].copy() #start from our guess of where the arm is now
ur3_over_place_q[0] = ur3_over_place_q[0] + np.pi #spin the bottom joint half way round, over the rail to the blue plate side
UR3_move_joints_lqr(model, data, viewer, ur3_over_place_q, encoder_random, steps=1500, item_offset=can_in_gripper, rail_to=0.0) #swing over while the base drives back to the middle of the rail (x = 0.75, the middle of the blue plate)
place_pose = np.array([0.75, 0.45, 0.01 + can_height + 0.01]) #middle of the blue plate, with the can bottom sitting on it. same spot as the swift version
move_gripper(place_pose, item_offset=can_in_gripper) #lower it onto the plate
move_gripper(place_pose + np.array([0, 0, 0.15])) #let go and move back up. no can passed in, so it stays where it was put
print("can placed at", data.qpos[7:10], "aiming for", place_pose[0:2]) #the real spot is only used here to mark it
input("Press Enter to continue...")

ur3_end_q = joints_estimate[:, 0].copy() #where the arm finished
while viewer.is_running(): #keeps going until you close the 3d window
    UR3_move_joints_lqr(model, data, viewer, ur3_end_q, encoder_random, steps=50) #keeps the lqr holding the arm up, otherwise it flops down onto the can

plot_rail_log(rail_log, "rail during the pick and place") #once the window is closed, show the rail report card from RailEstimation