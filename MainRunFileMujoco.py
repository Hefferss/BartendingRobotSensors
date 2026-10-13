import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from MujocoScene import build_scene
from LQRPart import UR3_move_joints_lqr, joints_estimate

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
viewer = mujoco.viewer.launch_passive(model, data) #opens the window. from raphaels simulate_ur3e.py and move_ur3e.py

encoder_random = np.random.default_rng(random_result + 2000) #random numbers for the encoder noise. seeded so a trial repeats, + 2000 so its not the same numbers the spawn or the camera used. same as swift

input("Press Enter to continue...") #waits at zero until you press enter in the terminal, same as the swift version

#look at the pick plate
ur3_look_q = [-1.9766, -1.1482, 0.8327, -1.2553, -1.5708, 2.7358] #camera straight down, 0.6 above the plate, over the middle of the spawn area. found with inverse kinematics so the whole spawn area is in the picture
UR3_move_joints_lqr(model, data, viewer, ur3_look_q, encoder_random)

#camera reading
camera_matrix = np.array([[499.766, 0, 319.254], [0, 499.829, 239.469], [0, 0, 1]]) #from CameraCalibration.py (week 3 calibration). the true camera is f = 500, centre (320, 240)
camera_random = np.random.default_rng(random_result + 1000) #random numbers for the camera noise. + 1000 so its not the same numbers the spawn used (week 2 tutorial trick)
pixel_sigma = 1.0 #how many pixels the reading is out by on average. picked by me, not from a lab
can_height = 0.135 #measured off Can.STL
renderer = mujoco.Renderer(model, 480, 640) #takes 640 x 480 photos from the wrist camera

#camera window on the side, so you can watch what the wrist camera sees. matplotlib like the tutorials
plt.ion() #lets the window keep updating while the code runs
camera_window, camera_axes = plt.subplots()
camera_picture = camera_axes.imshow(np.zeros((480, 640, 3), dtype=np.uint8)) #blank to start with
can_marker, = camera_axes.plot([], [], 'g+', markersize=25, markeredgewidth=2) #green cross on where it found the can
camera_axes.set_title("wrist camera")

def show_camera(photo): #puts a photo in the camera window
    camera_picture.set_data(photo)
    plt.pause(0.001) #gives the window a moment to redraw

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
    can_marker.set_data([u.mean()], [v.mean()]) #move the green cross onto the can
    show_camera(photo)
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

#camera check. same as the swift version (week 3 part C): works out which pixel the can should be on, then sees how far off the camera's pixel was. real can position only used to mark the camera, not to move the robot
camera_pose = data.cam_xpos[0] #where the camera is. same job as ur3.fkine(ur3.q) * wrist_to_camera in swift
can_to_camera = data.cam_xmat[0].reshape(3, 3).T @ (can_pose + np.array([0, 0, can_height]) - camera_pose) * np.array([1, -1, -1]) #top of the can measured from the camera, flipped to week 4's z forward y down. same as camera_pose.inv() * can_pose.t in swift
pixel_expected = (camera_matrix @ can_to_camera / can_to_camera[2])[0:2] #the pixel the can should be on. same line as swift
reprojection_error = np.linalg.norm(can_position_to_camera - pixel_expected) #how many pixels off the camera was. about 1 is good, big means the camera numbers are wrong. same as swift
print("reprojection error", reprojection_error, "pixels")

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

#moving the gripper. week 9 jacobian maths (q_dot = J+ * lambda * e), like rmrc in ir
gripper_id = model.site("attachment_site").id #end of the wrist. from the ur3e.xml
servo_gain = 2.0 #lambda in the week 9 slides. picked by me

def move_gripper(target_pose, steps=300, item_offset=None): #moves the gripper to target_pose in small steps, pointing straight down. same job as ikine then move in the swift version
    for k in range(steps):
        position_error = target_pose - data.site_xpos[gripper_id] #how far the gripper is from where it should be
        lean_error = np.cross(data.site_xmat[gripper_id].reshape(3, 3)[:, 2], [0, 0, -1]) #how much its leaning off straight down
        error = np.concatenate([position_error, lean_error]) #both together, e in the week 9 slides
        if np.linalg.norm(error) < 0.002: #close enough, 2mm
            break
        jacobian = np.zeros((6, model.nv))
        mujoco.mj_jacSite(model, data, jacobian[0:3], jacobian[3:6], gripper_id) #robot jacobian J(q), mujoco docs. same as jacob0 in ir
        q_dot = np.linalg.pinv(jacobian[:, 0:6]) @ (servo_gain * error) #joint speeds that move the gripper toward the target. q_dot = J+ * lambda * e, week 9 slides
        new_q = joints_estimate[:, 0] + q_dot * 0.2 #where the joints should be after 0.2s, starting from where the observer thinks they are
        UR3_move_joints_lqr(model, data, viewer, new_q, encoder_random, steps=100, item_offset=item_offset) #move there for 0.2s with the lqr
        renderer.update_scene(data, camera="wrist_cam")
        show_camera(renderer.render()) #update the camera window as it moves
    return np.linalg.norm(error)

#go to the can. pbvs from week 9: take a photo, update the guess, move a bit closer, repeat
for k in range(200):
    reading_pixel, reading_depth = can_camera_location() #new photo from where the camera is now
    y = can_location_estimate(reading_pixel, reading_depth)
    P = P + Qd #same week 7 kalman filter as above
    K = P @ np.linalg.inv(P + Rn)
    xh = xh + K @ (y - xh)
    P = (np.eye(3) - K) @ P
    if move_gripper(xh + np.array([0, 0, can_height + 0.15]), steps=1) < 0.002: #one step toward 0.15 above the can, so it stays in the picture
        break
can_pose_estimate = xh
can_marker.set_data([], []) #hide the cross, the can gets picked up from here
print("pbvs done after", k, "photos, gripper above the can by", (data.site_xpos[gripper_id] - can_pose)[0:2] * 1000, "mm in x y") #real position only used to score it
input("Press Enter to continue...")

#pick up the can
grab_pose = can_pose_estimate + np.array([0, 0, can_height + 0.01]) #0.01 above the top of the can, same as the swift version
move_gripper(grab_pose)
can_in_gripper = data.qpos[6:9] - data.site_xpos[gripper_id] #where the can is compared to the gripper right now. it keeps that spot while its carried, same as the swift version
move_gripper(grab_pose + np.array([0, 0, 0.3]), item_offset=can_in_gripper) #same spot, 0.3 higher
input("Press Enter to continue...")

#put the can on the blue plate
ur3_over_place_q = joints_estimate[:, 0].copy() #where the observer thinks the arm is now
ur3_over_place_q[0] = ur3_over_place_q[0] + np.pi #spin the base joint half way round, over the rail to the blue plate
UR3_move_joints_lqr(model, data, viewer, ur3_over_place_q, encoder_random, steps=1500, item_offset=can_in_gripper)
place_pose = np.array([0.75, 0.45, 0.01 + can_height + 0.01]) #middle of the blue plate, can bottom sitting on it. same spot as the swift version
move_gripper(place_pose, item_offset=can_in_gripper) #lower it onto the plate
move_gripper(place_pose + np.array([0, 0, 0.15])) #let go and back off. no item passed in, so the can stays where it was put
print("can placed at", data.qpos[6:9], "aiming for", place_pose[0:2]) #real position only used to score it
input("Press Enter to continue...")

ur3_end_q = joints_estimate[:, 0].copy() #where the arm finished
while viewer.is_running(): #keeps the window open until you close it
    UR3_move_joints_lqr(model, data, viewer, ur3_end_q, encoder_random, steps=50) #keeps the lqr holding the arm up, otherwise it flops down onto the can