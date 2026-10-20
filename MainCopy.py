import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from MujocoScene import build_scene
from LQRPart import UR3_move_joints_lqr, joints_estimate
import os, sys, csv, json, types, warnings
import control as ct
import LQRPart 

LQRPart.time = types.SimpleNamespace(sleep=lambda seconds: None) #doesnt slow down the simulation
warnings.filterwarnings("ignore", message="Mean of empty slice") #doesnt crash when can is out of frame
warnings.filterwarnings("ignore", message="invalid value encountered") #doesnt crash when can is out of frame


model, data = build_scene()

number_of_trails = 20

conditions = {
    "nominal": {"noise_scale": 1.0, "push": 0.0}, # normal conditions
    "camera_noise_3X": {"noise_scale": 3.0, "push": 0.0}, # camera noise increased by 3 times
    "push5N": {"noise_scale": 1.0, "push": 5.0}, # apply a 5N push to the arm
    "push7N": {"noise_scale": 1.0, "push": 7.0}, # apply a 7N push to the arm
    "push10N": {"noise_scale": 1.0, "push": 10.0}, # apply a 10N push to the arm
}

results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(results_folder, exist_ok=True)
torque_limit = model.actuator_ctrlrange[0:6, 1].copy() # each motrs tourqe limit 

class Recorder:
    def __init__(self):
        self.recording, self.every = False, 5
        self.push_force, self.push_steps_left = 0.0, 0
        self.steps, self.angle_error_sq = 0, np.zeros(6)
        self.target, self.phase,self.rows = np.zeros(6), "", []
        self.reset_counts()

    def reset_counts(self):
        self.steps, self.angle_error_sq, self.speed_error_sq = 0, np.zeros(6), np.zeros(6)
        self.saturated_steps, self.phase_steps, self.phase_saturated = np.zeros(6), {}, {}
        self.peak_speed = {}

    def sync(self):
        wrist = model.body("wrist_3_link").id
        data.xfrc_applied[wrist, 1] = self.push_force if self.push_steps_left > 0 else 0.0 #pushes arm in y direction
        self.push_steps_left = max(0, self.push_steps_left - 1)
        self.steps += 1
        self.angle_error_sq += (data.qpos[0:6] - joints_estimate[:, 0]) ** 2 #tracks obverver error over time 
        self.speed_error_sq += (data.qvel[0:6] - joints_estimate[:, 1]) ** 2 #do the same for speed
        saturated = np.abs(data.ctrl[0:6]) > torque_limit # do motors requie more torque than they can give?
        self.saturated_steps += saturated
        self.phase_steps[self.phase] = self.phase_steps.get(self.phase, 0) + 1
        self.phase_saturated[self.phase] = self.phase_saturated.get(self.phase, 0) + int(saturated.any())
        self.peak_speed[self.phase] = max(self.peak_speed.get(self.phase, 0.0), float(np.max(np.abs(data.qvel[0:6]))))
        if self.recording and self.steps % self.every == 0:
            self.rows.append(np.concatenate([[data.time], data.qpos[0:6], data.qvel[0:6], joints_estimate[:, 0], joints_estimate[:, 1], self.target,
                                                data.ctrl[0:6] - data.qfrc_bias[0:6], data.ctrl[0:6], data.actuator_force[0:6]])) #saves data 
        

    def is_running(self):
        return False

viewer = Recorder() #dummy viewer that records data instead of showing a window

belief_data = mujoco.MjData(model) # makes copy of the arm so it doesnt move the actual one
def robot_as_estimated(): #returns pose and observers joint estimate
    belief_data.qpos[:] = data.qpos
    belief_data.qpos[0:6] = joints_estimate[:, 0]
    mujoco.mj_forward(model, belief_data)
    return belief_data
def rotation_angle_deg(R1, R2): #angle between two rotation matrices
    return float(np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1))))

#camera reading
camera_matrix = np.array([[499.766, 0, 319.254], [0, 499.829, 239.469], [0, 0, 1]]) #from CameraCalibration.py (week 3 calibration). the true camera is f = 500, centre (320, 240)
pixel_sigma = 1.0 #how many pixels the reading is out by on average. picked by me, not from a lab
can_height = 0.135 #measured off Can.STL
renderer = mujoco.Renderer(model, 480, 640) #takes 640 x 480 photos from the wrist camera
last_red_pixels = 0 

def can_camera_location(): #takes a colour and depth photo, finds the red can in it and reads how far away it is. the robot side: it never gets can_pose
    global last_red_pixels
    renderer.update_scene(data, camera="wrist_cam")
    photo = renderer.render() #colour photo
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="wrist_cam")
    depth_photo = renderer.render() #depth photo, metres per pixel
    renderer.disable_depth_rendering()
    red_pixels = (photo[:, :, 0] > 120) & (photo[:, :, 1] < 80) & (photo[:, :, 2] < 100) #the can is the only red thing in the picture
    v, u = np.nonzero(red_pixels) #rows and columns of every red pixel
    last_red_pixels = len(u) 
    can_position_to_camera = np.array([u.mean(), v.mean()]) + camera_random.normal(0, 1, 2) * pixel_sigma #middle of the red pixels, with a bit of noise on it
    depth_true = depth_photo[int(v.mean()), int(u.mean())] #depth at the middle of the can. d = Z, week 4 lecture
    sigma = 0.0015 + 0.0022 * depth_true ** 2 #sigma_d grows with d^2, week 4 tutorial part 2
    can_depth_noisy = depth_true + camera_random.normal(0, 1) * sigma #same line as the week 4 tutorial depth image
    return can_position_to_camera, can_depth_noisy

def can_location_estimate(can_position_to_camera, can_depth_noisy, frame=None): #works the can position back out from the pixel and the depth. same maths as the swift version
    if frame is None:
        frame = data #uses the belived camera pose
    pixel_number_location = np.array([can_position_to_camera[0], can_position_to_camera[1], 1]) #called uv1 in the week 4 tutorial
    can_from_camera_measurement = np.linalg.inv(camera_matrix) @ pixel_number_location * can_depth_noisy #P_C in the week 4 tutorial 2.1
    can_from_camera_measurement = can_from_camera_measurement * np.array([1, -1, -1]) #mujoco cameras look down -z with y up. week 4 is z forward, y down. so flip y and z
    can_top = frame.cam_xpos[0] + frame.cam_xmat[0].reshape(3, 3) @ can_from_camera_measurement #moved into the world frame. same as T_RC * P_C in week 4 tutorial 2.2
    return can_top - np.array([0, 0, can_height]) #the camera sees the top of the can, take off the height to get the bottom, same point as can_pose

def belief_check(reading_pixel, reading_depth, y): #how different the reading would be with the camera pose worked out from the observer's joints
    b = robot_as_estimated()
    check["camera_position_gap_mm"].append(float(np.linalg.norm(b.cam_xpos[0] - data.cam_xpos[0]) * 1000))
    check["camera_angle_gap_deg"].append(rotation_angle_deg(b.cam_xmat[0].reshape(3, 3), data.cam_xmat[0].reshape(3, 3)))
    check["can_reading_gap_mm"].append(float(np.linalg.norm(can_location_estimate(reading_pixel, reading_depth, frame=b) - y) * 1000))



def true_depth_to_can(): #used to get true depth of can
    camera_rotation = data.cam_xmat[0].reshape(3, 3)
    return float(-(camera_rotation.T @ (can_pose + np.array([0, 0, can_height]) - data.cam_xpos[0]))[2])


#kalman filter on the can position. same as the swift version: week 7 kalman filter (part B) with A and C left out, because the can doesnt move
number_of_readings = 20
Qd = np.eye(3) * 1e-8 #how much the can might drift between readings. basically nothing
xy_sigma = pixel_sigma * 0.45 / camera_matrix[0, 0] #1 pixel of noise is this many metres at the can top, about 0.45 away
depth_sigma = 0.0015 + 0.0022 * 0.45 ** 2 #depth noise at 0.45 away, same formula as the camera
Rn = np.diag([xy_sigma ** 2, xy_sigma ** 2, depth_sigma ** 2]) #how noisy one reading is in x, y, z


#moving the gripper. week 9 jacobian maths (q_dot = J+ * lambda * e), like rmrc in ir
gripper_id = model.site("attachment_site").id #end of the wrist. from the ur3e.xml
servo_gain = 2.0 #lambda in the week 9 slides. picked by me

def move_gripper(target_pose, steps=300, item_offset=None): #moves the gripper to target_pose in small steps, pointing straight down. same job as ikine then move in the swift version
    for k in range(steps):
        position_error = target_pose - data.site_xpos[gripper_id] #how far the gripper is from where it should be
        lean_error = np.cross(data.site_xmat[gripper_id].reshape(3, 3)[:, 2], [0, 0, -1]) #how much its leaning off straight down
        error = np.concatenate([position_error, lean_error]) #both together, e in the week 9 slides
        b = robot_as_estimated() #the observer's belief about where the gripper is
        check["gripper_position_gap_mm"].append(float(np.linalg.norm(b.site_xpos[gripper_id] - data.site_xpos[gripper_id]) * 1000))
        if np.linalg.norm(error) < 0.002: #close enough, 2mm
            break
        jacobian = np.zeros((6, model.nv))
        mujoco.mj_jacSite(model, data, jacobian[0:3], jacobian[3:6], gripper_id) #robot jacobian J(q), mujoco docs. same as jacob0 in ir
        q_dot = np.linalg.pinv(jacobian[:, 0:6]) @ (servo_gain * error) #joint speeds that move the gripper toward the target. q_dot = J+ * lambda * e, week 9 slides
        new_q = joints_estimate[:, 0] + q_dot * 0.2 #where the joints should be after 0.2s, starting from where the observer thinks they are
        viewer.target = new_q
        UR3_move_joints_lqr(model, data, viewer, new_q, encoder_random, steps=100, item_offset=item_offset) #move there for 0.2s with the lqr
    return np.linalg.norm(error)

def run_trial(random_result, settings, record=False, stop_after="place"):
    global noise_scale, camera_random, encoder_random, can_pose, check
    noise_scale = settings["noise_scale"]
    mujoco.mj_resetData(model, data) 
    joints_estimate[:] = 0 #the observer starts at all zeros like the arm, same as the first run of the main file
    viewer.reset_counts()
    viewer.push_steps_left, viewer.push_force = 0, settings["push"]
    viewer.recording, viewer.every = record, 5 #the joint graphs only need one trial
    viewer.rows, viewer.target = [], np.zeros(6)
    check = {"camera_position_gap_mm": [], "camera_angle_gap_deg": [], "can_reading_gap_mm": [], "gripper_position_gap_mm": []}
    out = {"seed": random_result, "success": False, "failed_because": ""}
    details = {"seed": random_result}
    try:
        #Can.STL random spawning, same ranges as the swift version
        spawn_x_range = [0.55, 0.95] #stress tested in the swift version
        spawn_y_range = [-0.325, -0.19] #stress tested in the swift version
        random = np.random.default_rng(random_result) #week 4 content
        can_pose = np.array([random.uniform(spawn_x_range[0], spawn_x_range[1]), random.uniform(spawn_y_range[0], spawn_y_range[1]), 0.01]) #true position of the middle of the bottom of the can. only used to place it and to score
        data.qpos[6:9] = can_pose #qpos[6:9] is the can, qpos[0:6] is the arm
        out["can_x"], out["can_y"] = can_pose[0], can_pose[1]

        #UR3 starting pose
        ur3_start_q = [0, 0, 0, 0, 0, 0]
        data.qpos[0:6] = ur3_start_q
        encoder_random = np.random.default_rng(random_result + 2000) #random numbers for the encoder noise. seeded so a trial repeats, + 2000 so its not the same numbers the spawn or the camera used. same as swift

        #look at the pick plate
        viewer.phase, viewer.target = "look", np.array(ur3_look_q) 
        UR3_move_joints_lqr(model, data, viewer, ur3_look_q, encoder_random)
        camera_random = np.random.default_rng(random_result + 1000) #random numbers for the camera noise. + 1000 so its not the same numbers the spawn used (week 2 tutorial trick)

        can_position_to_camera, can_depth_noisy = can_camera_location()
        out["red_pixels_at_look"] = last_red_pixels 

        #camera check. works out which pixel the can should be on, then sees how far off the camera's pixel was. real can position only used to mark the camera, not to move the robot
        camera_pose = data.cam_xpos[0]
        can_to_camera = data.cam_xmat[0].reshape(3, 3).T @ (can_pose + np.array([0, 0, can_height]) - camera_pose) * np.array([1, -1, -1])
        pixel_expected = (camera_matrix @ can_to_camera / can_to_camera[2])[0:2]
        reprojection_error = np.linalg.norm(can_position_to_camera - pixel_expected)
        out["reprojection_error_px"] = reprojection_error

        #kalman filter on the can position. week 7 kalman filter (part B) with A and C left out, because the can doesnt move
        xh = np.array([0.75, -0.2575, 0.01]) #first guess: the middle of the spawn area
        P = np.eye(3) * 0.2 ** 2 #how unsure the first guess is. about 20cm either way
        details["truth"] = can_pose.tolist() 
        details["readings"], details["estimates"], details["P_diagonal"] = [], [], []
        for k in range(number_of_readings):
            reading_pixel, reading_depth = can_camera_location()
            y = can_location_estimate(reading_pixel, reading_depth) #this reading turned into x y z
            belief_check(reading_pixel, reading_depth, y)
            P = P + Qd
            K = P @ np.linalg.inv(P + Rn)
            xh = xh + K @ (y - xh)
            P = (np.eye(3) - K) @ P
            details["readings"].append(y.tolist()) 
            details["estimates"].append(xh.tolist())
            details["P_diagonal"].append(np.diag(P).tolist())
        e = xh - can_pose 
        out["kalman_error_mm"] = np.linalg.norm(e) * 1000
        out["kalman_sigma_mm"] = np.sqrt(np.trace(P)) * 1000
        out["kalman_nees"] = e @ np.linalg.inv(P) @ e #error squared divided by what the filter thinks its variance is. about 3 if the filter is honest
        out["kalman_inside_2sigma"] = bool(np.all(np.abs(e) < 2 * np.sqrt(np.diag(P)))) #every axis inside its own 2 sigma
        can_pose_estimate = xh #the filter's answer is what the robot uses from here

        #go to the can. pbvs from week 9: take a photo, update the guess, move a bit closer, repeat
        viewer.phase = "pbvs"
        details["depth_reading"], details["depth_true"] = [], []
        pbvs = details["pbvs"] = {"servo_error_mm": [], "true_distance_mm": [], "kalman_error_mm": [], "kalman_2sigma_mm": [], "reading_error_mm": [], "red_pixels": []} 
        target_above_can = can_pose + np.array([0, 0, can_height + 0.15]) #where the gripper should end up
        for k in range(200):
            if settings["push"] > 0 and k == 20: #shove the wrist sideways for 0.1 s
                viewer.push_steps_left = 50
            reading_pixel, reading_depth = can_camera_location() #new photo from where the camera is now
            details["depth_reading"].append(float(reading_depth)) 
            details["depth_true"].append(true_depth_to_can())
            y = can_location_estimate(reading_pixel, reading_depth)
            belief_check(reading_pixel, reading_depth, y) 
            P = P + Qd #same week 7 kalman filter as above
            K = P @ np.linalg.inv(P + Rn)
            xh = xh + K @ (y - xh)
            P = (np.eye(3) - K) @ P
            servo_error = move_gripper(xh + np.array([0, 0, can_height + 0.15]), steps=1) #one step toward 0.15 above the can, so it stays in the picture.
            pbvs["servo_error_mm"].append(float(servo_error * 1000)) 
            pbvs["true_distance_mm"].append(float(np.linalg.norm(data.site_xpos[gripper_id] - target_above_can) * 1000))
            pbvs["kalman_error_mm"].append(float(np.linalg.norm(xh - can_pose) * 1000))
            pbvs["kalman_2sigma_mm"].append(float(2 * np.sqrt(np.trace(P)) * 1000))
            pbvs["reading_error_mm"].append(float(np.linalg.norm(y - can_pose) * 1000))
            pbvs["red_pixels"].append(int(last_red_pixels))
            if servo_error < 0.002:
                break
        can_pose_estimate = xh
        out["pbvs_photos"] = k + 1 
        e = xh - can_pose #how honest the filter is after the pbvs readings too
        out["kalman_error_end_mm"] = np.linalg.norm(e) * 1000
        out["kalman_nees_end"] = e @ np.linalg.inv(P) @ e
      

        #pick up the can
        viewer.phase = "grab"
        grab_pose = can_pose_estimate + np.array([0, 0, can_height + 0.01]) #0.01 above the top of the can, same as the swift version
        move_gripper(grab_pose)
        out["grasp_offset_mm"] = np.linalg.norm((data.site_xpos[gripper_id] - can_pose)[0:2]) * 1000 #how far off centre the gripper is when it grabs
        if stop_after == "place":
            can_in_gripper = data.qpos[6:9] - data.site_xpos[gripper_id] #where the can is compared to the gripper right now
            move_gripper(grab_pose + np.array([0, 0, 0.3]), item_offset=can_in_gripper) #same spot, 0.3 higher

            #put the can on the blue plate
            viewer.phase = "swing"
            ur3_over_place_q = joints_estimate[:, 0].copy() #where the observer thinks the arm is now
            ur3_over_place_q[0] = ur3_over_place_q[0] + np.pi #spin the base joint half way round, over the rail to the blue plate
            viewer.target = ur3_over_place_q
            UR3_move_joints_lqr(model, data, viewer, ur3_over_place_q, encoder_random, steps=1500, item_offset=can_in_gripper)
            viewer.phase = "place"
            place_pose = np.array([0.75, 0.45, 0.01 + can_height + 0.01]) #middle of the blue plate, can bottom sitting on it
            move_gripper(place_pose, item_offset=can_in_gripper) #lower it onto the plate
            move_gripper(place_pose + np.array([0, 0, 0.15])) #let go and back off
            out["place_error_mm"] = np.linalg.norm(data.qpos[6:8] - place_pose[0:2]) * 1000 
            out["success"] = bool(out["grasp_offset_mm"] < 10 and out["place_error_mm"] < 20) #grabbed within 10 mm of the can centre, placed within 20 mm of the plate centre
        else:
            out["success"] = bool(out["grasp_offset_mm"] < 10)
    except (ValueError, IndexError): #the can left the picture, so there were no red pixels
        out["failed_because"] = "can left the camera picture"
    steps = max(viewer.steps, 1)
    out["observer_angle_rms_mrad"] = np.sqrt(np.mean(viewer.angle_error_sq / steps)) * 1000 #how far the observer's joint angles were from the real ones
    out["observer_speed_rms_mrad_s"] = np.sqrt(np.mean(viewer.speed_error_sq / steps)) * 1000 #same for the joint speeds (never measured)
    for phase in ["look", "swing"]: #% of the move to the look pose, and of the swing to the place plate, with a motor asked for more than its limit
        out[f"saturated_percent_{phase}"] = 100 * viewer.phase_saturated.get(phase, 0) / max(viewer.phase_steps.get(phase, 0), 1)
    out["saturated_percent_all"] = 100 * sum(viewer.phase_saturated.values()) / steps
    out["peak_joint_speed_pbvs"] = viewer.peak_speed.get("pbvs", np.nan)
    for name, values in check.items():
        out[name.replace("_mm", "_max_mm").replace("_deg", "_max_deg")] = max(values) if values else np.nan
    details["check"] = {name: values[:] for name, values in check.items()}
    return out, details


ur3_look_q = [-1.9766, -1.1482, 0.8327, -1.2553, -1.5708, 2.7358] #camera straight down, 0.6 above the plate, over the middle of the spawn area

columns = ["condition", "seed", "can_x", "can_y", "success", "failed_because", "grasp_offset_mm", "place_error_mm", "reprojection_error_px", "red_pixels_at_look",
           "kalman_error_mm", "kalman_sigma_mm", "kalman_nees", "kalman_inside_2sigma", "kalman_error_end_mm", "kalman_nees_end", "pbvs_photos",
           "observer_angle_rms_mrad", "observer_speed_rms_mrad_s", "saturated_percent_look", "saturated_percent_swing", "saturated_percent_all", "peak_joint_speed_pbvs",
           "camera_position_gap_max_mm", "camera_angle_gap_max_deg", "can_reading_gap_max_mm", "gripper_position_gap_max_mm"]

def plain(value): #numpy numbers and true/false into plain python ones, so json can save them
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))

def save_json(name, thing):
    with open(os.path.join(results_folder, name), "w") as f:
        json.dump(thing, f, default=plain)


csv_path = os.path.join(results_folder, "trials.csv")
with open(csv_path, "w", newline="") as f:
    csv.writer(f).writerow(columns)
all_details = []
for condition, settings in conditions.items():
    for random_result in range(number_of_trails): 
        record = condition == "nominal" and random_result == 0
        out, details = run_trial(random_result, settings, record=record)
        out["condition"], details["condition"] = condition, condition
        with open(csv_path, "a", newline="") as f: #saved after every trial, so stopping early keeps what finished
            csv.writer(f).writerow([out.get(c, "") for c in columns])
        all_details.append(details)
        save_json("trial_details.json", all_details)
        if record:
            np.save(os.path.join(results_folder, "joint_recording.npy"), np.array(viewer.rows))
        print(f"{condition:16s} trial {random_result + 1:3d}/{number_of_trails}  success {str(out['success']):5s}  grasp {out.get('grasp_offset_mm', float('nan')):5.1f} mm  "
                f"place {out.get('place_error_mm', float('nan')):5.1f} mm  {out['failed_because']}", flush=True)

def controller_numbers(Q=None, Rn_observer=None):
    Q = LQRPart.angle_and_speed_cost if Q is None else Q
    Rn_observer = LQRPart.encoder_noise if Rn_observer is None else Rn_observer
    A, C = LQRPart.joint_movement_actual, LQRPart.encoder_measurement
    observer_gain, observer_P, observer_poles = ct.lqe(A, np.eye(2), C, LQRPart.model_noise, Rn_observer)
    joints = []
    for i, B in enumerate(LQRPart.torque):
        gain, _, controller_poles = ct.lqr(A, B, Q, LQRPart.torque_cost)
        joints.append({"inertia": float(LQRPart.joint_inertia[i]), "lqr_gain": np.asarray(gain).ravel().tolist(),
                       "controller_poles": [[float(p.real), float(p.imag)] for p in controller_poles],
                       "pole_ratio": float(np.max(np.real(observer_poles)) / np.max(np.real(controller_poles))), #slowest observer pole / slowest controller pole. week 8 A.11 rule of thumb: 2 to 5
                       "torque_limit": float(torque_limit[i])})
    return {"observer_gain": np.asarray(observer_gain).ravel().tolist(), "observer_poles": [[float(p.real), float(p.imag)] for p in observer_poles],
            "observer_sigma": np.sqrt(np.diag(observer_P)).tolist(), "joints": joints}

def look_move(steps=1500, every=1): #the move from all zeros to the look pose, recorded at every sim step
    mujoco.mj_resetData(model, data)
    joints_estimate[:] = 0
    global encoder_random
    encoder_random = np.random.default_rng(2000)
    viewer.reset_counts()
    viewer.recording, viewer.every, viewer.rows, viewer.target, viewer.push_steps_left, viewer.phase = True, every, [], np.array(ur3_look_q), 0, "look"
    UR3_move_joints_lqr(model, data, viewer, ur3_look_q, encoder_random, steps=steps)
    return np.array(viewer.rows)

def summarise_move(r): #numbers from one recorded move
    dt = r[1, 0] - r[0, 0]
    q, qd, q_est, qd_est, target, lqr_torque, ctrl = r[:, 1:7], r[:, 7:13], r[:, 13:19], r[:, 19:25], r[:, 25:31], r[:, 31:37], r[:, 37:43]
    worst = np.degrees(np.max(np.abs(q - target), axis=1))
    outside = np.nonzero(worst > 1.0)[0]
    return {"effort": float(np.sum(lqr_torque ** 2) * dt), #control effort, integral of torque squared (LQR part)
            "tracking_ise": float(np.sum((q - target) ** 2) * dt), #integral of joint error squared, all six joints
            "final_error_deg": float(worst[-1]),
            "settle_s": float(r[outside[-1] + 1, 0] - r[0, 0]) if len(outside) and outside[-1] < len(r) - 1 else (0.0 if not len(outside) else None),
            "peak_torque_fraction": float(np.max(np.abs(ctrl) / torque_limit)),
            "saturated_percent": float(100 * np.mean(np.any(np.abs(ctrl) > torque_limit, axis=1))),
            "speed_estimate_rms": float(np.sqrt(np.mean((qd_est - qd) ** 2))), #rad/s, the observer's speed error
            "angle_estimate_rms_mrad": float(np.sqrt(np.mean((q_est - q) ** 2)) * 1000),
            "torque_chatter": float(np.sqrt(np.mean(np.diff(lqr_torque, axis=0) ** 2)))} #how much the torque jumps step to step. encoder noise getting through to the motors


save_json("controller_info.json", controller_numbers())

original_gains = [g.copy() for g in LQRPart.lqr_gain]
lqr_sweep = []
for weight in [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]:
    for i, B in enumerate(LQRPart.torque):
        LQRPart.lqr_gain[i], _, _ = ct.lqr(LQRPart.joint_movement_actual, B, np.diag([float(weight), 1.0]), LQRPart.torque_cost)
    result = summarise_move(look_move())
    result["weight"] = weight
    lqr_sweep.append(result)
    print(f"lqr sweep Q angle weight {weight:>6}: effort {result['effort']:8.1f}  final error {result['final_error_deg']:.3f} deg  peak torque {100 * result['peak_torque_fraction']:.0f}% of limit", flush=True)
LQRPart.lqr_gain[:] = original_gains 
save_json("lqr_sweep.json", lqr_sweep)

#observer Rn (graph 5). the observer is told the encoder is noisier or quieter than it is
original_observer_gain = LQRPart.observer_gain
observer_sweep = []
for scale in [0.01, 0.1, 1, 10, 100, 1000]:
    Rn_observer = LQRPart.encoder_noise * scale
    LQRPart.observer_gain, _, _ = ct.lqe(LQRPart.joint_movement_actual, np.eye(2), LQRPart.encoder_measurement, LQRPart.model_noise, Rn_observer)
    result = summarise_move(look_move())
    info = controller_numbers(Rn_observer=Rn_observer)
    result["scale"] = scale
    result["pole_ratio"] = [j["pole_ratio"] for j in info["joints"]]
    result["observer_poles"] = info["observer_poles"]
    observer_sweep.append(result)
    print(f"observer sweep Rn x{scale:<6}: speed estimate error {result['speed_estimate_rms']:.3f} rad/s  torque chatter {result['torque_chatter']:.3f}  "
            f"pole ratio {min(result['pole_ratio']):.1f} to {max(result['pole_ratio']):.1f}", flush=True)
LQRPart.observer_gain = original_observer_gain #put observer back
save_json("observer_sweep.json", observer_sweep)

#the joint recording at every sim step with the team's settings, for graphs 4 and 6
np.save(os.path.join(results_folder, "joint_recording_full.npy"), look_move())

#servo gain lambda (graph 8). the pick is run on the first few cans with each gain. lambda * 0.2 s is the share of the error removed per photo
servo_sweep = []
for gain in [0.5, 1.0, 2.0, 4.0, 7.0, 10.0]:
    servo_gain = gain
    for random_result in range(3):
        out, details = run_trial(random_result, conditions["nominal"], stop_after="grab")
        out["servo_gain"] = gain
        out["servo_error_mm"] = details.get("pbvs", {}).get("servo_error_mm", [])
        servo_sweep.append(out)
        print(f"servo gain {gain:4.1f} can {random_result}: photos {out.get('pbvs_photos', '-')}  peak joint speed {out['peak_joint_speed_pbvs']:.2f} rad/s  "
                f"grasp {out.get('grasp_offset_mm', float('nan')):5.1f} mm  {out['failed_because']}", flush=True)
        save_json("servo_gain_sweep.json", servo_sweep)
servo_gain = 2.0 

print("done. now run: python Graphs.py")