#The joint controller, adjusted from the LQR part of Swift. It holds the joint model with its controllability and observability checks, the observer that guesses joint speed from noisy encoders, and the LQR (plus gravity hold) that decides how hard each motor pushes. It also runs the rail kf and rail lqr, so the base can drive along the rail during any move.
import time
import numpy as np
import control as ct
import mujoco
from MujocoScene import build_scene
from RailEstimation import make_rail_model, kf_step

#joint model. one UR3 joint as a position servo, same as the dc motor position servo in the week 8 tutorial. the joint speed isnt measured, the observer further down works it out by predicting with this model and correcting with the encoder angle
model, data = build_scene() #our scene, only used here to read how heavy each joint is
mass_matrix = np.zeros((model.nv, model.nv))
mujoco.mj_forward(model, data)
mujoco.mj_fullM(model, data, mass_matrix) #how heavy each joint is to turn, from the ur3e model. mujoco docs (mj_fullM)
joint_inertia = mass_matrix.diagonal()[1:7] #called J in the week 8 tutorial. one for each of the six joints, off the real ur3e instead of the tutorial numbers. [0] is the rail now so the arm starts at 1
friction, motor_constant = 0.0, 1.0 #called b and k in the week 8 tutorial. the ur3e.xml has no joint friction so b = 0, and the motor pushes the joint directly so k = 1
joint_movement_actual = np.array([[0.0, 1.0], [0.0, -friction]]) #called A in the tutorial. angle changes with speed. speed would die off with friction, but theres none
torque = [] #called B in the tutorial. how the motor torque changes the speed. one for each joint, a heavier joint speeds up slower
for J in joint_inertia:
    torque.append(np.array([[0.0], [motor_constant / J]]))
encoder_measurement = np.array([[1.0, 0.0]]) #called C in the tutorial. encoder: position only

#controllability: can the one motor drive both states. week 5 tutorial part C
controllability_matrix = ct.ctrb(joint_movement_actual, torque[0]) #called Co in the tutorial
print("controllable:", np.linalg.matrix_rank(controllability_matrix) == joint_movement_actual.shape[0]) #True means the motor can move both the angle and the speed

#observability: can both states be worked out from the encoder alone. week 5 tutorial part C4
observability_matrix = ct.obsv(joint_movement_actual, encoder_measurement) #called Ob in the tutorial
print("observable:", np.linalg.matrix_rank(observability_matrix) == joint_movement_actual.shape[0]) #True means the speed can be worked out from the angle readings

#observer gain: how hard to correct the guess when the encoder disagrees with it. week 8 tutorial servo cell
model_noise = np.diag([1e-3, 1.0]) #called Qd in the tutorial. how wrong the joint model might be. tutorial numbers from week 8 stuff from the "DC motor position" stuff
encoder_noise = np.array([[1e-5]]) #called Rn in the tutorial. how noisy the encoder is. same as above
observer_gain, _, observer_poles = ct.lqe(joint_movement_actual, np.eye(2), encoder_measurement, model_noise, encoder_noise) #called Kf in the tutorial
print("observer gain:", observer_gain.ravel(), "observer poles:", observer_poles) #minus sign on the poles means the guess gets better over time, not worse

#lqr gain: works out how hard the motor should push. week 8 tutorial servo cell
angle_and_speed_cost = np.diag([100.0, 1.0]) #called Q in the tutorial. being off on the angle matters 100 times more than being off on the speed
torque_cost = np.array([[1.0]]) #called R in the tutorial. how much we care about not overworking the motor
lqr_gain = [] #called Kr in the tutorial. one for each joint. tells the system how hard to push the motor when the angle and speed are off
for B in torque:
    gain, _, _ = ct.lqr(joint_movement_actual, B, angle_and_speed_cost, torque_cost)
    lqr_gain.append(gain)
print("lqr gains:", np.vstack(lqr_gain)) #one row per joint

#rail. same kf and lqr as RailEstimationTest. the rail encoder gives position only, the kf gets position and speed, the rail lqr drives the base using only that guess
rail_A, rail_B, rail_C, rail_Qd, rail_Rn = make_rail_model(model.opt.timestep, model.body_subtreemass[model.body("base").id])
rail_K, _, _ = ct.dlqr(rail_A, rail_B, np.diag([1000, 1]), np.array([[0.01]])) #week 6 lqr, dlqr because the model is stepped every 2ms
rail_estimate = np.zeros(2) #the rail kf's guess of [position, speed]. 0 is x = 0.75. kept out here so its remembered between moves, like joints_estimate
rail_P = np.diag([0.01**2, 0.01**2]) #how unsure the rail guess is, starts about 1cm
rail_target = np.zeros(1) #where the rail is meant to be. stays put until a move gives it a new spot
rail_log = [] #time, target, estimate, true, sigma every step. true only used to score it and plot it, never by the controller

#move the arm with the lqr and the observer. same loop as swift, once for each of the six joints. mujoco moves the real joints now instead of working them out here
joints_estimate = np.zeros((6, 2)) #the observer's guess of [angle, speed] for the six joints. starts at all zeros like the arm. kept out here so its remembered between moves, like ur3.q in swift
def UR3_move_joints_lqr(model, data, viewer, target_q, encoder_random, steps=1000, item_offset=None, rail_to=None): #the move function. give it the arm, the angles to go to, the encoder's random numbers, and optionally a new rail spot
    if rail_to is not None:
        rail_target[0] = rail_to #new spot for the base on the rail
    gravity_data = mujoco.MjData(model) #a copy of the robot just for working out gravity. swift had no gravity so didnt need this
    for k in range(steps): #do 1000 steps (2 seconds)
        gravity_data.qpos[1:7] = joints_estimate[:, 0] #put the copy at the guessed angles. qpos[0] is the rail, the arm is 1 to 6
        mujoco.mj_forward(model, gravity_data) #works out how hard gravity pulls on each joint there. ends up in qfrc_bias, mujoco docs
        for i in range(6): #in each step, go through the six joints one at a time
            joint_target = np.array([target_q[i], 0.0]) #where this joint should end up, with speed 0
            motor_torque = (-lqr_gain[i] @ (joints_estimate[i] - joint_target)).item() #the lqr decides how hard to push, using only the guess
            data.ctrl[i] = motor_torque + gravity_data.qfrc_bias[i + 1] #push the motor that hard, plus enough to hold the arm up against gravity. i + 1 because the rail is first
            encoder_reading = data.qpos[i + 1] + np.sqrt(encoder_noise[0, 0]) * encoder_random.standard_normal(1) #the encoder reads the angle, with a bit of noise on it
            encoder_error = encoder_reading - encoder_measurement @ joints_estimate[i] #how far the guess is from the encoder reading
            joints_estimate[i] = joints_estimate[i] + model.opt.timestep * (joint_movement_actual @ joints_estimate[i] + torque[i].ravel() * motor_torque + observer_gain @ encoder_error) #the guess moves the same way, then gets nudged by that error
        rail_force = np.clip(-rail_K @ (rail_estimate - [rail_target[0], 0]), -200, 200).item() #rail lqr on the guess, not the real rail position
        data.ctrl[6] = rail_force
        mujoco.mj_step(model, data) #the real joints move from that push. same job as env.step in swift
        rail_reading = data.sensor("rail_encoder").data + 0.002 * encoder_random.standard_normal(1) #rail encoder with 2mm noise
        rail_estimate[:], rail_P[:] = kf_step(rail_estimate, rail_P, rail_force, rail_reading, rail_A, rail_B, rail_C, rail_Qd, rail_Rn) #week 7 kf step. [:] fills in the same arrays so main sees the new guess
        rail_log.append([data.time, rail_target[0], rail_estimate[0], data.qpos[0], np.sqrt(rail_P[0, 0])])
        if item_offset is not None: #if its carrying the can
            data.qpos[7:10] = data.site_xpos[model.site("attachment_site").id] + item_offset #move the can with the gripper. qpos[7:10] is the can now the rail is first
            data.qvel[7:13] = 0 #stop gravity pulling it out of the gripper
        viewer.sync() #draw the next frame
        time.sleep(model.opt.timestep) #slows it down to about real speed