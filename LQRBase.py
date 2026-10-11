import numpy as np
import control as ct
from UR3 import update_camera_mesh

#joint model. one UR3 joint as a position servo, same as the dc motor position servo in the week 8 tutorial
joint_inertia, friction, motor_constant = 0.01, 0.1, 0.05 #called J, b, k in the week 8 tutorial. the tutorial numbers, not measured off a real UR3
joint_movement_actual = np.array([[0.0, 1.0], [0.0, -friction / joint_inertia]]) #called A in the tutorial. angle changes with speed. speed dies off with friction
torque = np.array([[0.0], [motor_constant / joint_inertia]]) #called B in the tutorial. how the motor torque changes the speed
encoder_measurement = np.array([[1.0, 0.0]]) #called C in the tutorial. encoder: position only

#controllability: can the one motor drive both states. week 5 tutorial part C
controllability_matrix = ct.ctrb(joint_movement_actual, torque) #called Co in the tutorial
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
lqr_gain, _, lqr_poles = ct.lqr(joint_movement_actual, torque, angle_and_speed_cost, torque_cost) #called Kr in the tutorial. tellls the system how hard to push the motor when the angle and speed are off.
print("lqr gain:", lqr_gain.ravel(), "lqr poles:", lqr_poles) #minus sign on the poles means the joint stops on the target

#lqr check: drive the joint to a target angle using only the guess. same loop as servo_sim in the week 8 tutorial
time_step = 0.001 #called dts in the tutorial. how long one step is, in seconds
joint_target = np.array([1.0, 0.0]) #called xref in the tutorial. want the angle at 1 and the speed at 0
joint_actual = np.zeros(2) #called x in the tutorial. the real [angle, speed]
joint_estimate = np.array([0.0, 20.0]) #called xh in the tutorial. the guess. starts with the speed wrong by 20, same as the tutorial
for k in range(3000):
    motor_torque = (-lqr_gain @ (joint_estimate - joint_target)).item() #called u in the tutorial. how hard to push, worked out from the guess only
    encoder_reading = encoder_measurement @ joint_actual #called y in the tutorial. the encoder only gives the angle
    joint_actual = joint_actual + time_step * (joint_movement_actual @ joint_actual + torque.ravel() * motor_torque) #the real joint moves
    encoder_error = encoder_reading - encoder_measurement @ joint_estimate #how far the guess is from what the encoder says
    joint_estimate = joint_estimate + time_step * (joint_movement_actual @ joint_estimate + torque.ravel() * motor_torque + observer_gain @ encoder_error) #the guess moves like the real joint, then gets nudged by that error
print("target angle:", joint_target[0], "real angle:", joint_actual[0], "guessed angle:", joint_estimate[0]) #all three should match

#move the arm with the lqr and the observer. same loop as the check above, once for each of the six joints
swift_time_step = 0.01 #each step of the scene is a hundredth of a second, SAME METHOD AS ir
def UR3_move_joints_lqr(env, ur3, target_q, encoder_random, steps=200, camera_mesh=None, item=None, item_offset=None): #the new move function. give it the arm, the angles to go to, and the encoder's random numbers
    joints_actual = np.column_stack([ur3.q, np.zeros(ur3.n)]) #where the six joints really are, and how fast they're going. they start still
    joints_estimate = np.column_stack([ur3.q + np.sqrt(encoder_noise[0, 0]) * encoder_random.standard_normal(ur3.n), np.zeros(ur3.n)]) #the observer's guess for the six joints. starts from one noisy encoder reading, with speed 0
    for k in range(steps): #do 200 steps (2 seconds)
        for i in range(ur3.n): #in each step, go through the six joints one at a time
            joint_target = np.array([target_q[i], 0.0]) #where this joint should end up, with speed 0
            motor_torque = (-lqr_gain @ (joints_estimate[i] - joint_target)).item() #the lqr decides how hard to push, using only the guess
            encoder_reading = encoder_measurement @ joints_actual[i] + np.sqrt(encoder_noise[0, 0]) * encoder_random.standard_normal(1) #the encoder reads the angle, with a bit of noise on it
            joints_actual[i] = joints_actual[i] + swift_time_step * (joint_movement_actual @ joints_actual[i] + torque.ravel() * motor_torque) #the real joint moves from that push
            encoder_error = encoder_reading - encoder_measurement @ joints_estimate[i] #how far the guess is from the encoder reading
            joints_estimate[i] = joints_estimate[i] + swift_time_step * (joint_movement_actual @ joints_estimate[i] + torque.ravel() * motor_torque + observer_gain @ encoder_error) #the guess moves the same way, then gets nudged by that error
        ur3.q = joints_actual[:, 0] #put the real angles on the robot so you see it move
        update_camera_mesh(ur3, camera_mesh) #keep the camera model stuck to the wrist
        if item is not None: #if its carrying the can
            item.T = (ur3.fkine(ur3.q) * item_offset).A #move the can with the gripper
        env.step(swift_time_step) #draw the next frame