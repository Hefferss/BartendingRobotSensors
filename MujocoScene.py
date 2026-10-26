#Builds the MuJoCo scene, with all its items. It also swaps the arm’s built-in motors for plain torque motors so the LQR can drive them.
import os
import mujoco
import numpy as np

scene_folder = os.path.dirname(os.path.abspath(__file__)) #folder this file is in, so the stl and xml paths work from anywhere

def add_stl(spec, name, file_location, pos, euler=[0, 0, 0], rgba=[0.5, 0.5, 0.5, 1], body=None): #adds one of our stls to the scene. same job as Mesh(...) then env.add(...) in swift. stls are in mm so they get scaled by 0.001
    spec.add_mesh(name=name, file=file_location, scale=[0.001, 0.001, 0.001]) #loads the stl as a mesh asset
    if body is None:
        body = spec.worldbody #not given a body means it just sits still in the world
    geom = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, pos=pos, rgba=rgba) #puts the mesh in the scene at pos, in the colour we give it
    geom.alt.type = mujoco.mjtOrientation.mjORIENTATION_EULER #turn it, same as trotz(...) in swift
    geom.alt.euler = euler
    return geom #handed back so it can be turned a different way if needed

def build_scene():
    spec = mujoco.MjSpec.from_file(os.path.join(scene_folder, "universal_robots_ur3e", "ur3e.xml")) #raphaels ur3e from the mujoco_UR3_tutorial
    spec.delete(spec.keys[0]) #the ur3e file comes with a home pose for its 6 joints. the can adds more to the scene so that pose wont fit any more. the home angles get set in main instead

    #UR3 rail
    add_stl(spec, "rail", os.path.join(scene_folder, "RailForUR3.stl"), pos=[-1.5, 0.1, 0], rgba=[0.75, 0.75, 0.75, 1])

    #BasePlatePick.STL. turned so the long side runs along the rail. top covers x 0 to 1.5, y -0.45 to -0.05, at z = 0.01
    add_stl(spec, "plate_pick", os.path.join(scene_folder, "BasePlatePick.STL"), pos=[0, -0.05, 0], euler=[0, 0, -np.pi/2], rgba=[0.18, 0.545, 0.341, 1])

    #BasePlatePlace.STL. other side of the rail. top covers x 0.375 to 1.125, y 0.25 to 0.65, at z = 0.01
    add_stl(spec, "plate_place", os.path.join(scene_folder, "BasePlatePlace.STL"), pos=[0.375, 0.65, 0], euler=[0, 0, -np.pi/2], rgba=[0.118, 0.353, 0.659, 1])

    #UR3 Base. ur3e starts at x = 0.75 on the rail like the swift version, but now slides along it so its rail position can be estimated
    base = spec.body("base")
    base.pos = [0.75, 0.1, 0.25]
    turn_back = np.zeros(4)
    mujoco.mju_negQuat(turn_back, base.quat) #the base is turned around in the ur3e file. this undoes that turn so the rail and plate line up with the world
    rail_axis = np.zeros(3)
    mujoco.mju_rotVecQuat(rail_axis, np.array([1.0, 0, 0]), turn_back) #world x (along the rail) written in the base's own turned frame
    base.add_joint(name="rail_joint", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=rail_axis, range=[-0.75, 0.75], damping=100) #slides along the rail. 0 is x = 0.75, range keeps it over the plates (x 0 to 1.5). damping is the rail friction, also in the rail kf model

    #Purple base plate. stuck to the base so it slides with the ur3e. same world spot as before, just written from the base's point of view
    plate_offset = np.zeros(3)
    mujoco.mju_rotVecQuat(plate_offset, np.array([-0.08, 0, -0.1]), turn_back) #plate was at [0.67, 0.1, 0.15], base at [0.75, 0.1, 0.25], so it sits 8cm back and 10cm under
    plate = add_stl(spec, "ur3_base_plate", os.path.join(scene_folder, "BasePlatForUR3.stl"), pos=plate_offset, rgba=[0.263, 0.059, 0.596, 1], body=base)
    plate.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT #undo the base's turn so the plate faces the same way it did when it sat in the world
    plate.quat = turn_back
    plate.contype = 0 #no collisions for the plate, otherwise it rubs on the rail stl and wont slide. the rail joint already keeps it on the rail
    plate.conaffinity = 0

    #UR3 motors. the ur3e file comes with its own position servos. swapped for plain torque motors so our lqr does the controlling, same as LQRBase.py in swift
    for motor in spec.actuators:
        motor.set_to_motor() #from the mujoco python docs (model editing)
        motor.ctrlrange = motor.forcerange #ctrl is now a torque, so it gets the motors torque limits

    #Rail motor and encoder. added after the loop above so it doesnt get changed. ctrl is a force in N pushing the base along the rail
    rail_motor = spec.add_actuator(name="rail_motor", target="rail_joint", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    rail_motor.set_to_motor()
    rail_motor.ctrlrange = [-200, 200] #force limit in N
    spec.add_sensor(name="rail_encoder", type=mujoco.mjtSensor.mjSENS_JOINTPOS, objtype=mujoco.mjtObj.mjOBJ_JOINT, objname="rail_joint") #measures rail position. mujoco gives the perfect value, the noise gets added in main before the kf sees it

    #Camera. on the wrist looking out the end of the tool, 5cm off to the side, at the realsenses colour lens. fovy 51.28 deg on a 640 x 480 picture gives f = 500 px, same K as the swift version
    wrist = spec.body("wrist_3_link") #last link of the arm, the camera rides on this
    wrist_camera = wrist.add_camera(name="wrist_cam", pos=[0.0325, 0.08, 0.05], fovy=51.28)
    wrist_camera.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES #which way the camera points. mujoco cameras look down their own -z
    wrist_camera.alt.xyaxes = [1, 0, 0, 0, 0, 1]
    camera_model = mujoco.MjSpec.from_file(os.path.join(scene_folder, "realsense_d435i", "d435i.xml")) #intel realsense d435i rgb-d camera model from mujoco menagerie, the camera named in the week 10 slides
    camera_mount = wrist.add_site(name="camera_mount", pos=[0, 0.08, 0.05], quat=[0.7071, -0.7071, 0, 0]) #where the camera body sits on the wrist, centred over it. its colour lens is 3.25cm along the bar, which is where wrist_cam is
    camera_mount.attach_body(camera_model.body("d435i"), "camera_", "") #sticks the realsense onto the wrist

    #Can.STL. free to move so it can be spawned and picked up. stl origin is on its corner so its shifted back half the width, same as can_offset in swift
    can_body = spec.worldbody.add_body(name="can", pos=[0.75, -0.2575, 0.01])
    can_body.add_freejoint(name="can_joint") #6 dof joint so the can can fall, slide and get carried
    add_stl(spec, "can", os.path.join(scene_folder, "Can.STL"), pos=[-0.025, -0.025, 0], rgba=[0.784, 0.063, 0.18, 1], body=can_body)

    model = spec.compile() #turns the spec into the model mujoco actually simulates
    data = mujoco.MjData(model) #holds everything that changes each step (joint angles, speeds, sensor readings)
    return model, data