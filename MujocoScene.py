import os
import mujoco
import numpy as np

scene_folder = os.path.dirname(os.path.abspath(__file__))

def add_stl(spec, name, file_location, pos, euler=[0, 0, 0], rgba=[0.5, 0.5, 0.5, 1], body=None): #adds one of our stls to the scene. same job as Mesh(...) then env.add(...) in swift. stls are in mm so they get scaled by 0.001
    spec.add_mesh(name=name, file=file_location, scale=[0.001, 0.001, 0.001])
    if body is None:
        body = spec.worldbody #not given a body means it just sits still in the world
    geom = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, pos=pos, rgba=rgba)
    geom.alt.type = mujoco.mjtOrientation.mjORIENTATION_EULER #turn it, same as trotz(...) in swift
    geom.alt.euler = euler

def build_scene():
    spec = mujoco.MjSpec.from_file(os.path.join(scene_folder, "universal_robots_ur3e", "ur3e.xml")) #raphaels ur3e from the mujoco_UR3_tutorial
    spec.delete(spec.keys[0]) #the ur3e file comes with a home pose for its 6 joints. the can adds more to the scene so that pose wont fit any more. the home angles get set in main instead

    #UR3 rail
    add_stl(spec, "rail", os.path.join(scene_folder, "RailForUR3.stl"), pos=[-1.5, 0.1, 0], rgba=[0.75, 0.75, 0.75, 1])

    #BasePlatePick.STL. turned so the long side runs along the rail. top covers x 0 to 1.5, y -0.45 to -0.05, at z = 0.01
    add_stl(spec, "plate_pick", os.path.join(scene_folder, "BasePlatePick.STL"), pos=[0, -0.05, 0], euler=[0, 0, -np.pi/2], rgba=[0.18, 0.545, 0.341, 1])

    #BasePlatePlace.STL. other side of the rail. top covers x 0.375 to 1.125, y 0.25 to 0.65, at z = 0.01
    add_stl(spec, "plate_place", os.path.join(scene_folder, "BasePlatePlace.STL"), pos=[0.375, 0.65, 0], euler=[0, 0, -np.pi/2], rgba=[0.118, 0.353, 0.659, 1])

    #UR3 Base. purple base plate and the ur3e sitting on it, parked at x = 0.75 on the rail like the swift version
    add_stl(spec, "ur3_base_plate", os.path.join(scene_folder, "BasePlatForUR3.stl"), pos=[0.67, 0.1, 0.15], rgba=[0.263, 0.059, 0.596, 1])
    spec.body("base").pos = [0.75, 0.1, 0.25]

    #UR3 motors. the ur3e file comes with its own position servos. swapped for plain torque motors so our lqr does the controlling, same as LQRBase.py in swift
    for motor in spec.actuators:
        motor.set_to_motor() #from the mujoco python docs (model editing)
        motor.ctrlrange = motor.forcerange #ctrl is now a torque, so it gets the motors torque limits

    #Camera. on the wrist looking out the end of the tool, 5cm off to the side, at the realsenses colour lens. fovy 51.28 deg on a 640 x 480 picture gives f = 500 px, same K as the swift version
    wrist = spec.body("wrist_3_link")
    wrist_camera = wrist.add_camera(name="wrist_cam", pos=[0.0325, 0.08, 0.05], fovy=51.28)
    wrist_camera.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES #which way the camera points. mujoco cameras look down their own -z
    wrist_camera.alt.xyaxes = [1, 0, 0, 0, 0, 1]
    camera_model = mujoco.MjSpec.from_file(os.path.join(scene_folder, "realsense_d435i", "d435i.xml")) #intel realsense d435i rgb-d camera model from mujoco menagerie, the camera named in the week 10 slides
    camera_mount = wrist.add_site(name="camera_mount", pos=[0, 0.08, 0.05], quat=[0.7071, -0.7071, 0, 0]) #where the camera body sits on the wrist, centred over it. its colour lens is 3.25cm along the bar, which is where wrist_cam is
    camera_mount.attach_body(camera_model.body("d435i"), "camera_", "") #sticks the realsense onto the wrist

    #Can.STL. free to move so it can be spawned and picked up. stl origin is on its corner so its shifted back half the width, same as can_offset in swift
    can_body = spec.worldbody.add_body(name="can", pos=[0.75, -0.2575, 0.01])
    can_body.add_freejoint(name="can_joint")
    add_stl(spec, "can", os.path.join(scene_folder, "Can.STL"), pos=[-0.025, -0.025, 0], rgba=[0.784, 0.063, 0.18, 1], body=can_body)

    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data