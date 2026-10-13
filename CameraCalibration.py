import numpy as np
import mujoco
import cv2
from machinevisiontoolbox import CentralCamera, Image

#calibrates the wrist camera the same way as the week 3 tutorial (part b and c). like calibrating the realsense on a bench before putting it on the wrist

#make the checkerboard. 9 x 6 inside corners, same as the week 3 board
gx, gy, sq = 9, 6, 0.03 #inside corners across, inside corners down, square size in metres. same names as the tutorial
board = np.full((9 * 60, 12 * 60), 255, dtype=np.uint8) #white page, 12 x 9 squares of 60 pixels (the board plus a white border)
for j in range(7):
    for i in range(10):
        if (i + j) % 2 == 0: #every second square goes black
            board[(j + 1) * 60:(j + 2) * 60, (i + 1) * 60:(i + 2) * 60] = 0
cv2.imwrite("checkerboard.png", board)

#a scene with just a camera and the checkerboard. built the same way as MujocoScene.py
spec = mujoco.MjSpec()
spec.add_texture(name="checkerboard", type=mujoco.mjtTexture.mjTEXTURE_2D, file="checkerboard.png")
spec.add_material(name="checkerboard", textures=["", "checkerboard"])
board_body = spec.worldbody.add_body(name="board", mocap=True) #mocap means it can be put anywhere for each photo
board_body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[6 * sq, 4.5 * sq, 0.001], material="checkerboard") #12 x 9 squares including the border
spec.worldbody.add_camera(name="bench_cam", fovy=51.28) #same settings as wrist_cam in MujocoScene.py
model = spec.compile()
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 480, 640)

#14 photos of the board at different spots and tilts, like the 14 images in the week 3 tutorial
random = np.random.default_rng(0)
images = []
for k in range(14):
    data.mocap_pos[0] = [random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05), -random.uniform(0.45, 0.65)] #0.45 to 0.65 away, about the same as the can
    mujoco.mju_euler2Quat(data.mocap_quat[0], random.uniform(-0.5, 0.5, 3), "xyz") #tilted up to about 30 degrees each way
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="bench_cam")
    images.append(Image(renderer.render()))

#part b. calibrate. same line as the week 3 tutorial
K, dist, frames = CentralCamera.images2C(images, gridshape=(gx, gy), squaresize=sq)
print(f'{len(frames)} of {len(images)} images used. K =\n', np.round(K, 3)) #f should be about 500, centre about (320, 240)

#part c. reprojection error, same as the tutorial: where the calibrated camera puts the corners vs where they were found
P_board = np.array([[i * sq, j * sq, 0] for j in range(gy) for i in range(gx)]).T #3d corners on the board
cal_cam = CentralCamera(f=K[0, 0], pp=(K[0, 2], K[1, 2]), imagesize=(640, 480), rho=1) #pinhole camera with the calibrated numbers
errors = []
for fr in frames:
    ok, p_meas = cv2.findChessboardCorners(images[fr.id].mono().array, (gx, gy)) #where the corners were found
    cal_cam.pose = fr.pos
    p_proj = cal_cam.project_point(P_board) #where the calibrated camera puts them
    errors.append(np.linalg.norm(p_meas.reshape(-1, 2).T - p_proj, axis=0).mean())
print(f'reprojection error {np.mean(errors):.3f} px') #under 1 pixel is good