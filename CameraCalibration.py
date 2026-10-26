#Works out the camera's K numbers (its zoom and picture centre) the same way as the week 3 tutorial (part b and c). like calibrating the realsense on a bench before putting it on the wrist. main uses the K this prints out
import numpy as np
import mujoco
import cv2
from machinevisiontoolbox import CentralCamera, Image

#make a checkerboard picture. 9 x 6 inside corners (where 4 squares meet), same as the week 3 board
corners_across, corners_down, square_size = 9, 6, 0.03 #corners across, corners down, and each square is 3cm. called gx, gy, sq in the tutorial
board = np.full((9 * 60, 12 * 60), 255, dtype=np.uint8) #a white page, room for 12 x 9 squares of 60 pixels (the board plus a white border)
for j in range(7):
    for i in range(10):
        if (i + j) % 2 == 0: #every second square goes black, like a chess board
            board[(j + 1) * 60:(j + 2) * 60, (i + 1) * 60:(i + 2) * 60] = 0
cv2.imwrite("checkerboard.png", board) #save it so mujoco can paint it onto a board

#a little world with just a camera and the checkerboard. built the same way as MujocoScene.py
spec = mujoco.MjSpec()
spec.add_texture(name="checkerboard", type=mujoco.mjtTexture.mjTEXTURE_2D, file="checkerboard.png") #load the picture
spec.add_material(name="checkerboard", textures=["", "checkerboard"]) #turn it into a "paint" we can put on a shape
board_body = spec.worldbody.add_body(name="board", mocap=True) #mocap = we can move it anywhere by hand for each photo
board_body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[6 * square_size, 4.5 * square_size, 0.001], material="checkerboard") #a thin flat box with the checkerboard painted on. 12 x 9 squares including the border
spec.worldbody.add_camera(name="bench_cam", fovy=51.28) #same settings as wrist_cam in MujocoScene.py, so the K we get is the wrist camera's K
model = spec.compile()
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 480, 640) #takes 640 x 480 photos

#take 14 photos of the board at different spots and tilts, like the 14 images in the week 3 tutorial
random = np.random.default_rng(0) #fixed random numbers so it gives the same answer every time
images = []
for k in range(14):
    data.mocap_pos[0] = [random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05), -random.uniform(0.45, 0.65)] #move the board a bit left/right/up/down, and 0.45 to 0.65m away (about how far the can is)
    mujoco.mju_euler2Quat(data.mocap_quat[0], random.uniform(-0.5, 0.5, 3), "xyz") #tilt it up to about 30 degrees each way. tilted views are what let the calibration work out the zoom
    mujoco.mj_forward(model, data) #update the world with the new board spot
    renderer.update_scene(data, camera="bench_cam")
    images.append(Image(renderer.render())) #take the photo

#part b. calibrate: find the checkerboard corners in every photo and work out the K that explains them all. same line as the week 3 tutorial
K, dist, frames = CentralCamera.images2C(images, gridshape=(corners_across, corners_down), squaresize=square_size) #K = zoom and centre, dist = lens bend, frames = where the board was in each photo
print(f'{len(frames)} of {len(images)} images used. K =\n', np.round(K, 3)) #zoom should be about 500, centre about (320, 240). these go into camera_matrix in main

#part c. reprojection error: check the K is right. using K, work out where the corners SHOULD show up, and compare to where they actually showed up. same as the tutorial
P_board = np.array([[i * square_size, j * square_size, 0] for j in range(corners_down) for i in range(corners_across)]).T #the real 3d spot of every corner on the flat board
cal_cam = CentralCamera(f=K[0, 0], pp=(K[0, 2], K[1, 2]), imagesize=(640, 480), rho=1) #a pretend camera using our K
errors = []
for calibration_result in frames: #for each photo
    ok, corners_found = cv2.findChessboardCorners(images[calibration_result.id].mono().image, (corners_across, corners_down)) #where the corners actually are in the photo. .image like measured_corners in the tutorial
    cal_cam.pose = calibration_result.pose #put the pretend camera where the real one was for this photo. called fr.pose in the tutorial
    errors.append(np.linalg.norm(corners_found.reshape(-1, 2).T - cal_cam.project_point(P_board), axis=0).mean()) #average pixel gap between "where they are" and "where K says they should be"
print(f'reprojection error {np.mean(errors):.3f} px') #under 1 pixel means the K is good