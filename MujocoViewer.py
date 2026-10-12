import mujoco
import mujoco.viewer
from MujocoScene import build_scene

#opens the scene and holds the arm at its home pose. same as simulate_ur3e.py from raphaels mujoco_UR3_tutorial, just using our scene
model, data = build_scene()
ur3_home_q = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0] #home pose from the ur3e file. points the wrist camera straight down at the green plate
data.qpos[0:6] = ur3_home_q #qpos[0:6] is the six arm joints, qpos[6:13] is the can
data.ctrl[:] = ur3_home_q #the arm motors hold whatever angle is in ctrl

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()