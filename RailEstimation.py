#Works out where the robot base is on the rail. the rail sensor is a bit wobbly (noisy), so this cleans it up, and also works out how fast the base is going even though nothing measures speed
#how it works: every step it makes a guess, then checks the guess against the sensor and fixes it. it also keeps a number for how sure it is (P)
import numpy as np
import matplotlib.pyplot as plt

def make_rail_model(dt, mass, damping=100): #writes down the rules for how the base moves. dt is one tiny time step (2ms), mass is how heavy the arm + base is, damping is how much the rail slows it (same 100 as MujocoScene)
    A = np.array([[1, dt], [0, 1 - damping * dt / mass]]) #the base on its own. we track 2 things: where it is and how fast. new spot = old spot + speed x time. new speed = old speed, minus a little lost to rail friction. week 7 A.1 cart
    B = np.array([[0], [dt / mass]]) #what the motor does. pushing makes it speed up, and a heavier base speeds up less (F = ma). week 7 A.1
    C = np.array([[1.0, 0]]) #what the sensor sees. 1 = it sees where the base is. 0 = it cant see speed. the filter figures speed out on its own, week 7 A.4
    Qd = np.diag([1e-8, 1e-3]) #how much we doubt our own rules. a bit more doubt on speed, because the arm swinging shoves the base around. more doubt here = listen to the sensor more. week 7 A.7
    Rn = np.array([[0.002**2]]) #how wobbly the sensor is, about 2mm. squared because thats how the maths wants it. more wobble here = listen to our rules more. week 7 A.7
    return A, B, C, Qd, Rn

def kf_step(xh, P, u, y, A, B, C, Qd, Rn): #one guess-then-fix step. copied from kf_step in week 7 B.3. xh = our guess, P = how unsure we are, u = how hard the motor pushed, y = what the sensor said
    xh_pred = A @ xh + B @ np.atleast_1d(u) #GUESS: use the rules to work out where the base should be now, after the motor pushed
    P_pred = A @ P @ A.T + Qd #guessing without checking makes us a bit less sure
    S = C @ P_pred @ C.T + Rn #how far apart the guess and the sensor could be just from wobble
    K = P_pred @ C.T @ np.linalg.inv(S) #K is a trust slider. close to 1 = believe the sensor, close to 0 = believe our guess. it picks this by comparing how unsure the guess is vs how wobbly the sensor is
    nu = np.atleast_1d(y) - C @ xh_pred #FIX: how far the sensor reading is from what we guessed
    xh_new = xh_pred + K @ nu #move our guess toward the sensor by that much. this fixes the speed too, even though speed was never measured
    P_new = (np.eye(len(xh)) - K @ C) @ P_pred #checking with the sensor makes us more sure
    return xh_new, P_new

def plot_rail_log(rail_log, title): #the report card. shows how close the guess was to the real answer. same check as week 7 B.6
    t, target, est, true, sigma = np.array(rail_log).T #pull the saved numbers out into columns
    error = est - true #how wrong the guess was each step. the real answer (true) is only used here for marking, the robot never sees it
    print(f"rail rms error {np.sqrt(np.mean(error**2))*1000:.2f}mm, inside 2 sigma {np.mean(np.abs(error) < 2*sigma)*100:.0f}% of the time (want about 95%)") #average error, and how often the error stayed inside the filter's own 'I could be this wrong' lines. 95% means the filter is honest
    plt.ioff() #normal plot mode so the window stays open
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(t, target, "g--", label="target") #top graph: where we told the base to go, where it really was, and where the filter thought it was
    ax[0].plot(t, true, "k", lw=2, label="true")
    ax[0].plot(t, est, "r--", label="kf estimate")
    ax[0].set_ylabel("rail position [m]"); ax[0].legend(); ax[0].set_title(title)
    ax[1].plot(t, error*1000, "r", lw=0.8, label="estimate - true") #bottom graph: how wrong the guess was, with the filter's 'I could be this wrong' lines. the red should mostly stay between them
    ax[1].plot(t, 2*sigma*1000, "k--", label="2 sigma the kf claims")
    ax[1].plot(t, -2*sigma*1000, "k--")
    ax[1].set_ylabel("error [mm]"); ax[1].set_xlabel("t [s]"); ax[1].legend()
    plt.tight_layout(); plt.show()