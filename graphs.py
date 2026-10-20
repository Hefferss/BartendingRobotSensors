import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") #the mean of a condition where every trial failed is nan, which numpy warns about

os.chdir(os.path.dirname(os.path.abspath(__file__))) 
os.makedirs("figures", exist_ok=True)

def load(name): 
    path = os.path.join("results", name)
    if not os.path.exists(path):
        return None
    if name.endswith(".json"):
        return json.load(open(path))
    if name.endswith(".npy"):
        return np.load(path)
    return list(csv.DictReader(open(path)))

def column(rows, name, condition): #one column of trials.csv for one condition, as numbers, blanks left out
    values = [float(r[name]) for r in rows if r["condition"] == condition and r[name] not in ("", "nan")]
    return np.array(values)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join("figures", name))
    plt.close(fig)
    print("saved", name)

rows = load("trials.csv")
details = load("trial_details.json")
controller = load("controller_info.json")
recording = load("joint_recording_full.npy")
lqr_sweep = load("lqr_sweep.json")
observer_sweep = load("observer_sweep.json")
servo_sweep = load("servo_gain_sweep.json")
conditions = list(dict.fromkeys(r["condition"] for r in rows)) if rows else []
nominal = [d for d in details if d["condition"] == "nominal" and len(d.get("estimates", [])) == 20] if details else []


#graph 1: kalman estimate, 2 sigma and the true position for the first nominal trial
if nominal:
    d = nominal[0]
    truth = np.array(d["truth"])
    readings = (np.array(d["readings"]) - truth) * 1000 #error from the true position, in mm
    estimates = (np.array(d["estimates"]) - truth) * 1000
    sigma = np.sqrt(np.array(d["P_diagonal"])) * 1000
    n = np.arange(1, 21)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for i, name in enumerate(["x", "y", "z"]):
        axes[i].fill_between(n, estimates[:, i] - 2 * sigma[:, i], estimates[:, i] + 2 * sigma[:, i], alpha=0.2, label="±2Sigma from P")
        axes[i].plot(n, readings[:, i], "o", label="single reading")
        axes[i].plot(n, estimates[:, i], label="kalman estimate")
        axes[i].axhline(0, color="k", linestyle="--", label="true position")
        axes[i].set_title(name)
        axes[i].set_xlabel("reading number")
        axes[i].set_ylabel("error from true position (mm)")
    axes[0].legend()
    save(fig, "graph_1_kalman_estimate_vs_truth.png")

#graph 2: NEES for every nominal trial. an honest filter averages 3, and 95% of trials are under 7.8
if rows:
    look = column(rows, "kalman_nees", "nominal")
    end = column(rows, "kalman_nees_end", "nominal")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(look, np.zeros(len(look)), "o", label="after 20 look-pose readings")
    ax.plot(end, np.ones(len(end)), "o", label="after the visual servoing")
    ax.axvline(3, color="k", linestyle="--", label="3 = honest average")
    ax.axvline(7.815, color="grey", linestyle=":", label="7.8 = 95% line")
    ax.set_xscale("log")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["look pose", "after servoing"])
    ax.set_xlabel("NEES = eᵀ P⁻¹ e  (one dot per trial)")
    ax.set_title("Kalman filter consistency: above 7.8 means over-confident")
    ax.legend()
    save(fig, "graph_2_kalman_consistency.png")

#graph 3: the saved readings put through the filter again with Rn scaled
if nominal:
    Rn = np.diag([(0.45 / 499.766) ** 2, (0.45 / 499.766) ** 2, (0.0015 + 0.0022 * 0.45 ** 2) ** 2]) #the main file's Rn
    scales = [0.01, 0.1, 1, 10, 100, 1000]
    error_after_3, error_after_20, average_nees = [], [], []
    for scale in scales:
        e3, e20, nees = [], [], []
        for d in nominal:
            xh, P = np.array([0.75, -0.2575, 0.01]), np.eye(3) * 0.2 ** 2
            for k, y in enumerate(d["readings"]):
                P = P + np.eye(3) * 1e-8
                K = P @ np.linalg.inv(P + Rn * scale)
                xh = xh + K @ (np.array(y) - xh)
                P = (np.eye(3) - K) @ P
                if k == 2:
                    e3.append(np.linalg.norm(xh - d["truth"]) * 1000)
            e = xh - np.array(d["truth"])
            e20.append(np.linalg.norm(e) * 1000)
            nees.append(e @ np.linalg.inv(P) @ e)
        error_after_3.append(np.mean(e3))
        error_after_20.append(np.mean(e20))
        average_nees.append(np.mean(nees))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(scales, error_after_3, "o-", label="after 3 readings")
    axes[0].plot(scales, error_after_20, "o-", label="after 20 readings")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Rn scale (1 = team's)")
    axes[0].set_ylabel("mean position error (mm)")
    axes[0].set_title("Accuracy vs Rn")
    axes[0].legend()
    axes[1].plot(scales, average_nees, "o-")
    axes[1].axhline(3, color="k", linestyle="--", label="honest = 3")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Rn scale (1 = team's)")
    axes[1].set_ylabel("average NEES")
    axes[1].set_title("Consistency vs Rn")
    axes[1].legend()
    save(fig, "graph_3_rn_tradeoff.png")

#graphs 4 and 6: the move to the look pose 
if recording is not None:
    t = recording[:, 0] - recording[0, 0]
    q, qd = recording[:, 1:7], recording[:, 7:13] #true angles and speeds
    q_est, qd_est = recording[:, 13:19], recording[:, 19:25] #the observer's guess
    target, ctrl = recording[:, 25:31], recording[:, 37:43] #where each joint is going, torque asked for

    #graph 4: shoulder speed, never measured, vs the observer's estimate
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    first_2s = t <= 2
    axes[0].plot(t[first_2s], qd[first_2s, 1], "k", label="true shoulder speed (never measured)")
    axes[0].plot(t[first_2s], qd_est[first_2s, 1], "--", label="observer estimate")
    axes[0].set_ylabel("speed (rad/s)")
    axes[0].set_title("Observer: joint speed worked out from the encoder angle alone")
    axes[0].legend()
    axes[1].plot(t[first_2s], qd_est[first_2s, 1] - qd[first_2s, 1], label="estimate - truth")
    if controller:
        speed_sigma = controller["observer_sigma"][1] #steady-state sigma from ct.lqe's P
        axes[1].axhline(2 * speed_sigma, color="grey", linestyle="--", label="±2Sigma the observer claims")
        axes[1].axhline(-2 * speed_sigma, color="grey", linestyle="--")
    axes[1].set_ylabel("speed error (rad/s)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend()
    save(fig, "graph_4_observer_vs_truth.png")

    #graph 6: joint error and torque as a % of each motor's limit
    limits = np.array([56, 56, 28, 12, 12, 12]) #from the ur3e.xml forcerange
    names = ["base", "shoulder", "elbow", "wrist 1", "wrist 2", "wrist 3"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    first_3s = t <= 3
    for j in range(6):
        axes[0].plot(t[first_3s], np.degrees(q[first_3s, j] - target[first_3s, j]), label=names[j])
        axes[1].plot(t[first_3s], 100 * np.abs(ctrl[first_3s, j]) / limits[j])
    axes[1].axhline(100, color="k", linestyle="--", label="motor limit")
    axes[0].set_ylabel("joint error (deg)")
    axes[0].set_title("LQR tracking: move from all zeros to the look pose")
    axes[0].legend(ncol=6)
    axes[1].set_ylabel("torque asked for (% of limit)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend()
    save(fig, "graph_6_joint_tracking.png")

#graph 5: observer Rn sweep
if observer_sweep:
    scale = [s["scale"] for s in observer_sweep]
    ratio = np.array([s["pole_ratio"] for s in observer_sweep])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].plot(scale, [s["speed_estimate_rms"] for s in observer_sweep], "o-")
    axes[0].set_ylabel("speed estimate RMS error (rad/s)")
    axes[0].set_title("Speed estimate error")
    axes[1].plot(scale, [s["torque_chatter"] for s in observer_sweep], "o-")
    axes[1].set_ylabel("RMS torque jump per step (N·m)")
    axes[1].set_title("Encoder noise reaching the motors")
    for j, name in enumerate(["base", "shoulder", "elbow", "wrist 1", "wrist 2", "wrist 3"]):
        axes[2].plot(scale, ratio[:, j], "o-", label=name)
    axes[2].axhspan(2, 5, alpha=0.15, label="2 to 5 rule of thumb")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("observer pole ÷ LQR pole")
    axes[2].set_title("How much faster the observer is")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("observer's Rn scale (1 = team's)")
    save(fig, "graph_5_observer_speed_tradeoff.png")

#graph 7: LQR Q weight sweep
if lqr_sweep:
    weight = [s["weight"] for s in lqr_sweep]
    effort = [s["effort"] for s in lqr_sweep]
    tracking = [s["tracking_ise"] for s in lqr_sweep]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(effort, tracking, "o-")
    for w, x, y in zip(weight, effort, tracking):
        axes[0].annotate(f"Q={w}", (x, y), fontsize=7)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("control effort ∫τ² dt")
    axes[0].set_ylabel("tracking error ∫Σe² dt")
    axes[0].set_title("Trade-off curve (team's Q = 100)")
    axes[1].plot(weight, [100 * s["peak_torque_fraction"] for s in lqr_sweep], "o-")
    axes[1].axhline(100, color="k", linestyle="--", label="motor limit")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Q angle weight")
    axes[1].set_ylabel("peak torque asked for (% of limit)")
    axes[1].set_title("Where the motors saturate")
    axes[1].legend()
    save(fig, "graph_7_lqr_weight_tradeoff.png")

#graph 8: servo gain sweep. trials that lost the can are left out of the photos plot
if servo_sweep:
    gains = sorted(set(s["servo_gain"] for s in servo_sweep))
    photos = [np.mean([s["pbvs_photos"] for s in servo_sweep if s["servo_gain"] == g and not s["failed_because"]] or [np.nan]) for g in gains]
    speed = [max(s["peak_joint_speed_pbvs"] for s in servo_sweep if s["servo_gain"] == g) for g in gains]
    lost = [sum(1 for s in servo_sweep if s["servo_gain"] == g and s["failed_because"]) for g in gains]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].plot(gains, photos, "o-")
    axes[0].set_ylabel("photos until within 2 mm")
    axes[0].set_title("Time to reach the can")
    axes[1].plot(gains, speed, "o-")
    axes[1].axhline(3, color="k", linestyle="--", label="about 3 rad/s: real arm limit")
    axes[1].set_ylabel("peak joint speed (rad/s)")
    axes[1].set_title("Peak joint speed while servoing")
    axes[1].legend()
    axes[2].bar(gains, lost, width=0.5)
    axes[2].set_ylabel("trials that lost the can")
    axes[2].set_title("Lost the can")
    for ax in axes:
        ax.set_xlabel("servo gain λ (team's = 2)")
    save(fig, "graph_8_servo_gain_tradeoff.png")

#graph 9: reprojection error and depth error
if rows:
    measured = np.concatenate([d.get("depth_reading", []) for d in details if d["condition"] == "nominal"])
    true = np.concatenate([d.get("depth_true", []) for d in details if d["condition"] == "nominal"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].hist(column(rows, "reprojection_error_px", "nominal"), bins=15)
    axes[0].set_xlabel("reprojection error (pixels)")
    axes[0].set_ylabel("number of trials")
    axes[0].set_title("Camera check at the look pose")
    axes[1].plot(true, (measured - true) * 1000, ".", alpha=0.4, label="depth reading − true distance")
    distance = np.linspace(true.min(), true.max(), 50)
    axes[1].plot(distance, 2000 * (0.0015 + 0.0022 * distance ** 2), "r--", label="±2σ noise model")
    axes[1].plot(distance, -2000 * (0.0015 + 0.0022 * distance ** 2), "r--")
    axes[1].set_xlabel("true distance to the can top (m)")
    axes[1].set_ylabel("depth error (mm)")
    axes[1].set_title("Depth readings during the visual servoing")
    axes[1].legend()
    save(fig, "graph_9_sensing.png")

#graph 10: the visual servoing photo by photo, first trial of each condition
if details:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for c in conditions:
        d = next((d for d in details if d["condition"] == c and d.get("pbvs", {}).get("true_distance_mm")), None)
        if d is None:
            continue
        n = np.arange(1, len(d["pbvs"]["true_distance_mm"]) + 1)
        axes[0].plot(n, d["pbvs"]["true_distance_mm"], label=c)
        axes[1].plot(n, d["pbvs"]["kalman_error_mm"], label=c)
        axes[2].plot(n, d["pbvs"]["red_pixels"], label=c)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("gripper to target (mm)")
    axes[0].set_title("Visual servoing photo by photo (push at photo 21)")
    axes[0].legend()
    axes[1].set_ylabel("can position error (mm)")
    axes[2].set_ylabel("red pixels in the photo")
    axes[2].set_xlabel("photo number")
    save(fig, "graph_10_pbvs_convergence.png")

#graph 11: grasp and place error for every trial, one column of dots per condition
if rows:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, c in enumerate(conditions):
        axes[0].plot([i] * len(column(rows, "grasp_offset_mm", c)), column(rows, "grasp_offset_mm", c), "o", alpha=0.6)
        axes[1].plot([i] * len(column(rows, "place_error_mm", c)), column(rows, "place_error_mm", c), "o", alpha=0.6)
    labels = [f"{c}\n{100 * np.mean([r['success'] == 'True' for r in rows if r['condition'] == c]):.0f}% success" for c in conditions]
    for ax, tolerance, name in [(axes[0], 10, "grasp offset from can centre (mm)"), (axes[1], 20, "place error from plate centre (mm)")]:
        ax.axhline(tolerance, color="k", linestyle="--", label=f"tolerance {tolerance} mm")
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(name)
        ax.legend()
    axes[0].set_title("Grasp accuracy")
    axes[1].set_title("Place accuracy")
    save(fig, "graph_11_robustness.png")

#summary table for the project page: one row per condition
if rows:
    lines = ["| condition | trials | success | grasp offset (mm) | place error (mm) | kalman error (mm) | NEES under 7.8 | failures |",
             "|---|---|---|---|---|---|---|---|"]
    for c in conditions:
        trials = [r for r in rows if r["condition"] == c]
        success = np.mean([r["success"] == "True" for r in trials]) * 100
        failures = sum(1 for r in trials if r["failed_because"])
        nees = column(rows, "kalman_nees", c)
        lines.append(f"| {c} | {len(trials)} | {success:.0f}% | {column(rows, 'grasp_offset_mm', c).mean():.1f} | {column(rows, 'place_error_mm', c).mean():.1f} | "
                     f"{column(rows, 'kalman_error_mm', c).mean():.1f} | {np.mean(nees < 7.815) * 100:.0f}% | {failures} |")
    open(os.path.join("figures", "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
