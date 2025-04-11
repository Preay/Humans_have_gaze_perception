import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import threading
import mediapipe as mp
from PIL import Image, ImageTk
import time
import csv
import os

# Gaze estimation parameters
g_space = np.linspace(-30, 30, 500)
sigma_e = 5
sigma_h = 10
sigma_p = 10

# Gaussian functions
def gaussian(x, mu, sigma):
    return norm.pdf(x, mu, sigma)

def compute_posterior(g_est_eye, g_est_head):
    P_eye = gaussian(g_space, g_est_eye, sigma_e)
    P_head = gaussian(g_space, g_est_head, sigma_h)
    P_prior = gaussian(g_space, 0, sigma_p)
    P_post_unnorm = P_eye * P_head * P_prior
    P_post = P_post_unnorm / np.sum(P_post_unnorm)
    g_est = g_space[np.argmax(P_post)]
    return g_space, P_post, g_est, P_eye, P_head, P_prior

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

cap = cv2.VideoCapture(0)
g_est_eye = 0
g_est_head = 0
latest_frame = None

# Ensure snapshot directory exists
os.makedirs("snapshots", exist_ok=True)

# Webcam processing thread
def process_camera():
    global g_est_eye, g_est_head, latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        latest_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            eye_center_x = (landmarks[33].x + landmarks[263].x) / 2
            head_center_x = landmarks[1].x
            g_est_eye = (eye_center_x - 0.5) * 60
            g_est_head = (head_center_x - 0.5) * 60

# Start the thread
threading.Thread(target=process_camera, daemon=True).start()

# GUI update logic
def update_plot():
    global latest_frame

    g_space, P_post, g_est, P_eye, P_head, P_prior = compute_posterior(g_est_eye, g_est_head)

    # Category and direct gaze probability
    if abs(g_est) < 5:
        category = "Direct"
    elif g_est > 0:
        category = "Right"
    else:
        category = "Left"

    prob_direct = np.sum(P_post[(g_space >= -5) & (g_space <= 5)])

    # Plot
    ax.clear()
    ax.plot(g_space, P_post, label='Posterior', linewidth=2)
    ax.plot(g_space, P_eye, color='blue', alpha=0.3, label='Eye Likelihood')
    ax.plot(g_space, P_head, color='orange', alpha=0.3, label='Head Likelihood')
    ax.plot(g_space, P_prior, color='gray', linestyle=':', label='Prior')
    ax.axvline(g_est_eye, color='blue', linestyle='--', label='Eye Obs')
    ax.axvline(g_est_head, color='orange', linestyle='--', label='Head Obs')
    ax.axvline(g_est, color='red', linestyle='--', label='Estimated Gaze')
    ax.set_title("Live Bayesian Gaze Estimation")
    ax.set_xlabel("Gaze Angle (°)")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid(True)
    canvas.draw()

    # Text updates
    label_eye.config(text=f"Eye Obs: {g_est_eye:.2f}°")
    label_head.config(text=f"Head Obs: {g_est_head:.2f}°")
    label_result.config(text=f"Estimated Gaze: {g_est:.2f}°")
    label_category.config(text=f"Gaze Category: {category}  |  P(Direct): {prob_direct:.2f}")

    # Camera feed
    if latest_frame is not None:
        frame_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize((300, 225))
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)

        # Save snapshot
        timestamp = int(time.time())
        img_path = f"snapshots/{timestamp}.png"
        cv2.imwrite(img_path, latest_frame)

        # Save to CSV
        with open("gaze_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, g_est_eye, g_est_head, g_est, category, prob_direct, img_path])

    root.after(1000, update_plot)

# GUI setup
root = tk.Tk()
root.title("Live Bayesian Gaze Perception Simulator")
root.geometry("850x850")

label_eye = ttk.Label(root, text="Eye Obs: --")
label_eye.pack(pady=2)

label_head = ttk.Label(root, text="Head Obs: --")
label_head.pack(pady=2)

label_result = ttk.Label(root, text="Estimated Gaze: --")
label_result.pack(pady=2)

label_category = ttk.Label(root, text="Gaze Category: --")
label_category.pack(pady=5)

camera_label = ttk.Label(root)
camera_label.pack(pady=10)

frame_plot = ttk.Frame(root)
frame_plot.pack(fill=tk.BOTH, expand=True)

fig, ax = plt.subplots(figsize=(6, 4))
canvas = FigureCanvasTkAgg(fig, master=frame_plot)
canvas.get_tk_widget().pack()

# Start
update_plot()
root.mainloop()
cap.release()
