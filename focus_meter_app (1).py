import cv2
import dlib
import pyttsx3
from scipy.spatial import distance
import time
import threading
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import customtkinter as ctk  # Custom tkinter for better UI

# INITIALIZING THE pyttsx3 SO THAT ALERT AUDIO MESSAGE CAN BE DELIVERED
engine = pyttsx3.init()

# FACE DETECTION OR MAPPING THE FACE TO GET THE EYE AND EYES DETECTED
face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate the aspect ratio for the eye
def Detect_Eye(eye):
    poi_A = distance.euclidean(eye[1], eye[5])
    poi_B = distance.euclidean(eye[2], eye[4])
    poi_C = distance.euclidean(eye[0], eye[3])
    aspect_ratio_Eye = (poi_A + poi_B) / (2 * poi_C)
    return aspect_ratio_Eye

# Function to calculate concentration score
def conc_calc(start_time, end_time, focus_loss):
    duration = round((end_time - start_time) / 10)
    duration_points = duration
    focus_loss = focus_loss * 10
    current_points = duration_points - focus_loss
    return duration, focus_loss, current_points

class DrowsinessDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drowsiness Detector")
        # Set window size to half of the screen size
        # Set window size to half of the screen size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = screen_width // 2
        window_height = screen_height // 2
        self.root.geometry(f"{window_width}x{window_height}")
        bg_image = ctk.CTkImage(Image.open("focus-pic.png"), size=(500,300))

        # Create a label to hold the background image
        bg_label = ctk.CTkLabel(root, image=bg_image)
        bg_label.pack(expand=True)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Configure the customtkinter theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.frame = ctk.CTkFrame(master=root)
        self.frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.start_btn = ctk.CTkButton(master=self.frame, text="Start Detection", command=self.start_detection)
        self.start_btn.pack(pady=10)

        self.stop_btn = ctk.CTkButton(master=self.frame, text="Stop Detection", command=self.stop_detection)
        self.stop_btn.pack(pady=10)
        self.stop_btn.configure(state=tk.DISABLED)

        self.video_label = Label(self.frame)
        self.video_label.pack()

        self.info_label = ctk.CTkLabel(master=self.frame, text="Press 'Start Detection' to begin", font=("Helvetica", 14))
        self.info_label.pack(pady=20)

        self.cap = None
        self.running = False
        self.detection_thread = None
        self.session_start_time = None
        self.focus_loss_counter = 0

    def start_detection(self):
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.running = True
        self.session_start_time = time.time()
        self.focus_loss_counter = 0

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.info_label.configure(text="Error: Could not open video stream.")
            self.stop_detection()
            return

        self.detection_thread = threading.Thread(target=self.detect_drowsiness)
        self.detection_thread.start()

    def stop_detection(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image='')

        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)

        if self.session_start_time:
            session_end_time = time.time()
            duration, focus_loss, focus_score = conc_calc(self.session_start_time, session_end_time, self.focus_loss_counter)
            self.info_label.configure(text=f"Duration: {duration}s, Distractions: {focus_loss}, Focus Score: {focus_score}")

    def detect_drowsiness(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray_scale)

            for face in faces:
                face_landmarks = dlib_facelandmark(gray_scale, face)
                leftEye = [] 
                rightEye = [] 

                for n in range(42, 48):
                    x = face_landmarks.part(n).x
                    y = face_landmarks.part(n).y
                    rightEye.append((x, y))
                    next_point = n + 1
                    if n == 47:
                        next_point = 42
                    x2 = face_landmarks.part(next_point).x
                    y2 = face_landmarks.part(next_point).y
                    cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

                for n in range(36, 42):
                    x = face_landmarks.part(n).x
                    y = face_landmarks.part(n).y
                    leftEye.append((x, y))
                    next_point = n + 1
                    if n == 41:
                        next_point = 36
                    x2 = face_landmarks.part(next_point).x
                    y2 = face_landmarks.part(next_point).y
                    cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

                right_Eye = Detect_Eye(rightEye)
                left_Eye = Detect_Eye(leftEye)
                Eye_Rat = (left_Eye + right_Eye) / 2
                Eye_Rat = round(Eye_Rat, 2)

                if Eye_Rat < 0.15:
                    cv2.putText(frame, "DROWSINESS DETECTED", (50, 100),
                                cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 210), 3)
                    cv2.putText(frame, "Alert!!!! WAKE UP DUDE", (50, 450),
                                cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)
                    self.focus_loss_counter += 1

                    engine.say("Alert!!!! WAKE UP DUDE")
                    engine.runAndWait()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        if self.cap:
            self.cap.release()
        self.video_label.config(image='')

if __name__ == "__main__":
    root = ctk.CTk()
    app = DrowsinessDetectorApp(root)
    root.mainloop()
