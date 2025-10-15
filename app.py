import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
from model import build_emotion_model
import time

# Incarcarea modelului antrenat
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
model = build_emotion_model(input_shape=(48,48,1), num_classes=7)
model.load_weights("model.weights.h5")

# Incarcarea clasificatorului de fete
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Parametrii pentru gestionarea predicțiilor
last_prediction = None
last_pred_time = 0
prediction_interval = 3  # seconds

# GUI setup
root = tk.Tk()
root.title("Emotion Recognition")
root.geometry("900x600")
root.configure(bg='white')

# Cadre
top_frame = tk.Frame(root, bg='white')
top_frame.pack(pady=10)

middle_frame = tk.Frame(root, bg='white')
middle_frame.pack(pady=10)

bottom_frame = tk.Frame(root, bg='white')
bottom_frame.pack(pady=10)

# Elemente
webcam_label = tk.Label(middle_frame, bg='white')
webcam_label.grid(row=0, column=0, padx=10)

image_label = tk.Label(middle_frame, bg='white')
image_label.grid(row=0, column=1, padx=10)

result_label = tk.Label(bottom_frame, text="", font=("Arial", 16), bg='white')
result_label.pack()

# Predictia emoției dintr-o imagine
def predict_emotion(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)
        preds = model.predict(face)
        emotion = emotion_labels[np.argmax(preds)]
        return emotion
    return "No face found"

# Incarca imaginea
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        emotion = predict_emotion(file_path)
        result_label.config(text=f"Predicted Emotion: {emotion}")

        img = Image.open(file_path).resize((400, 400))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

# Logica webcam
cap = None
running = False

def update_webcam():
    global cap, running, last_prediction, last_pred_time

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        current_time = time.time()

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (48, 48))
            face_resized = face_resized.astype("float32") / 255.0
            face_resized = np.expand_dims(face_resized, axis=-1)
            face_resized = np.expand_dims(face_resized, axis=0)

            # Preziceză doar la intervale regulate
            if last_prediction is None or (current_time - last_pred_time) > prediction_interval:
                preds = model.predict(face_resized)
                new_pred = np.argmax(preds)
                if new_pred != last_prediction:
                    last_prediction = new_pred
                last_pred_time = current_time

            # Deseneaza chenarul si eticheta
            if last_prediction is not None:
                emotion_text = emotion_labels[last_prediction]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Afiseaza frame-ul
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame).resize((400, 400))
        img_tk = ImageTk.PhotoImage(image=img_pil)
        webcam_label.config(image=img_tk)
        webcam_label.image = img_tk

def open_camera():
    global cap, running
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
        threading.Thread(target=update_webcam).start()

def close_camera():
    global running, cap
    running = False
    if cap:
        cap.release()
        cap = None
    webcam_label.config(image='')  # Inchide camera
    webcam_label.image = None

def clear_uploaded_image():
    image_label.config(image='')
    image_label.image = None
    result_label.config(text='')


# Butoane
camera_btn = tk.Button(top_frame, text="Open Camera", command=open_camera, font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
close_cam_btn = tk.Button(top_frame, text="Close Camera", command=close_camera, font=("Arial", 12), bg="#f44336", fg="white", padx=10, pady=5)
upload_btn = tk.Button(top_frame, text="Upload Image", command=upload_image, font=("Arial", 12), bg="#2196F3", fg="white", padx=10, pady=5)
clear_img_btn = tk.Button(top_frame, text="Clear Image", command=clear_uploaded_image, font=("Arial", 12), bg="#9E9E9E", fg="white", padx=10, pady=5)

camera_btn.grid(row=0, column=0, padx=10)
close_cam_btn.grid(row=0, column=1, padx=10)
upload_btn.grid(row=0, column=2, padx=10)
clear_img_btn.grid(row=0, column=3, padx=10)

def on_closing():
    global running
    running = False
    if cap:
        cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
