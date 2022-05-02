import tkinter as tk
from tkinter import messagebox
import cv2
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import mediapipe as mp
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from keras.models import load_model
import pyttsx3 as p3

def main():
    wind=tk.Tk()
    wind.geometry("800x400")
    wind.title("Sign Language Detect System")

    canvas = tk.Canvas(wind, height=400, width=800)
    login_background = Image.open("bc1.jpg").resize((800, 400))
    login_background = ImageTk.PhotoImage(login_background)
    login_image = canvas.create_image(0, 0, anchor='nw', image=login_background)
    canvas.pack(side='top')

    title_lab=tk.Label(wind,text="Welcome To Our Sign Language Detect System",bg="yellow",font="仿宋 17 bold")
    title_lab.place(x=140,y=30)

    userlab=tk.Label(wind, text="username", font="仿宋 20 bold", fg="blue", width=8)
    userlab.place(x=200,y=100)
    user_entry=tk.Entry(wind, width=15,bg="white",font="仿宋 20 bold")
    user_entry.place(x=350,y=100)

    sslab=tk.Label(wind, text="password", font="仿宋 20 bold", fg="blue", width=8)
    sslab.place(x=200,y=200)
    ss_entry=tk.Entry(wind, width=15,bg="white",font="仿宋 20 bold",show="*")
    ss_entry.place(x=350,y=200)

    def login():
        username = user_entry.get()
        password = ss_entry.get()
        if username == 'admin' and password == 'admin':
            print('login success.')
            messagebox.showinfo("info", "login success.")
            wind.destroy()
            gui = Gui()
        else:
            print('username or password error, please check it!')
            messagebox.showinfo("info", "username or password error, please check it!")

    login_btn=tk.Button(wind,text="Login", font="仿宋 20 bold", fg="blue", width=8, command=login)
    login_btn.place(x=350,y=300)
    wind.mainloop()


class Gui(object):
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.camera = 0
        self.actions = np.array(['hello', 'how', 'you'])
        #self.actions = np.array(['hello', 'iloveyou','thanks'])
        self.map = {label:num for num, label in enumerate(self.actions)}

        self.model = self.load_model()
        self.root = Tk()
        self.root.title("Sign Language Detect System")
        #self.canvas = Canvas(self.root, bg='#c4c2c2', width=600, height=400)
        #self.canvas.pack(padx=10, pady=10)
        self.root.config(cursor="arrow")
        self.btn = Button(self.root, width=350, text="open the camera", command=self.openCamera1)
        self.btn.pack(fill="both", expand=True, padx=10, pady=10)
        #self.btn1 = Button(self.root, text="close the camera", command=self.exist)
        #self.btn1.pack(fill="both", expand=True, padx=11, pady=10)
        self.root.mainloop()

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False  # Image is no longer writeable
        results = model.process(image)  # Make prediction
        image.flags.writeable = True  # Image is now writeable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
        return image, results

    def draw_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION)  # Draw face connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                                  self.mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                                  self.mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections

    def draw_styled_landmarks(self, image, results):
        # Draw face connections
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                                  self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )
        # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                  self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )
        # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                  self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )
        # Draw right hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                  self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

    def extract_keypoints(self, results):
        self.pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        self.face = np.array([[res.x, res.y, res.z] for res in
                         results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        self.lh = np.array([[res.x, res.y, res.z] for res in
                       results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
            21 * 3)
        self.rh = np.array([[res.x, res.y, res.z] for res in
                       results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
            21 * 3)
        return np.concatenate([self.pose, self.face, self.lh, self.rh])

    def prob_viz(self, res, actions, input_frame, colors):
        self.output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(self.output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
            cv2.putText(self.output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

        return self.output_frame

    def load_model(self):
        model = load_model("hellohowyou.h5")
        return model

    def openCamera(self):
        self.camera = cv2.VideoCapture(0)
        while True:
            success, img = self.camera.read()
            img = cv2.flip(img, 1)
            (h, w) = img.shape[:2]
            print(h)
            print(w)
            if success:
                cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
                current_image = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=current_image)
                self.canvas.create_image(0, 0, anchor='nw', image=imgtk)
            self.root.update_idletasks()

            self.root.update()

    def getMapper(self, label):
        return self.map.get(label)

    def openCamera1(self):
        self.sequence = []
        self.sentence = []
        self.predictions = []
        self.threshold = 0.5
        self.colors = [(245,117,16), (117,245,16), (16,117,245)]

        self.cap = cv2.VideoCapture(0)
        # Set mediapipe model
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.cap.isOpened():
                #engine = p3.init()
                # Read feed
                ret, frame = self.cap.read()

                # Make detections
                image, results = self.mediapipe_detection(frame, holistic)

                # Draw landmarks
                self.draw_styled_landmarks(image, results)

                # 2. Prediction logic
                keypoints = self.extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:]

                if len(self.sequence) == 30:
                    res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
                    previous = self.actions[np.argmax(res)]
                    if previous != self.actions[np.argmax(res)]:
                        print(self.actions[np.argmax(res)])
                    #engine.say(self.actions[np.argmax(res)])
                    #engine.runAndWait()
                    self.predictions.append(np.argmax(res))

                    # 3. Viz logic
                    if np.unique(self.predictions[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > self.threshold:

                            if len(self.sentence) > 0:
                                if self.actions[np.argmax(res)] != self.sentence[-1]:
                                    self.sentence.append(self.actions[np.argmax(res)])
                            else:
                                self.sentence.append(self.actions[np.argmax(res)])

                    if len(self.sentence) > 5:
                        self.sentence = self.sentence[-5:]

                    # Viz probabilities
                    #image = self.prob_viz(res, self.actions, image, self.colors)

                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(self.sentence), (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Show to screen
                cv2.imshow('OpenCV Feed', image)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            #cap.release()
            #cv2.destroyAllWindows()

        # self.camera = cv2.VideoCapture(0)
        # with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        #     while self.camera.isOpened():
        #         ret, frame = self.camera.read()
        #         if frame is None:
        #             break
        #         (h, w) = frame.shape[:2]
        #         width = 1200
        #         r = width / float(w)
        #         dim = (600, 400)
        #         frame1 = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        #         gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        #         # Make detections
        #         image, results = self.mediapipe_detection(frame1, holistic)
        #         # print(results)
        #
        #         # Draw landmarks
        #         self.draw_styled_landmarks(image, results)
        #
        #         dim2 = (200, 200)
        #         frame2 = cv2.resize(frame, dim2, interpolation=cv2.INTER_AREA)
        #         frame2 = np.array(frame2).reshape(1, 200, 200, 3)
        #         # gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        #         # print(gray2.shape)
        #         prediction = self.model.predict(frame2)
        #         results = [np.argmax(prediction[i]) for i in range(len(prediction))]
        #         predict_label = self.getMapper(results[0])
        #
        #         cv2.putText(image, "Predict action: {}".format(predict_label), (10, 30),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #         key = cv2.waitKey(10) & 0xFF
        #
        #         if key == 27:
        #             break
        #         # cv2image = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGBA)
        #         current_image = Image.fromarray(image)
        #         imgtk = ImageTk.PhotoImage(image=current_image)
        #         self.canvas.create_image(0, 0, anchor='nw', image=imgtk)
        #         self.root.update_idletasks()
        #         self.root.update()
        #
        # self.root.mainloop()
        # # self.camera.release()
        # # cv2.destroyAllWindows()

    def exist(self):
        self.camera.release()
        cv2.destroyAllWindows()
        self.root.quit()

    def __del__(self):
        self.camera.release()
        cv2.destroyAllWindows()
        self.root.quit()

if __name__ == '__main__':
    main()
