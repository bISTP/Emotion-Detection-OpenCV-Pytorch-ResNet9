from dis import dis
from genericpath import exists
import numpy as np
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import PIL.Image, PIL.ImageTk
from shutil import copyfile
import time, os
import cv2
import torch
import torchvision.transforms as tt
import torch.nn as nn
import torch.nn.functional as F


face_classifier = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
model_state = torch.load("./models/emotion_detection_model_state.pth", map_location=torch.device('cpu'))
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ELU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 128)
        self.conv2 = conv_block(128, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.drop1 = nn.Dropout(0.5)
        
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 256, pool=True)
        self.res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        self.drop2 = nn.Dropout(0.5)
        
        self.conv5 = conv_block(256, 512)
        self.conv6 = conv_block(512, 512, pool=True)
        self.res3 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.drop3 = nn.Dropout(0.5)
        
        self.classifier = nn.Sequential(nn.MaxPool2d(6), 
                                        nn.Flatten(),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.drop1(out)
        
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.drop2(out)
        
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.res3(out) + out
        out = self.drop3(out)
        
        return self.classifier(out)

def clf_model():
    model = ResNet(1, len(class_labels))
    model.load_state_dict(model_state)
    return model



########################################################################
class videoGUI:

    def __init__(self, window, window_title):

        self.window = window
        self.window.geometry("1080x930")
        self.window.title(window_title)

        self.snap = False
        self.on_video = False
        self.real_time = False
        self.pause = False   # Parameter that controls pause button
        self.record_face_status = False   # Parameter that controls face detection and cropping face

        # After it is called once, the show_frame method will be automatically called every delay (milliseconds)
        self.delay = 40   # ms [1000 ms / 25 frames = 40 ms / frame]

        self.cascPath = "haarcascade_frontalface_default.xml"   # OpenCV face detector
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)

        self.result_video = 'output.mp4'   # filename to save output video file
        self.result_frame_size = (200, 200)  # Final frame size to save video file


        # Define the codec and create VideoWriter object, 'avi' works fine with DIVX codec
        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')

        ##### GUI Design #####
        top_frame = Frame(self.window)
        top_frame.pack(side=TOP, pady=5)

        bottom_frame = Frame(self.window)
        bottom_frame.pack(side=BOTTOM, pady=5)

        # Create a canvas that can fit the above video source size
        self.canvas = Canvas(top_frame)
        self.canvas.pack()
        self.canvas.config(width = 970, height = 640)
        self.canvas.create_rectangle(0, 0, 970, 640, outline="black", fill="grey")
        self.myimage = PIL.ImageTk.PhotoImage(file='emotions2.png')
        self.canvas.create_image(485, 320, image=self.myimage, anchor='c')

        # Select Button
        self.btn_select=Button(bottom_frame, text="Select video file", width=15, command=self.open_file, bg='skyBlue', fg='black', padx=15, pady = 15, borderwidth=4, relief="solid")
        self.btn_select.grid(row=0, column=2, padx=15, pady = 15)

        # Play Button
        self.btn_play=Button(bottom_frame, text="Play", width=15, state=DISABLED, command=self.play_video, bg='green', fg="black", padx=15, pady = 15, borderwidth=4, relief="solid")
        self.btn_play.grid(row=0, column=4, padx=15, pady = 15)

        # Pause Button
        self.btn_pause=Button(bottom_frame, text="Pause", width=15, state=DISABLED, command=self.pause_video, bg='red', fg="black", padx = 15, pady = 15, borderwidth=4, relief="solid")
        self.btn_pause.grid(row=0, column=5, padx=15, pady = 15)

        # Face Detection Label
        self.btn_real_time_detection = Button(bottom_frame, text="Real Time Detection", width=20, command=self.start_real_time, bg="lightGreen", fg="black", padx=15, pady = 15, borderwidth=4, relief="solid")
        self.btn_real_time_detection.grid(row=0, column=0, padx=15, pady = 15)

        # Snapshot Button
        self.btn_snapshot=Button(bottom_frame, text="Snapshot", width=15, state=DISABLED, command=self.snapshot, fg="black", padx=15, pady = 15, borderwidth=4, relief="solid")
        self.btn_snapshot.grid(row=0, column=6, padx=(15,0), pady = 15)

        # Status bar
        self.status = Label(bottom_frame, text='Snapshot feature is only for realtime detection and in one execution only do one type of detection!!', bd=1, relief=SUNKEN, anchor=W)  # anchor is for text inside Label
        self.status.grid(row=1, columnspan= 5, sticky=E+W, padx=15, pady = 15)  # side is for Label location in Window

        self.window.mainloop()


    def start_real_time(self):
        if (self.on_video):
            self.cap.release()
            cv2.destroyAllWindows()
            self.on_video = False
        self.real_time = True
        self.real_cap = cv2.VideoCapture(0)
        self.btn_play['state'] = NORMAL
        self.play_video()


    def open_file(self):
        if self.real_time:
            self.real_time = False
            self.real_cap.release()
            cv2.destroyAllWindows()

        self.on_video = True
        self.pause = False

        self.filename = filedialog.askopenfilename(title="Select file", filetypes=(("MP4 files", "*.mp4"),
                                                                                   ("AVI files", "*.avi")))
        print("\n Filename : ", self.filename, "\n")

        # Open the video file
        self.cap = cv2.VideoCapture(self.filename)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 970)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

        self.btn_play['state'] = NORMAL


    def get_frame(self):   # get only one frame

        try:

            if self.cap.isOpened():
                ret, frame = self.cap.read()

                if ret:
                    # Return a boolean success flag and the current frame (color map is BGR)
                    return (ret, frame)
                else:
                    return (ret, None)

            else:
                raise ValueError("Unable to open video file : ", self.filename)

        except:
            messagebox.showerror(title='Video file not found', message='Please select a video file.')


    def play_video(self):
        model = clf_model()
        self.btn_pause['state'] = NORMAL
        self.btn_snapshot['state'] = NORMAL

        # Get a frame from the video source, and go to the next frame automatically
        if self.real_time:
            ret, frame = self.real_cap.read()
        else:
            ret, frame = self.get_frame()
        

        if ret:
            
            #####################################################

            frame = cv2.flip(frame, 1)
            labels = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y : y + h, x : x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = tt.functional.to_pil_image(roi_gray)
                    roi = tt.functional.to_grayscale(roi)
                    roi = tt.ToTensor()(roi).unsqueeze(0)

                    # make a prediction on the ROI
                    tensor = model(roi)
                    pred = torch.max(tensor, dim=1)[1].tolist()
                    label = class_labels[pred[0]]

                    label_position = (x, y)
                    cv2.putText(
                    frame,
                    label,
                    label_position,
                    cv2.FONT_HERSHEY_COMPLEX,
                    2,
                    (0, 255, 0),
                    3,
                    )
                else:
                    cv2.putText(
                    frame,
                    "No Face Found",
                    (20, 60),
                    cv2.FONT_HERSHEY_COMPLEX,
                    2,
                    (0, 255, 0),
                    3,
                    )

            #####################################################
            frame = cv2.resize(frame, (970, 640))
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))   # convert BGR into RGB color map
            

            self.canvas.create_image(0, 0, image = self.photo, anchor = NW)

        after_id = self.window.after(self.delay, self.play_video)

        if self.snap:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".png", frame)
            self.snap = False
        
        if self.pause:
            self.window.after_cancel(after_id)
            self.pause = False


    def pause_video(self):
        self.pause = True
        # self.status['text'] = 'Video paused'
        # print('I am in pause function : ', self.pause)


    def snapshot(self):
        self.snap = True



    # Release the video source when the object is destroyed
    def __del__(self):
        if self.on_video:
            if self.cap.isOpened():
                self.cap.release()

##### End Class #####


# Create a window and pass it to GUI Class
videoGUI(Tk(), "Emotion Detector")