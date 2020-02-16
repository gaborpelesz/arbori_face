import config
import PySimpleGUI as sg
import cv2
import numpy as np

"""
Demo program that displays a webcam using OpenCV
"""

def create():

    sg.theme('Black')

    # define the window layout
    layout = [
        [sg.Text('OpenCV Demo', size=(40, 1), justification='center', font='Helvetica 20')],
        [sg.Image(filename='', key='image')],
        [ sg.Button('Shoot photo', size=(10, 1), font='Helvetica 14'), ],
        [ sg.Button('Exit', size=(10, 1), font='Helvetica 14'), ]
    ]

    # create the window and show it without the plot
    window = sg.Window(config.WINDOW_TITLE,
                       layout, location=(0, 0))

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    cap = cv2.VideoCapture(0)
    recording = True

    while True:
        event, values = window.read(timeout=20)
        if event == 'Exit' or event is None:
            break

        if recording:
            ret, frame = cap.read()

            frame = cv2.resize(frame, (int(config.IMAGE_SIZE/frame.shape[0]*frame.shape[1]), int(config.IMAGE_SIZE)))

            frame_to_detect = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = config.RECOGNIZER.recognize(frame_to_detect)
            
            if faces is not None:
                COLOR = (0,255,0) if len(faces) == 1 else (0,0,255) # color red in BGR
                THICKNESS = 2

                for detected_face in faces:
                    detected_face = detected_face.astype(np.int)

                    left_top = detected_face[0], detected_face[1]
                    bottom_right = detected_face[2], detected_face[3]

                    cv2.rectangle(frame, left_top, bottom_right, COLOR, THICKNESS)

            imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)

    window.close()