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

            # TODO RECOGNIZER

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (int(config.IMAGE_SIZE/frame.shape[0]*frame.shape[1]), int(config.IMAGE_SIZE)))

            imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)

    window.close()