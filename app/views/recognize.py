import config
import PySimpleGUI as sg
import cv2
import numpy as np
from Controller import Controller

"""
Demo program that displays a webcam using OpenCV
"""

def create(controller=None):
    if controller is None:
        controller = Controller()

    sg.theme('Black')

    # define the window layout
    layout = [
        [sg.Text('OpenCV Demo', size=(40, 1), justification='center', font='Helvetica 20')],
        [sg.Image(filename='', key='image')],
        [ sg.Button('Back', size=(10, 1), font='Helvetica 14'), ]
    ]

    # create the window and show it without the plot
    window = sg.Window(config.WINDOW_TITLE,
                       layout)

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    cap = cv2.VideoCapture(0)
    recording = True

    while True:
        event, values = window.read(timeout=20)
        if event == 'Back' or event is None:
            break

        if recording:
            ret, frame = cap.read()

            window_image = controller.handle_recognition(frame)

            imgbytes = cv2.imencode('.png', window_image)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)

    window.close()