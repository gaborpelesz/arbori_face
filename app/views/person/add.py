import config
import PySimpleGUI as sg
import cv2
import numpy as np
from Controller import Controller

def create(controller=None):
    if controller is None:
        controller = Controller()

    sg.theme('Black')

    # define the window layout
    layout = [
        [sg.Text('Demo face recognition', size=(40, 1), justification='center', font='Helvetica 20')],
        [sg.Image(filename='', key='image')],
        [sg.Button('Shoot photo', size=(10, 1), font='Helvetica 14')],
        [sg.Text('Enter name:'), sg.InputText(size=(10,1), key='-name-'), sg.Text(size=(40,1), key='-error-text-', text_color='red')],
        [sg.Button('Back', size=(10, 1), font='Helvetica 14')]
    ]

    # create the window and show it without the plot
    window = sg.Window(config.WINDOW_TITLE,
                       layout)

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    cap = cv2.VideoCapture(0)
    recording = True
    is_error_counter_on = False
    error_counter = 0
    detected_face_count = 0

    while True:
        event, values = window.read(timeout=20)
        if event in ('Back', None):
            break

        if event == 'Shoot photo':

            if values['-name-'] == '':
                window['-error-text-'].update('No name was specified')
                error_counter = 0
                is_error_counter_on = True
            elif detected_face_count > 1:
                window['-error-text-'].update('Too many faces were detected')
                error_counter = 0
                is_error_counter_on = True
            elif detected_face_count == 0:
                window['-error-text-'].update('No face was detected')
                error_counter = 0
                is_error_counter_on = True
            else:
                controller.handle_add_person(values['-name-'])

        if recording:
            ret, frame = cap.read()
            window_image, detected_face_count = controller.handle_detection_live_feed(frame)
            imgbytes = cv2.imencode('.png', window_image)[1].tobytes()
            window['image'].update(data=imgbytes)

        # Handling User errors
        if error_counter == 10:
            is_error_counter_on = False
            window['-error-text-'].update('')
        if is_error_counter_on:
            error_counter += 1

    window.close()