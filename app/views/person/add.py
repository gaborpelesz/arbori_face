import config
import PySimpleGUI as sg
import cv2
import numpy as np
from Controller import Controller

def create():

    sg.theme('Black')

    # define the window layout
    layout = [
        [sg.Text('OpenCV Demo', size=(40, 1), justification='center', font='Helvetica 20')],
        [sg.Image(filename='', key='image')],
        [sg.Button('Shoot photo', size=(10, 1), font='Helvetica 14')],
        [sg.Text('Enter name:'), sg.InputText(size=(10,1), key='-name-'), sg.Text(size=(40,1), key='-error-text-', text_color='red')],
        [sg.Button('Exit', size=(10, 1), font='Helvetica 14')]
    ]

    # create the window and show it without the plot
    window = sg.Window(config.WINDOW_TITLE,
                       layout, location=(0, 0))

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    cap = cv2.VideoCapture(0)
    controller = Controller()
    recording = True
    is_error_counter_on = False
    error_counter = 0

    while True:
        event, values = window.read(timeout=20)
        if event == 'Exit' or event is None:
            break

        if event == 'Shoot photo':
            if ret is None:
                window['-error-text-'].update('No face was detected')
                is_error_counter_on = True
            if values['-name-'] == '':
                window['-error-text-'].update('No name was specified')
                is_error_counter_on = True
            else:
                controller.handle_add_person(values['-name-'])

        if recording:
            ret, frame = cap.read()
            window_image = controller.handle_detection_live_feed(frame)
            imgbytes = cv2.imencode('.png', window_image)[1].tobytes()
            window['image'].update(data=imgbytes)

        # Handling User errors
        if error_counter == 10:
            error_counter = 0
            is_error_counter_on = False
            window['-error-text-'].update('')
        if is_error_counter_on:
            error_counter += 1

    window.close()