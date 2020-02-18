import config
import PySimpleGUI as sg
from Controller import Controller

def create(controller=None):
    if controller is None:
        controller = Controller()
    all_names = controller.get_names()

    sg.theme('Black')
    # define the window layout
    layout = [
        [sg.Text(name, font='Helvetica 20'),
            sg.Button('Delete person', font='Helvetica 14', key=name)]
        for name in all_names]

    layout.append([sg.Quit('Back')])

    # create the window and show it without the plot
    window = sg.Window(config.WINDOW_TITLE, layout, size=(400, 300))

    while True:
        event, values = window.read()
        if event in (None, 'Back'):
            break
        if event in all_names:
            person_name = event
            print('deleting person')
            pop_up_layout = [
                [sg.Text(f'Do you want to delete "{person_name}"')],
                [sg.Submit(), sg.Cancel()]
            ]
            sg.theme('Black')
            pop_up_menu = sg.Window(config.WINDOW_TITLE, pop_up_layout, size=(200, 100))
            while True:
                event, values = pop_up_menu.read()
                if event is 'Submit':
                    controller.delete_person(person_name)
                    pop_up_menu.close()
                    window.close()
                    return
                if event is 'Cancel':
                    pop_up_menu.close()
                    break


    window.close()