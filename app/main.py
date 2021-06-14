import PySimpleGUI as sg  
import views
import config

def main():
    sg.theme('Black')

    layout = [
        [sg.Text('Face recognition demo app', justification='center')],          
        [sg.Button('Recognize')],
        [sg.Button('Add person')],
        [sg.Button('List people')],
        [sg.Quit('Exit')]
    ]      

    window = sg.Window(config.WINDOW_TITLE, layout, size=(400, 300))    

    while True:
        event, values = window.read()
        
        if event in (None, 'Exit'):
            break
        if event is 'Recognize':
            window.Hide()
            views.recognize.create()
            window.UnHide()
        if event is 'Add person':
            window.Hide()
            views.person.add.create()
            window.UnHide()
        if event is 'List people':
            window.Hide()
            views.person.list.create()
            window.UnHide()

    window.close()

if __name__ == '__main__':
    print('WARNING:')
    print(' - Deprecated running of the program...')
    print(' - Please run by executing "python app" from the root of the repository instead.')