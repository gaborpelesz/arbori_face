import config
import cv2
import numpy as np
from FaceDatabaseHandler import FaceDatabaseHandler

class Controller:
    def __init__(self):
        self.database = FaceDatabaseHandler()

    def handle_detection_live_feed(self, frame):
        frame = cv2.resize(frame, (int(config.IMAGE_SIZE/frame.shape[0]*frame.shape[1]), int(config.IMAGE_SIZE)))
        frame_to_detect = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces, landmarks = config.DETECTOR.detect(frame_to_detect)
        is_shoot_available = (len(faces) == 1)

        color_red = (0,0,255)
        color_green = (0,255,0)
        COLOR = color_green if is_shoot_available else color_red
        THICKNESS = 2

        cut_out_face = None

        added_margin_pixels = int(frame.shape[1] * 0.01)

        if faces is not None:
            for i, detected_face in enumerate(faces):

                # # add margin and normalize outliers
                # for j in range(len(detected_face)):
                #     detected_face[j] += -added_margin_pixels if j < 2 else added_margin_pixels
                #     detected_face[j] = 0 if detected_face[j] < 0 else frame.shape[abs(j%2-1)] if frame.shape[abs(j%2-1)] < detected_face[j] else detected_face[j]
                
                top_left = detected_face[0], detected_face[1] # x, y
                bottom_right = detected_face[2], detected_face[3] # x, y
                
                right_eye = landmarks[i][0]
                left_eye = landmarks[i][1]

                if is_shoot_available:
                    self.ready_to_save_face = config.DETECTOR.align_face(frame, right_eye, left_eye)

                cv2.rectangle(frame, top_left, bottom_right, COLOR, THICKNESS)
                cv2.circle(frame, tuple(right_eye), 2, (0,0,255), thickness=2)
                cv2.circle(frame, tuple(left_eye), 2, (0,0,255), thickness=2)

        return frame, len(faces)

    def handle_add_person(self, name):
        if self.ready_to_save_face is None:
            return
        result_embedding = config.RECOGNIZER.get_embedding(self.ready_to_save_face)[0]
        self.database.add_embedding(name, result_embedding)
        print('Adding face was successful!')

        
