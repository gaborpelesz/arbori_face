import config
import cv2
import numpy as np
from FaceDatabaseHandler import FaceDatabaseHandler
from collections import Counter
import operator

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

                # optional circles at eyes
                # cv2.circle(frame, tuple(right_eye), 2, (0,0,255), thickness=2)
                # cv2.circle(frame, tuple(left_eye), 2, (0,0,255), thickness=2)

        return frame, len(faces)

    def handle_add_person(self, name):
        if self.ready_to_save_face is None:
            return
        result_embedding = config.RECOGNIZER.get_embedding(self.ready_to_save_face)[0]
        self.database.add_embedding(name, result_embedding)
        print('Adding face was successful!')

    def get_names(self):
        return self.database.get_people_names()

    def delete_person(self,name):
        self.database.delete_person(name)
        print('Deleting person was successful!')

    def handle_recognition(self, frame):
        db_faces = self.database.get_all_faces()

        frame = cv2.resize(frame, (int(config.IMAGE_SIZE/frame.shape[0]*frame.shape[1]), int(config.IMAGE_SIZE)))
        frame_to_detect = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_faces, landmarks = config.DETECTOR.detect(frame_to_detect)


        THICKNESS = 2
        FONT_SCALE = 0.5

        for i in range(len(detected_faces)):
            top_left = detected_faces[i][0], detected_faces[i][1] # x, y
            bottom_right = detected_faces[i][2], detected_faces[i][3] # x, y

            right_eye = landmarks[i][0]
            left_eye = landmarks[i][1]

            aligned_detected_face = config.DETECTOR.align_face(frame, right_eye, left_eye)
            name = self.recognize_person(aligned_detected_face, db_faces)

            COLOR = (0,0,255) if name == 'Unknown' else (0,255,0)

            cv2.rectangle(frame, top_left, bottom_right, COLOR, THICKNESS)
            cv2.putText(frame, name, (top_left[0], top_left[1]-5), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR, 1, cv2.LINE_AA)

        return frame


    def recognize_person(self, face_to_recognize, db_faces):
        if len(db_faces) < 5:
            return 'Unknown'

        embedding_to_recognize = config.RECOGNIZER.get_embedding(face_to_recognize)[0]

        distances = []
        for db_face in db_faces:
            name, embedding = db_face
            dist, sim = config.RECOGNIZER.compare_embeddings(embedding_to_recognize, embedding)
            distances.append((dist, name))

        distances = sorted(distances, key=lambda x: x[0])
        #print(distances, '\n')
        # return distances[0][1]

        db_faces = sorted(db_faces, key=lambda face: self._face_sort_comparison(face,embedding_to_recognize))

        if distances[0][0] > config.UNKNOWN_DISTANCE_THRESHOLD:
            return 'Unknown'

        # K-NN
        k_neighbors = 3
        k_nearest_names = dict(Counter(list(zip(*db_faces[:k_neighbors]))[0]))
        predicted_name = max(k_nearest_names.items(), key=operator.itemgetter(1))[0]
        return predicted_name

    def _face_sort_comparison(self, face, embedding_to_recognize):
        name, embedding = face
        dist, sim = config.RECOGNIZER.compare_embeddings(embedding_to_recognize, embedding)
        return dist




        
