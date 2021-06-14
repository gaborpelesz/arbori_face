import config
import cv2
import numpy as np
from FaceDatabaseHandler import FaceDatabaseHandler
from collections import Counter
import operator
import time

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
                if is_shoot_available:
                    self.ready_to_save_face = config.DETECTOR.preprocess_face(frame, detected_face, landmarks[i])
                else:
                    top_left = detected_face[0], detected_face[1] # x, y
                    bottom_right = detected_face[2], detected_face[3] # x, y
                    cv2.rectangle(frame, top_left, bottom_right, COLOR, THICKNESS)

        window_image = self.ready_to_save_face if is_shoot_available else frame
        return window_image, len(faces)

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
        detected_face_bboxes, landmarks = config.DETECTOR.detect(frame_to_detect)

        
        processes = []
        for detected_face in zip(detected_face_bboxes, landmarks):
            face_bbox = detected_face[0]
            landmarks = detected_face[1]
            aligned_detected_face = config.DETECTOR.preprocess_face(frame, face_bbox, landmarks)
            self.process_recognition(frame, face_bbox, aligned_detected_face, db_faces)

        return frame

    def process_recognition(self, frame, face_bbox, aligned_detected_face, db_faces):
        t_start = time.time()
        name = self.recognize_person(aligned_detected_face, db_faces)
        print(f'recognition runtime: {(time.time() - t_start) * 1000:0.2f}ms')

        COLOR = (0,0,255) if name.startswith('Unknown') else (0,255,0)

        top_left = face_bbox[0], face_bbox[1] # x, y
        bottom_right = face_bbox[2], face_bbox[3] # x, y
        cv2.rectangle(frame, top_left, bottom_right, COLOR, 2)
        cv2.putText(frame, name, (top_left[0], top_left[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 1, cv2.LINE_AA)

    def recognize_person(self, face_to_recognize, db_faces):
        if len(db_faces) < 5:
            return 'Unknown'

        distances = self._calculate_distances(face_to_recognize, db_faces)
        #print(distances, '\n')

        similarities = list(zip(*distances))[0]
        similarities = np.array(similarities, dtype=np.float64)
        confidence_percentage = (np.exp(similarities) / np.sum(np.exp(similarities)))[0]

        if config.USE_SIMILARITY_MEASUREMENT:
            if similarities[0] < config.UNKNOWN_SIMILARITY_THRESHOLD:
                return f'Unknown-{confidence_percentage*100:.1f}%'
        elif distances[0][1] > config.UNKNOWN_DISTANCE_THRESHOLD:
            return 'Unknown'

        predicted_name = distances[0][2]
        if config.K_NEIGHBORS > 1:
            # K-NN
            k_neighbors = config.K_NEIGHBORS
            k_nearest_names = dict(Counter(list(zip(*distances[:k_neighbors]))[2]))
            predicted_name = max(k_nearest_names.items(), key=operator.itemgetter(1))[0]

        return f'{predicted_name}-{confidence_percentage*100:.1f}%'

    def _calculate_distances(self, face_to_recognize, db_faces):
        is_reverse = False
        choosen_measurement = 1
        
        if config.USE_SIMILARITY_MEASUREMENT:
            is_reverse = True
            choosen_measurement = 0 # measurements array will contain sim in the first elements

        embedding_to_recognize = config.RECOGNIZER.get_embedding(face_to_recognize)[0]

        measurements = []

        for db_face in db_faces:
            name, embedding = db_face
            dist, sim = config.RECOGNIZER.compare_embeddings(embedding_to_recognize, embedding)
            measurements.append((sim, dist, name))

        return sorted(measurements, key=lambda x: x[choosen_measurement], reverse=is_reverse)




        
