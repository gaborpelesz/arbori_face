import detection.RetinaFace
import recognition.InsightFace

# GUI
WINDOW_TITLE = 'beta1.0 - Demo Face recognizer applicaton'
IMAGE_SIZE = 200

# Face recognition
DETECTOR = detection.RetinaFace.RetinaFace()
RECOGNIZER = recognition.InsightFace.InsightFace()

# Face recognition parameters
UNKNOWN_DISTANCE_THRESHOLD = 320
UNKNOWN_SIMILARITY_THRESHOLD = 170
USE_SIMILARITY_MEASUREMENT = True
K_NEIGHBORS = 1