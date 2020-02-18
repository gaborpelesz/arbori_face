import detection.RetinaFace
import recognition.InsightFace

# GUI
WINDOW_TITLE = 'Demo - Face recognizer applicaton'
IMAGE_SIZE = 200

# Face recognition
DETECTOR = detection.RetinaFace.RetinaFace()
RECOGNIZER = recognition.InsightFace.InsightFace()

# Face recognition parameters
UNKNOWN_DISTANCE_THRESHOLD = 320
UNKNOWN_SIMILARITY_THRESHOLD = 100
SIMILARITY_MEASURE = True
K_NEIGHBORS = 1