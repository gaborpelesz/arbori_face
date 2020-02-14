import config
import recognition.FaceRecognizer

class InsightFace(recognition.FaceRecognizer.FaceRecognizer):
    def __init__(self):
        super().__init__()

    def recognize(self, image_frame):
        super().recognize(image_frame)
        return self.faces