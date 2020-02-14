from abc import ABC, abstractmethod
import config

class FaceRecognizer(ABC):
    def __init__(self):
        super().__init__()
        self.faces = []
        self.landmarks = []
        self.detector = config.DETECTOR
        self.detector.init()

    @abstractmethod
    def recognize(self, image_frame):
        self.faces, self.landmarks = self.detector.detect(image_frame)