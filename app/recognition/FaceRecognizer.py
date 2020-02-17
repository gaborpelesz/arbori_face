from abc import ABC, abstractmethod
import config

class FaceRecognizer(ABC):
    def __init__(self):
        super().__init__()
        self.faces = []
        self.landmarks = []

    @abstractmethod
    def recognize(self, image_frame):
        pass