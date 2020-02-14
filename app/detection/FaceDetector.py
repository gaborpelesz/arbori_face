from abc import ABC, abstractmethod

class FaceDetector(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def detect(self, image_frame):
        pass