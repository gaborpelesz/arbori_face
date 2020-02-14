import insightface
import detection.FaceDetector

class RetinaFace(detection.FaceDetector.FaceDetector):
    def __init__(self):
        super().__init__()

    def init(self):
        self.model = insightface.model_zoo.get_model('retinaface_r50_v1')
        self.model.prepare(ctx_id = -1, nms=0.4)

    def detect(self, image_frame):
        bbox, landmark = self.model.detect(image_frame, threshold=0.5, scale=1.0)
        return bbox, landmark