import insightface

class Detect:
    def __init__(self):
        self.model = insightface.model_zoo.get_model('retinaface_r50_v1')
        self.model.prepare(ctx_id = -1, nms=0.4)

    def detect_faces(self, img):
        bbox, landmark = self.model.detect(img, threshold=0.5, scale=1.0)
        return bbox, landmark
