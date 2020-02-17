import config
import insightface
import recognition.FaceRecognizer
import numpy as np

class InsightFace(recognition.FaceRecognizer.FaceRecognizer):
    def __init__(self):
        super().__init__()
        self.model = insightface.model_zoo.get_model('arcface_r100_v1')
        self.model.prepare(ctx_id = -1)

    def recognize(self, image_frame):
        pass

    def get_embedding(self, face):
        emb = self.model.get_embedding(face)
        return emb

    def compare_embeddings(self, embedding1, embedding2):
        dist = np.sum(np.square(embedding1-embedding2))
        sim = np.dot(embedding1, embedding2.T)
        #print('dist:', dist, '\tsim:', sim)
        return dist, sim