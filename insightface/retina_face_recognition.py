import insightface

class Recognition():
    def __init__(self):
        self.model = insightface.model_zoo.get_model('arcface_r100_v1')
        self.model.prepare(ctx_id = -1)
        
    def get_embedding(self, img):
        emb = self.model.get_embedding(img)
        return emb