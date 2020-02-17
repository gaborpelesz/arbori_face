import insightface
import numpy as np
import cv2
from detection.FaceDetector import FaceDetector

class RetinaFace(FaceDetector):
    def __init__(self):
        super().__init__()
        self.desiredLeftEye = (0.32,0.32)
        self.desiredFaceWidth = 112
        self.desiredFaceHeight = 112

        self.model = insightface.model_zoo.get_model('retinaface_r50_v1')
        self.model.prepare(ctx_id = -1, nms=0.4)

    def detect(self, image_frame):
        bboxes, landmarks = self.model.detect(image_frame, threshold=0.5, scale=1.0)

        bboxes = bboxes.astype(np.int)
        landmarks = self._convert_landmarks_to_coords(landmarks)

        return bboxes, landmarks

    def align_face(self, image, rightEyeCenter, leftEyeCenter):
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2, (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth

        scale = desiredDist / dist if dist != 0 else 0

        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])


        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        aligned_face = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        return aligned_face

    def _convert_landmarks_to_coords(self, landmarks):
        # convert landmarks into appropriate format
        landmarks = landmarks.astype(int)
        for i in range(len(landmarks)):
            landmarks[i] = list(map(lambda x: x.tolist(), landmarks[i]))
        return landmarks