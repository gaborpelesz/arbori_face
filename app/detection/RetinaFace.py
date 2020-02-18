import insightface
import numpy as np
import cv2
from skimage import transform as trans
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

    def align_face(self, image, landmarks):
        rightEyeCenter = landmarks[0]
        leftEyeCenter = landmarks[1]

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

    def preprocess_face(self, img, bbox=None, landmark=None):
        M = None
        image_size = [112,112]

        if landmark is not None:
            assert len(image_size)==2
            src = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041] ], dtype=np.float32 )
            if image_size[1]==112:
                src[:,0] += 8.0
            dst = landmark.astype(np.float32)

            tform = trans.SimilarityTransform()
            tform.estimate(dst, src)
            M = tform.params[0:2,:]

        if M is None:
            if bbox is None: # use center crop
                det = np.zeros(4, dtype=np.int32)
                det[0] = int(img.shape[1]*0.0625)
                det[1] = int(img.shape[0]*0.0625)
                det[2] = img.shape[1] - det[0]
                det[3] = img.shape[0] - det[1]
            else:
                det = bbox
                margin = 44
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-margin/2, 0)
                bb[1] = np.maximum(det[1]-margin/2, 0)
                bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
                bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
                ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
            if len(image_size)>0:
                ret = cv2.resize(ret, (image_size[1], image_size[0]))
            return ret 
        else: # do align using landmark
            assert len(image_size)==2
            warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
            return warped