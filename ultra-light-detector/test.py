import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare

def preprocessing(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 480))
    img_mean = np.array([127, 127, 127])
    img = (img - img_mean) / 128
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    return img

def main():
    shape = (1280, 720) # FHD width, height
    height_to = 224

    w = int(height_to/shape[1] * shape[0])

    cam = cv2.VideoCapture(0)

    onnx_model = onnx.load('ultra_light/ultra_light_models/Mb_Tiny_RFB_FD_train_input_640.onnx')
    predictor = prepare(onnx_model)
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    while True:
        ret_val, frame = cam.read()
        
        h, w, _ = frame.shape

        igm = preprocessing(frame)

        cv2.imshow('Detect faces', frame)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()