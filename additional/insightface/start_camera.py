import cv2
import numpy as np
from retina_face_detect import Detect
from retina_face_recognition import Recognition
from numpy.linalg import norm

def boundFaces(img, bboxs, landmarks=None, name=""):
    for i, bbox in enumerate(bboxs):
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 3)
        if landmarks:
            for point in landmarks[i]:
                cv2.circle(img, tuple(point), 3, (0,255,0), -1)
        if name != "":
            cv2.putText(img, name, (int(bbox[0]), int(bbox[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA) 
    return img

def capture(detect, img_size):
    data = np.load('faces.npz')
    gabor_embs, nono_embs = data['arr_0'], data['arr_1']

    cap = cv2.VideoCapture(0)  # capture from camera
    recog = Recognition()

    images = []

    while True:
        ret_val, frame = cap.read()

        frame = cv2.resize(frame, img_size)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        bboxs, landmark = detect(frame)

        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        

        if len(bboxs) != 0:
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            bbox = bboxs[0]
            face = frame[int(bbox[1]):int(bbox[0]),int(bbox[3]):int(bbox[2])]
            
            face = cv2.resize(face, (112,112))
            embedding = recog.get_embedding(face)

            min_gabor = float('inf')
            for gabor_emb in gabor_embs:
                dist = np.dot(embedding, gabor_emb)/(norm(embedding)*norm(gabor_emb))
                if min_gabor > dist:
                    min_gabor = dist

            min_nono = float('inf')
            for nono_emb in nono_embs:
                dist = np.dot(embedding, nono_embs)/(norm(embedding)*norm(nono_emb))
                if min_nono > dist:
                    min_nono = dist
            
            if min_gabor < min_nono:
                frame = boundFaces(frame, [bbox], name='Gabor')
            else:
                frame = boundFaces(frame, [bbox], name='Nono')

            #top_left

            face = frame[int(bbox[1]):int(bbox[0]),int(bbox[3]):int(bbox[2])]
            images.append(face)

            cv2.namedWindow('Detect faces', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Detect faces', (frame.shape[0]*5,frame.shape[1]*5))

            #frame = boundFaces(frame, bboxs)
            cv2.imshow('Detect faces', frame)

            

        if cv2.waitKey(1) == 27: 
            break  # esc to quit

    # for i, image in enumerate(images):
    #    cv2.imwrite('images/faces/nono/{}.jpg'.format(i), image)

    cv2.destroyAllWindows()

def main():
    shape = (1280, 720) # FHD width, height
    h = 224
    w = int(h/shape[1] * shape[0])

    detect_object = Detect()

    capture(detect_object.detect_faces, img_size=(w,h))

if __name__ == '__main__':
    main()