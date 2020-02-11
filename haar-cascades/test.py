import cv2

def detect(gray):
    
    faces = face_cascade.detectMultiScale(gray)

    print('Detected faces: {}'.format(len(faces)), end='\r')

    for x, y, width, height in faces:
        cv2.rectangle(img, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)

face_cascade = cv2.CascadeClassifier("cascade.xml")
mirror = False
shape = (1280, 720) # FHD width, height
height_to = 224

w = int(height_to/shape[1] * shape[0])

cam = cv2.VideoCapture(0)
while True:
    ret_val, frame = cam.read()
    
    img = cv2.resize(frame, (w,height_to))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #gray = cv2.convertScaleAbs(gray, alpha=1, beta=80)
    equalized = cv2.equalizeHist(gray)


    detect(equalized)
    #detect(equalized)
    cv2.imshow('Detect faces', img)
    #cv2.imshow('Detect equalized', equalized)
    if cv2.waitKey(1) == 27: 
        break  # esc to quit
cv2.destroyAllWindows()