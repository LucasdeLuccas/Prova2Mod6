import cv2
import threading
import numpy as np


cascPath = "haarcascade_frontalface_default.xml"
eyePath = "haarcascade_eye.xml"


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + eyePath)



if face_cascade.empty() or eye_cascade.empty():
    print("Erro: Não foi possível carregar os classificadores Haar Cascade.")
    exit()


cap = cv2.VideoCapture('video/la_cabra.mp4')
if not cap.isOpened():
    print("Erro: Não foi possível abrir o arquivo de vídeo.")
    exit()

cap_lock = threading.Lock()

while True:
    ret, img = cap.read()
    if not ret:
        break
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
   
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
   
    
    
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (55, 1, 10), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    
  
   
    cv2.imshow('img', img)
    
   
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()