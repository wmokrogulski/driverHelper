import cv2
import dlib
import numpy as np

sp_path='shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(sp_path)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,500)
while True:
    ret, frame=cap.read()
    cv2.imshow('frame',frame)
    # frame=cv2.resize(frame,(500,frame.shape[:2]))
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects=detector(gray,1)

    cv2.waitKey(10)
