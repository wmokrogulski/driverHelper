import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor()
cap = cv2.VideoCapture(0)
while True:
    ret, frame=cap.read()
    cv2.imshow('frame',frame)
    cv2.waitKey(10)
