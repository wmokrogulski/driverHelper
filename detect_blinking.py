
from project_utils import *
from eye_analyzer import *
import argparse
import cv2
import time
import dlib

EYE_AR_TRESH = 0.3          #prog
EYE_AR_CONSEC_FRAMES = 3    #liczba ramek dla ktorych ma byc przekroczony prog zeby zaliczyc jako mrugniecie

COUNTER = 0
TOTAL = 0
class BlinksAnalyzer:
    def parsing(self):
        ap=argparse.ArgumentParser()
        ap.add_argument("-p","--shape-predictor", required=True,
                help="path to facial plandmark predictor")
        # ap.add_argument("-v", "--video",type=str,default="",help="path to input video file")
        args = vars(ap.parse_args())


    def write_blinks(self,avgEAR, frame):
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10,30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
        cv2.putText(frame, "EAR: {:.2f}".format(avgEAR), (300,30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
