import cv2
import numpy
from project_utils import *
EYE_AR_TRESH = 0.3

class EyeAnalyzer:
    def analyze_eyes(self, coords):
        lefteyepoints = coords[36:42]
        righteyepoints = coords[42:48]
        print('lefteyepoints', lefteyepoints, 'righteyepoints', righteyepoints)
        return (lefteyepoints,righteyepoints)

    def analyze_eye(self, lefteyepoints, righteyepoints):
        leftear = count_ear(lefteyepoints)
        rightear = count_ear(righteyepoints)
        print(leftear, rightear)
        avgEAR = (leftear+rightear)/2.0
        if leftear < EYE_AR_TRESH and rightear < EYE_AR_TRESH:
            print('Nie śpij!')





