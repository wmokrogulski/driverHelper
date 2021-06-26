import cv2
import numpy
from project_utils import *
EYE_AR_TRESH = 0.3

class EyeAnalyzer:
    def analyze_eyes(self, coords):
        lefteyepoints = coords[34:42]
        righteyepoints = coords[42:48]
        return (lefteyepoints,righteyepoints)

    def analyze_eye(self, lefteyepoints, righteyepoints):
        leftear = policz(lefteyepoints)
        rightear = policz(righteyepoints)
        print(leftear, rightear)

        if leftear < EYE_AR_TRESH and rightear < EYE_AR_TRESH:
            print('Nie śpij!')



