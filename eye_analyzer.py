import cv2
import numpy
from project_utils import *

EYE_AR_TRESH = 0.3


class EyeAnalyzer:
    def extract_eyes(self, coords):
        lefteyepoints = coords[36:42]
        righteyepoints = coords[42:48]
        print('lefteyepoints', lefteyepoints, 'righteyepoints', righteyepoints)
        return lefteyepoints, righteyepoints

    def disp_ear(self, frame, left_ear, right_ear):
        cv2.putText(frame, 'left EAR: {:.3f}'.format(left_ear), (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'right EAR: {:.3f}'.format(right_ear), (10, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        if left_ear < EYE_AR_TRESH and right_ear < EYE_AR_TRESH:
            cv2.putText(frame, 'Nie spij!', (10, 70),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            print('Nie śpij!')

    def analyze_eyes(self, lefteyepoints, righteyepoints):
        left_ear = calculate_ear(lefteyepoints)
        right_ear = calculate_ear(righteyepoints)
        print('left EAR: {}\nright EAR: {}'.format(left_ear, right_ear))
        # avgEAR = (left_ear+right_ear)/2.0
        return left_ear, right_ear
