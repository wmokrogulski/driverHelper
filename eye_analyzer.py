import cv2
import numpy
EYE_AR_TRESH = 0.3


class EyeAnalyzer:

    def analyze_eye(self, coords):
        al = numpy.linalg.norm(coords[37] - coords[41])           # wyliczenie poziomu otwarcia oka lewego
        bl = numpy.linalg.norm(coords[38] - coords[40])
        cl = numpy.linalg.norm(coords[36] - coords[39])
        earl = (al + bl)/(2*cl)

        ar = numpy.linalg.norm(coords[43] - coords[47])           # wyliczenie poziomu otwarcia oka lewego
        br = numpy.linalg.norm(coords[44] - coords[46])
        cr = numpy.linalg.norm(coords[42] - coords[45])
        earr = (ar + br)/(2*cr)
        print(earl, earr)

        if earl < EYE_AR_TRESH and earr < EYE_AR_TRESH:
            print('Nie śpij!')


    def analyze_eyes(self, coords):
        pass
