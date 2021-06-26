import numpy as np
import cv2


def rect_to_bb(rect):                               # konwersja współrzędnych prostokąta z dlib na opencv
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):                # konwersja punktów do listy numpy
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def draw_predictions(frame, rect, shape):              # rysowanie prostakątów i punktóœ na obrazie
    (x, y, w, h) = rect_to_bb(rect)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y) in shape:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

def count_ear(eyepoints):
    a = np.linalg.norm(eyepoints[1] - eyepoints[5])
    b = np.linalg.norm(eyepoints[2] - eyepoints[4])
    c = np.linalg.norm(eyepoints[0] - eyepoints[3])
    ear = (a + b)/(2*c)

    return ear
