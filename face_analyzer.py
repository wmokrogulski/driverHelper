import cv2
import dlib
from project_utils import *
from config import *


class FaceAnalyzer:

    def __init__(self, sp_path=SP_PATH):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(sp_path)
        self.cap = None
        self.shapes = []
        self.rects = []

    def cam_init(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)

    def analyse_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.rects = self.detector(gray, 1)
        self.shapes.clear()
        for rect in self.rects:
            shape = self.predictor(gray, rect)
            shape = shape_to_np(shape)
            self.shapes.append(shape)
            draw_predictions(frame, rect, shape)
        print(f'shapes: {self.shapes}')
        return frame

    def analyse_still_image(self, image='eyes_closed.jpg'):
        frame = cv2.imread(image)
        frame = self.analyse_frame(frame)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(0)
        if k == 27 or k == ord('q'):
            cv2.destroyAllWindows()

    def analyse_camera_view(self):
        self.cam_init()
        while True:
            ret, frame = self.cap.read()
            frame = self.analyse_frame(frame)
            cv2.imshow('frame', frame)
            k = cv2.waitKey(1)
            if k == 27 or k == ord('q'):
                break

    def run(self):
        # self.analyse_camera_view()
        self.analyse_still_image()


if __name__ == '__main__':
    fa = FaceAnalyzer()
    fa.run()
