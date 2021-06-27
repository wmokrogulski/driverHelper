import cv2
import dlib
import imutils

from project_utils import *
from config import *
from eye_analyzer import *
from detect_blinking import *


class FaceAnalyzer:

    def __init__(self, sp_path=SP_PATH):
        self.detector = dlib.get_frontal_face_detector()  # detektor twarzy
        self.predictor = dlib.shape_predictor(sp_path)  # predyktor pkt na twarzach
        self.ea = EyeAnalyzer()
        self.cap = None  # kamera

    def cam_init(self):
        self.cap = cv2.VideoCapture(0)  # kamera inicjalizacja
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)  # ustawienie szerokości kamery

    def video_init(self, video):
        self.cap = cv2.VideoCapture(video)  # kamera inicjalizacja zapisanego filmu
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)  # ustawienie szerokości kamery

    def analyse_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # zamiana na obraz w skali szarości
        rects = self.detector(gray, 1)  # wyznaczanie współrzędnych prostokątów zawierających twarze
        for rect in rects:  # wykrywanie punktów dla każdej twarzy
            shapes = self.predictor(gray, rect)
            shapes = shape_to_np(shapes)
            draw_predictions(frame, rect, shapes)  # rysowanie punktów i prostokąta
            print(f'shapes: {shapes}')
            lep, rep = self.ea.extract_eyes(shapes)
            left_ear, right_ear = self.ea.analyze_eyes(lep, rep)
            self.ea.disp_ear(frame, left_ear, right_ear)

        return frame

    def analyse_still_image(self, image):
        frame = cv2.imread(image)  # wczytanie obrazu
        frame = imutils.resize(frame, width=500)
        frame = self.analyse_frame(frame)  # analiza
        cv2.imshow('frame', frame)  # wyświetlenie obrazu
        k = cv2.waitKey(0)  # oczekiwanie na klawisz
        if k == 27 or k == ord('q'):  # dla q i Esc zamyka okno
            cv2.destroyAllWindows()

    def analyse_camera_view(self):  # to samo tylko dla obrazu z kamery
        self.cam_init()
        while True:
            ret, frame = self.cap.read()
            frame = self.analyse_frame(frame)
            cv2.imshow('frame', frame)
            k = cv2.waitKey(1)
            if k == 27 or k == ord('q'):
                break

    def analyse_video(self, video):  # to samo tylko dla obrazu z filmu
        self.video_init(video)
        while True:
            ret, frame = self.cap.read()
            frame = self.analyse_frame(frame)
            cv2.imshow('frame', frame)
            k = cv2.waitKey(1)
            if k == 27 or k == ord('q'):
                break

    def run(self):  # funkcja do wykonania
        # self.analyse_camera_view()
        self.analyse_still_image('images/eyes_closed.jpg')
        # self.analyse_video('images/test.mp4')


if __name__ == '__main__':  # uruchomienie programu z tego pliku
    fa = FaceAnalyzer()
    fa.run()
