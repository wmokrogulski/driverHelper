import cv2
import dlib
from project_utils import *
from config import *


class FaceAnalyzer:

    def __init__(self, sp_path=SP_PATH):
        self.detector = dlib.get_frontal_face_detector()    # detektor twarzy
        self.predictor = dlib.shape_predictor(sp_path)      # predyktor pkt na twarzach
        self.cap = None                                     # kamera
        self.shapes = []                                    # zbiór  punktów na twarzach
        self.rects = []                                     # boxy twarzy

    def cam_init(self):
        self.cap = cv2.VideoCapture(0)                      # kamera inicjalizacja
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)         # ustawienie szerokości kamery

    def analyse_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # zamiana na obraz w skali szarości
        self.rects = self.detector(gray, 1)                 # wyznaczanie współrzędnych prostokątów zawierających twarze
        self.shapes.clear()                                 # czysczenie, żeby dla każdej klatki była nowa lista
        for rect in self.rects:                             # wykrywanie punktów dla każdej twarzy
            shape = self.predictor(gray, rect)
            shape = shape_to_np(shape)
            self.shapes.append(shape)
            draw_predictions(frame, rect, shape)            # rysowanie punktów i prostokąta
        print(f'shapes: {self.shapes}')
        return frame

    def analyse_still_image(self, image=EYES_CLOSED_IM):
        frame = cv2.imread(image)                           # wczytanie obrazu
        frame = self.analyse_frame(frame)                   # analiza
        cv2.imshow('frame', frame)                          # wyświetlenie obrazu
        k = cv2.waitKey(0)                                  # oczekiwanie na klawisz
        if k == 27 or k == ord('q'):                        # dla q i Esc zamyka okno
            cv2.destroyAllWindows()

    def analyse_camera_view(self):                          # to samo tylko dla obrazu z kamery
        self.cam_init()
        while True:
            ret, frame = self.cap.read()
            frame = self.analyse_frame(frame)
            cv2.imshow('frame', frame)
            k = cv2.waitKey(1)
            if k == 27 or k == ord('q'):
                break

    def run(self):                                          # funkcja do wykonania
        self.analyse_camera_view()
        # self.analyse_still_image(EYES_OPEN_IM)


if __name__ == '__main__':                                  # uruchomienie programu z tego pliku
    fa = FaceAnalyzer()
    fa.run()
