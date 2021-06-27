import cv2
import dlib
from project_utils import *
from config import *
from eye_analyzer import *
from detect_blinking import *


class FaceAnalyzer:

    def __init__(self, sp_path=SP_PATH):
        self.detector = dlib.get_frontal_face_detector()    # detektor twarzy
        self.predictor = dlib.shape_predictor(sp_path)      # predyktor pkt na twarzach
        self.ea=EyeAnalyzer()
        self.cap = None                                     # kamera

    def cam_init(self):
        self.cap = cv2.VideoCapture(0)                      # kamera inicjalizacja
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)         # ustawienie szerokości kamery

    def analyse_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # zamiana na obraz w skali szarości
        rects = self.detector(gray, 1)                      # wyznaczanie współrzędnych prostokątów zawierających twarze
        for rect in rects:                                  # wykrywanie punktów dla każdej twarzy
            shapes = self.predictor(gray, rect)
            shapes = shape_to_np(shapes)
            self.ea.analyze_eyes(shapes)
            draw_predictions(frame, rect, shapes)            # rysowanie punktów i prostokąta
            print(f'shapes: {shapes}')
        lep, rep = EyeAnalyzer.analyze_eyes(EyeAnalyzer, shapes)
        earAVG =EyeAnalyzer.analyze_eye(EyeAnalyzer, lep, rep)

        return frame,earAVG

    def analyse_still_image(self, image= 'images/lewe_zamkniete_Julka.JPG'):
        frame = cv2.imread(image)                           # wczytanie obrazu
        frame = self.analyse_frame(frame)                   # analiza
        cv2.imshow('frame', frame)                          # wyświetlenie obrazu
        k = cv2.waitKey(0)                                  # oczekiwanie na klawisz
        if k == 27 or k == ord('q'):                        # dla q i Esc zamyka okno
            cv2.destroyAllWindows()

    def analyse_camera_view(self,earAVG):                          # to samo tylko dla obrazu z kamery
        self.cam_init()
        while True:
            ret, frame = self.cap.read()
            frame = self.analyse_frame(frame)
            cv2.imshow('frame', frame)
            if earAVG < EYE_AR_TRESH:
                COUNTER=+ 1 #liczba klatek

            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:

                    TOTAL=+1 #liczba naliczonych mrugniec

                COUNTER=0 #zerujemy licznik klatek
            BlinksAnalyzer.write_blinks(earAVG,frame)
            k = cv2.waitKey(1)
            if k == 27 or k == ord('q'):
                break

    def run(self):                                          # funkcja do wykonania
        # self.analyse_camera_view()
        self.analyse_still_image('images/lewe_zamkniete_Julka.JPG')


if __name__ == '__main__':                                  # uruchomienie programu z tego pliku
    fa = FaceAnalyzer()
    fa.run()
