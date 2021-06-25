# --------------------
# ---- the module ----
# --------------------

import cv2
import mediapipe as mp
import time

class FaceMesh():
    def __init__(self, static_mode=False, face_numbers=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.static_mode = static_mode
        self.face_numbers = face_numbers
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.cap = cv2.VideoCapture(0)

        self.mp_meshes = mp.solutions.face_mesh
        self.meshes = self.mp_meshes.FaceMesh(self.static_mode, self.face_numbers, self.detection_confidence, self.tracking_confidence)

        self.mp_draw = mp.solutions.drawing_utils
        self.specs = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1)
    
    def find_faces(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.meshes.process(img_rgb)
        faces = []
        face = []

        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, face_landmarks, self.mp_meshes.FACE_CONNECTIONS, self.specs, self.specs)

                for id, landmark in enumerate(face_landmarks.landmark):
                    h, w, c = img.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    face.append([x, y])
                faces.append(face)
        return img, faces

def main():
    cap = cv2.VideoCapture(0)
    current_time = 0
    previous_time = 0
    fps = 0
    detector = FaceMesh()

    while True:
        succes, img = cap.read()
        img, faces = detector.find_faces(img)
        if len(faces) != 0:
            print(len(faces))
        current_time = time.time()
        fps = int(1/(current_time-previous_time))
        previous_time = current_time
        cv2.putText(img, str(fps), (5, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

        cv2.imshow("Video", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()