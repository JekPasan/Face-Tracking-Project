# -----------------------------------------
# ---- the bare minimum necessary code ----
# -----------------------------------------

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mp_meshes = mp.solutions.face_mesh
meshes = mp_meshes.FaceMesh()

mp_draw = mp.solutions.drawing_utils
specs = mp_draw.DrawingSpec(thickness=1, circle_radius=1)

current_time = 0
previous_time = 0
fps = 0

while True:
    succes, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = meshes.process(img_rgb)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(img, landmarks, mp_meshes.FACE_CONNECTIONS, specs, specs)

            for id, landmark in enumerate(landmarks.landmark):
                h, w, c = img.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                print(id, x, y)

    current_time = time.time()
    fps = int(1/(current_time-previous_time))
    previous_time = current_time
    cv2.putText(img, str(fps), (5, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

    cv2.imshow("Video", img)
    cv2.waitKey(1)