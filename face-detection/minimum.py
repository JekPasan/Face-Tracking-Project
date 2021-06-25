# -----------------------------------------
# ---- the bare minimum necessary code ----
# -----------------------------------------

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mp_faces = mp.solutions.face_detection
faces = mp_faces.FaceDetection(min_detection_confidence=0.76)

mp_draw = mp.solutions.drawing_utils

current_time = 0
previous_time = 0
fps = 0

while True:
    succes, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faces.process(img_rgb)

    if results.detections:
        for id, detection in enumerate(results.detections):
            bbox_location = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            bounding_box = int(bbox_location.xmin * w), int(bbox_location.ymin * h), int(bbox_location.width * w), int(bbox_location.height * h)
            cv2.rectangle(img, bounding_box, (76, 99, 20), 2)
            cv2.putText(img, str(int(detection.score[0]*100))+"%", (bounding_box[0], bounding_box[1]-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (33, 10, 89), 2)
    
    
    current_time = time.time()
    fps = int(1/(current_time-previous_time))
    previous_time = current_time
    cv2.putText(img, str(fps), (5, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

    cv2.imshow("Video", img)
    cv2.waitKey(1)
    