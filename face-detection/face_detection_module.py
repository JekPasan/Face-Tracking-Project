# --------------------
# ---- the module ----
# --------------------

import cv2
import mediapipe as mp
import time

class FaceDetection():
    def __init__(self, detection_confidence=0.76):
        self.detection_confidence = detection_confidence

        self.cap = cv2.VideoCapture(0)

        self.mp_faces = mp.solutions.face_detection
        self.faces = self.mp_faces.FaceDetection(self.detection_confidence)

        self.mp_draw = mp.solutions.drawing_utils
    
    def find_faces(self, img, draw=True):
        self.img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faces.process(self.img_rgb)
        
        bounding_boxes = []

        if draw and self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bbox_location = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                bounding_box = int(bbox_location.xmin * w), int(bbox_location.ymin * h), int(bbox_location.width * w), \
                    int(bbox_location.height * h)
                bounding_boxes.append([id, bounding_box, detection.score])

                img = self.custom_draw(img, bounding_box)
                cv2.putText(img, str(int(detection.score[0]*100))+"%", (bounding_box[0], bounding_box[1]-10), \
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (33, 10, 89), 2)

        return img, bounding_boxes
    
    def custom_draw(self, img, bounding_box, lenght=30, thickness=10):
        up_left, down_left, w, h = bounding_box
        up_right, down_right = up_left + w, down_left + h

        cv2.rectangle(img, bounding_box, (76, 99, 20), 2)

        cv2.line(img, (up_left, down_left), (up_left+30, down_left), (76, 99, 20), thickness)
        cv2.line(img, (up_left, down_left), (up_left, down_left+30), (76, 99, 20), thickness)

        cv2.line(img, (up_right, down_right), (up_right-30, down_right), (76, 99, 20), thickness)
        cv2.line(img, (up_right, down_right), (up_right, down_right-30), (76, 99, 20), thickness)

        return img

def main():
    detector = FaceDetection()
    cap = cv2.VideoCapture(0)
    current_time = 0
    previous_time = 0
    fps = 0

    while True:
        succes, img = cap.read()
        current_time = time.time()
        img, bounding_boxes = detector.find_faces(img)
        fps = int(1/(current_time-previous_time))
        previous_time = current_time
        cv2.putText(img, str(fps), (5, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

        cv2.imshow("Video", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()