from ultralytics import YOLO
import cv2
import time

model_path = 'models/yolov8n-pose.pt'
cap = cv2.VideoCapture(0)
model = YOLO(model_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose = False)[0]
    for result in results:
        keypoints_np = result.keypoints.xy[0].numpy()
        for keypoint_indx, keypoint in enumerate(keypoints_np):
            x, y = map(int, keypoint[:2])
            cv2.putText(frame, str(keypoint_indx), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
