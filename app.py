import cv2
import time
from ultralytics import YOLO

model = YOLO('model.pt')
#model = YOLO("yolov8n.pt")
print(model.names)

webcamera = cv2.VideoCapture(0)

prev_time = 0

while True:
    success, frame = webcamera.read()
    if not success:
        break
    
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    results = model.track(frame, conf=0.8, imgsz=320)
    
    cv2.putText(frame, f"Total: {len(results[0].boxes)}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("Live Camera", results[0].plot())

    if cv2.waitKey(1) == ord('q'):
        break

webcamera.release()
cv2.destroyAllWindows()
