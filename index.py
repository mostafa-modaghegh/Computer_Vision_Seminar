import cv2
import time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  
print(model.names) 

webcamera = cv2.VideoCapture(0)
webcamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   
webcamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   

prev_time = 0

while True:
    success, frame = webcamera.read()
    if not success:
        print("Failed to capture frame from camera.")
        break


    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time


    results = model.predict(frame, conf=0.5, imgsz=640)  

    
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, f"Total Objects: {len(results[0].boxes)}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

   
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)


    if cv2.waitKey(1) == ord('q'):
        break

webcamera.release()
cv2.destroyAllWindows()
