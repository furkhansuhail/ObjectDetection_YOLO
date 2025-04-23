import cv2
import torch
from torch import amp  # <-- updated import
import pandas as pd


class ObjectDetection():
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True).to(self.device)
        self.cap = None
        self.CapturingVideo()

    def CapturingVideo(self):
        self.cap = cv2.VideoCapture(0)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Use torch.amp.autocast for automatic mixed precision (if needed)
            # with amp.autocast(device_type='cuda'):  # <-- updated usage
            with torch.amp.autocast('cuda'):
                results = self.model(frame)

            # Draw bounding boxes
            for result in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = result
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{self.model.names[int(cls)]} {conf:.2f}",
                            (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

            cv2.imshow('Real-time Object Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

ObjectDetectionObj = ObjectDetection()
