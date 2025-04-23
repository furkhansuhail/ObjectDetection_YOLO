import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch
import random
from ultralytics import YOLO
from PIL import Image
from torch import nn
from torchvision import transforms


class ObjectDetection:
    def __init__(self):
        self.VideoPath = ("sample-videos-master/people-detection.mp4")
        self.cap = None
        self.ModelDriver()

    def ModelDriver(self):
        self.readVideo()
        self.VideoProcessing()

    def readVideo(self):
        if not os.path.exists(self.VideoPath):
            raise FileNotFoundError(f"Video file not found: {self.VideoPath}")
        self.cap = cv2.VideoCapture(self.VideoPath)

    def detect_objects(self, image):
        # Load YOLO model
        model = YOLO('yolov8n.pt')  # Load the model

        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform detection
        results = model(image_rgb)[0]

        # Create a copy of the image for drawing
        annotated_image = image_rgb.copy()

        # Generate random colors for classes
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

        # Process detections
        boxes = results.boxes

        return boxes, results.names, annotated_image, colors

    def ProcessedImage(self, image, confidence_threshold):
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get detection results
        boxes, class_names, annotated_image, colors = self.detect_objects(image)

        # Process each detected object and apply confidence threshold filtering
        class_labels = {}
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            if confidence > confidence_threshold:
                class_id = int(box.cls[0])
                class_name = class_names[class_id]
                color = colors[class_id % len(colors)].tolist()
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                class_labels[class_name] = color

        # Plot results
        plt.figure(figsize=(15, 7))

        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(original_image)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Detected Objects')
        plt.imshow(annotated_image)
        plt.axis('off')

        # Add legend
        legend_handles = []
        for class_name, color in class_labels.items():
            normalized_color = np.array(color) / 255.0
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=class_name,
                                             markerfacecolor=normalized_color, markersize=10))

        plt.legend(handles=legend_handles, loc='upper right', title='Classes')
        plt.tight_layout()
        plt.show()

    def get_random_frame(self):
        video_capture = self.cap
        if not video_capture.isOpened():
            return None

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return None

        random_frame_number = random.randint(0, total_frames - 1)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
        success, frame = video_capture.read()
        if success:
            return frame
        else:
            return None

    def process_full_video(self, output_path="annotated_output.mp4", confidence_threshold=0.3):
        model = YOLO('yolov8n.pt')  # Load the model

        if not self.cap.isOpened():
            print("❌ Failed to open video.")
            return

        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define the codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        print("🚀 Starting object detection on video...")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convert to RGB for YOLO
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(image_rgb)[0]

            boxes = results.boxes
            class_names = results.names
            colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

            for box in boxes:
                confidence = float(box.conf[0])
                if confidence < confidence_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_name = class_names[class_id]
                color = colors[class_id % len(colors)].tolist()

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out.write(frame)

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")

        self.cap.release()
        out.release()
        print(f"✅ Detection complete! Annotated video saved to: {output_path}")

    def VideoProcessing(self):
        ret, _ = self.cap.read()
        random_frame = self.get_random_frame()
        # self.cap.release()

        if not ret or random_frame is None:
            print("❌ Failed to read the video frame.")
            return

        # Display the random frame (optional)
        plt.imshow(random_frame[:, :, ::-1])
        plt.title("Random Frame")
        plt.axis("off")
        plt.show()

        # Run detection
        self.ProcessedImage(random_frame, confidence_threshold=0.2)
        self.process_full_video()


if __name__ == '__main__':
    ObjectDetection()