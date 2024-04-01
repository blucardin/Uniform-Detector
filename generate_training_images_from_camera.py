from ultralytics import YOLO
import cv2
import numpy as np
import glob
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

model = YOLO('yolov8m-pose.pt')

i = 700 

# Open the video file
video_path = 0 #"gymnasts.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    results = model.predict(source=frame)

    for box in results[0].boxes.xyxy:

        x1, y1, x2, y2 = list(map(int, box))

        cropped = frame[y1:y2, x1:x2]

        name = f"{i:04d}.jpg"


        cv2.imshow("YOLOv8 Tracking", cropped)

        cv2.imwrite("trainingImages2/uniform/"+name, cropped)
        i += 1
            
        key = cv2.waitKey(1) & 0xFF 

        # Break the loop if 'q' is pressed
        if key == ord("d"):
            break