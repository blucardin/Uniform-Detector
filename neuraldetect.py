
import cv2
from ultralytics import YOLO
import numpy as np
import math
import keras

print(keras.__version__)

# Load the YOLOv8 model
model = YOLO('yolov8m-pose.pt')

uniformModel = keras.saving.load_model("2024-03-25-19-39-20.keras")
softmax_layer = keras.layers.Softmax()

uniformModel.summary()


# Open the video file
video_path = 0 #"gymnasts.mp4"
cap = cv2.VideoCapture(video_path)


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = frame
        
         #results[0].plot()

        for person, box in zip(results[0].keypoints.xy, results[0].boxes.xyxy):

            # personNew = []

            # for bodypart in [5, 6, 11, 12]: 
            #     personNew.append(person[bodypart])

            # coords = []

            # for keypoint in personNew: 
            #     if keypoint[0] != 0 and keypoint[1] != 0: 
            #         coords.append((int(keypoint[0]), int(keypoint[1])))


            x1, y1, x2, y2 = list(map(int, box))

            cropped = cv2.resize(annotated_frame[y1:y2, x1:x2], (500, 1000))

            cropped = cropped.astype("float32")

            print(cropped.shape)

            cropped = np.expand_dims(cropped, axis=0)

            print(cropped.shape)

            uniform = np.argmax(softmax_layer(uniformModel.predict(cropped))[0])

            box = list(map(int, box))

            print(softmax_layer(uniformModel.predict(cropped))[0])

            if uniform == 1: 
                cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 5)
                print("Uniform: True")
            else: 
                cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 5)
                print("Uniform: False")

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
