from ultralytics import YOLO
import cv2
import numpy as np
import glob
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

model = YOLO('yolov8m-pose.pt')

i = 0 

files = glob.glob("rawImages/*.*")


for filename in files:

    image = np.array(Image.open(filename).convert('RGB'))[:, :, ::-1]

    results = model.predict(source=image)

    for box in results[0].boxes.xyxy:

        x1, y1, x2, y2 = list(map(int, box))

        cropped = image[y1:y2, x1:x2]

        name = f"{i:04d}" + filename + ".jpg"

        # Display the annotated frame
        while True: 
            cv2.imshow("YOLOv8 Tracking", cropped)

            key = cv2.waitKey(1) & 0xFF 

            # Break the loop if 'q' is pressed
            if key == ord("u"):
                cv2.imwrite("trainingImages/uniform/"+name, cropped)
                i += 1
                break

            if key == ord("n"):
                cv2.imwrite("trainingImages/non-uniform/"+name, cropped)
                i += 1
                break

            # Break the loop if 'q' is pressed
            if key == ord("d"):
                break

    print(filename)