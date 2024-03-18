
import cv2
from ultralytics import YOLO
import numpy as np
import math


def rgb_distance(color1, color2):
    """
    Calculate the normalized Euclidean distance between two RGB colors.
    
    Parameters:
    - color1: A tuple representing the first RGB color (R1, G1, B1).
    - color2: A tuple representing the second RGB color (R2, G2, B2).
    
    Returns:
    - The normalized Euclidean distance between the two colors, in the range [0, 1].
    """
    # Compute the Euclidean distance
    distance = math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))
    
    # Maximum possible Euclidean distance in RGB space
    max_distance = math.sqrt(3 * (255 ** 2))
    
    # Normalize the distance
    normalized_distance = distance / max_distance
    
    return normalized_distance


def get_average_color(image, coords, radius):
    """
    Calculate the average color of the specified circular region in an image.
    
    Parameters:
    - image: The image as a NumPy array (e.g., loaded using cv2.imread).
    - coords: The center coordinates of the circular region.
    - radius: The radius of the circular region.
    
    Returns:
    - A tuple of (B, G, R) values representing the average color of the region.
    """
    x, y = coords
    # Ensure coordinates and radius are within the image boundaries
    height, width = image.shape[:2]
    x = np.clip(x, 0, width-1)
    y = np.clip(y, 0, height-1)
    radius = np.clip(radius, 0, min(width, height))

    # Create a mask with the same dimensions as the image, initially all zeros (black)
    mask = np.zeros((height, width), np.uint8)

    # Draw a filled circle (region of interest) on the mask
    cv2.circle(mask, (x, y), radius, (255, 255, 255), thickness=-1)

    # Use the mask to select the region of interest
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Calculate the sum and count of pixels in the region to find the average
    sum_b, sum_g, sum_r, count = [np.sum(masked_image[:, :, i][mask == 255]) for i in range(3)] + [np.count_nonzero(mask == 255)]
    
    # Avoid division by zero
    if count == 0:
        return (0, 0, 0)
    
    # Calculate average color
    average_color = (sum_b // count, sum_g // count, sum_r // count)

    return average_color


# Load the YOLOv8 model
model = YOLO('yolov8m-pose.pt')

# Open the video file
video_path = 0 #"gymnasts.mp4"
cap = cv2.VideoCapture(video_path)

# authorizedColors = [(102, 0, 26), (18, 105, 6), (242, 242, 242), (0, 2, 99)]

# authorizedColors = [[65, 81, 70], [116, 65, 80], [55, 58, 73]]

authorizedColors = [[129, 62, 79], [51,60,80], [54,87,76]]

authorizedColors = [(color[2], color[1], color[0]) for color in authorizedColors]

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = frame #results[0].plot()

        for person, box in zip(results[0].keypoints.xy, results[0].boxes.xyxy):
            average_colors = [] 

            personNew = []

            for bodypart in [5, 6, 11, 12]: 
                personNew.append(person[bodypart])

            for keypoint in personNew: 
                coord = int(keypoint[0]), int(keypoint[1])
                average_colors.append((get_average_color(frame, coord, 30), coord))

                cv2.circle(annotated_frame, coord, 30, average_colors[-1][0], -1)

            uniform = False

            

            for color, coord in average_colors: 
                if coord[0] == 0 and coord[1] == 0: 
                    continue

                for authorizedColor in authorizedColors:

                    similarity =  1 - rgb_distance(color, authorizedColor)

                    if similarity > 0.9: 
                        print(authorizedColor)
                        cv2.circle(annotated_frame, coord, 30, authorizedColor, 5)
                        uniform = True
                        break

                if uniform is True: 
                    break

            box = list(map(int, box))

            if uniform: 
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
