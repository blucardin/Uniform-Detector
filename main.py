from ultralytics import YOLO

model = YOLO('yolov8m-pose.pt')

results = model.track(source=0, show=True, tracker='bytetrack.yaml')