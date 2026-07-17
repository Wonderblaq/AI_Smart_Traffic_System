import cv2
import ultralytics
from torchgen import model
from ultralytics import YOLO

# create YOLO class
class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # Initialize and load the YOLO model using the model_path
        self.model = YOLO(model_path)

    def detect(self, frame):
        # run frame through model and get results
        results = self.model(frame)
        # Get the first frame results from the model

        frame_result = results[0]
        detections = [] # create an empty list

        # Loop through frame_result.boxes
        for box in frame_result.boxes:
            coor = box.xyxy[0].tolist() # extract the coordinates of the box
            x1 = int(coor[0])
            y1 = int(coor[1])
            x2 = int(coor[2])
            y2 = int(coor[3])

            # Extract their confidence scores
            conf = float(box.conf[0])
            # Extract the id assigned to the detected vehicle as a tensor and convert to a whole number, e.g car = 2
            cls = int(box.cls[0])
            if cls in [2,3,5,7]:
                detections.append([x1,y1,x2,y2,conf,cls])


        return detections

