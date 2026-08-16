import ultralytics
from ultralytics import YOLO


# create YOLO class
class VehicleDetector:
    def __init__(self, model_path="yolov8s.pt"):
        # Initialize and load the YOLO model using the model_path
        self.model = YOLO(model_path)


    # Method for performing detections
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

    # tracker method , that uses bytetrack
    def track(self, frame):
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        detection_list = []
        frame_result = results[0]

        if frame_result is not None:
            for box in frame_result.boxes:
                coor = box.xyxy[0].tolist() # Extract and unpack bb box coordinates
                x1 = int(coor[0])
                y1 = int(coor[1])
                x2 = int(coor[2])
                y2 = int(coor[3])

                # Extract confidence score
                conf = float(box.conf[0])

                if box.id is not None:
                    track_id = int(box.id[0])
                else:
                    track_id = -1

                    # Extract the id assigned to the detected vehicle as a tensor and convert to a whole number, e.g car = 2
                cls = int(box.cls[0])
                if cls in [2, 3, 5, 7]:
                    detection_list.append([x1, y1, x2, y2, track_id,conf, cls])

                pass

            return detection_list









