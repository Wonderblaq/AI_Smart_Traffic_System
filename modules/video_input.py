import cv2
import numpy as np
import os
from ultralytics import YOLO

# Open video path
# Current video file is only a testing video, A drone footage of a road intersection would be used once available
traffic_vid = cv2.VideoCapture("../ASSETS/Final_year_datavideo.mp4")

# Check if video opened
if traffic_vid.isOpened():
    print("Video opened")
    print("Width:", int(traffic_vid.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("Height:", int(traffic_vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
else:
    print("Video not opened")
    exit()

cv2.namedWindow("frame")  # create window once
model =YOLO("yolov8n.pt")
while True:
    ret, frame = traffic_vid.read()   # read frame inside loop
    if not ret:
        break  # stop if no frame is returned

    temp = frame.copy()
    resized_frame = cv2.resize(temp, (640, 480))  # resize for display
    resized_frame = model.predict(resized_frame)
    resized_frame = resized_frame[0]
    cv2.imshow("frame", resized_frame.plot())

    # quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

traffic_vid.release()
cv2.destroyAllWindows()
