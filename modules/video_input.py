import cv2
import numpy as np
import os

# Open video path
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

while True:
    ret, frame = traffic_vid.read()   # read frame inside loop
    if not ret:
        break  # stop if no frame is returned

    temp = frame.copy()
    resized_frame = cv2.resize(temp, (640, 480))  # resize for display
    cv2.imshow("frame", resized_frame)

    # quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

traffic_vid.release()
cv2.destroyAllWindows()
