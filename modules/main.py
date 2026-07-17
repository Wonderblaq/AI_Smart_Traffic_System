import cv2
import numpy as np
from roi import draw_roi_from_points

from modules.models import VehicleDetector


# (Also import your video_input or roi functions if they are set up)

# 1. Initialize your detector class
detector = VehicleDetector()
# 2. Hardcode your saved coordinates array from our polygon session
# (Use the 4 lane polygons you drew earlier)
MY_LANES = [
    [(158, 292), (374, 206), (14, 10), (8, 104), (92, 182)], # Lane 1
    [(4, 384), (6, 470), (84, 472), (432, 238), (336, 176)], # Lane 2
    [(232, 382), (388, 478), (628, 470), (632, 372), (416, 226)], # Lane 3
    [(304, 200), (402, 262), (636, 76), (632, 26), (486, 90)], # Lane 4
]

# 3. Start your video stream loop (use your video_input module or standard cv2.VideoCapture)
cap = cv2.VideoCapture("../ASSETS/Final_year_datavideo.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))

    # --- WRITE YOUR INTEGRATION CODE HERE ---

    # A. Build the master mask using your MY_LANES coordinates
    mask = np.zeros(frame.shape[:2], dtype="uint8")
    # (Create empty black mask, loop through MY_LANES, fill them with white)
    new_frame,masked, annotated = draw_roi_from_points(frame, MY_LANES)
    # B. Extract the ROI frame using cv2.bitwise_and

    # C. Pass the ROI frame into your detector object to get raw numeric data
    # Hint: current_detections = detector.detect(...)

    # D. Print your raw detections list to the console live as the video plays!

    # ----------------------------------------

    # Optional: Display the ROI frame to watch the video stream
    cv2.imshow("Live AI Traffic Stream", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()