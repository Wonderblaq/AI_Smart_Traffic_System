import cv2
from modules.models import VehicleDetector
from roi import draw_roi_from_points
from modules.analytics import LaneAnalyzer
from modules.TrafficDataLogger import TrafficDataLogger




# Initialize your detector class
detector = VehicleDetector()

# Your hardcoded polygon coordinates
MY_LANES = [
    [(6, 98), (154, 274), (428, 224), (116, 56), (16, 2)],       # Lane 1
    [(10, 378), (298, 170), (448, 202), (244, 388), (72, 476)],   # Lane 2
    [(628, 22), (250, 198), (350, 300), (542, 146), (636, 66)],   # Lane 3
    [(204, 380), (386, 472), (632, 474), (630, 366), (420, 208)], # Lane 4
]

# INITIALIZE ANALYTICS ENGINE BEFORE THE LOOP
# We pass MY_LANES so it creates sets and polygon arrays for all 4 lanes
analytics = LaneAnalyzer(MY_LANES)

# Record every 5 seconds of video time (at 30 FPS = 150 frames per row)
logger = TrafficDataLogger(num_lanes=len(MY_LANES), csv_filename="traffic_data.csv", window_seconds=5, fps=30)

# Start your video stream loop
cap = cv2.VideoCapture("../ASSETS/Final_year_datavideo.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Build the mask and extract the ROI frame using your custom function
    roi_frame, masked, annotated = draw_roi_from_points(frame, MY_LANES)

    # Pass the ROI frame into your detector object to get raw numeric data
    current_detections = detector.track(roi_frame)

    # PASS DETECTIONS TO ANALYTICS ENGINE FOR THIS FRAME
    # Returns two dicts:
    # live_density = {0: count, 1: count, 2: count, 3: count}
    # total_counts = {0: total, 1: total, 2: total, 3: total}
    live_density, total_counts = analytics.process_frame(current_detections)


    # Log frame stats to aggregator
    logger.log_frame(live_density, total_counts)

    # Print analytics live to console to verify during testing
    print(f"Live Density per lane: {live_density} | Cumulative Total: {total_counts}")

    # Loop through each vehicle found in current_detections
    for det in current_detections:
        # Unpack the raw data
        x1, y1, x2, y2, track_id, conf, cls = det

        # Draw bounding box
        cv2.rectangle(roi_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Draw tire contact point (bottom-center) so you can visually verify
        # which point is being checked against the lane polygons!
        cx = int((x1 + x2) / 2)
        cy = int(y2)
        cv2.circle(roi_frame, (cx, cy), 4, (0, 0, 255), -1)  # Red dot at tire contact point

        # Add text label showing track_id
        label = f"ID #{track_id} ({conf:.2f})"
        cv2.putText(
            roi_frame,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )


    # Display the lane stats at the top of the video feed
    y_offset = 30
    for lane_idx in range(len(MY_LANES)):
        info_text = f"Lane {lane_idx + 1} -> Active: {live_density[lane_idx]} | Total: {total_counts[lane_idx]}"
        cv2.putText(
            roi_frame,
            info_text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0), # Cyan text

        )
        y_offset += 25  # Move next lane text down by 25 pixels

    # Display window
    cv2.imshow("Live AI Traffic Stream", roi_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()