import cv2
import numpy as np

# -----------------------------
# Global Variables
# -----------------------------
points = []

# -----------------------------
# Mouse Callback
# -----------------------------
def draw_ROI(event, x, y, flags, params):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))



# Load Video
cap = cv2.VideoCapture("../ASSETS/Final_year_datavideo.mp4")

ret, frame = cap.read()
frame = cv2.resize(frame, (640, 480))


if not ret:
    print("Could not read video.")
    exit()


# Create Window
cv2.namedWindow("ROI Selector")
cv2.setMouseCallback("ROI Selector", draw_ROI)


# Main Loop
while True:

    # Always work on a copy for drawing
    temp = frame.copy()

    # Draw clicked points
    for p in points:
        cv2.circle(temp, p, 5, (0, 0, 255), -1)

    # Default image to display
    roi_frame = frame.copy()

    # Only create ROI when polygon exists
    if len(points) >= 4:

        # Draw polygon
        cv2.polylines(
            temp,
            [np.array(points, dtype=np.int32)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2
        )

        # Create empty mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Fill polygon
        cv2.fillPoly(
            mask,
            [np.array(points, dtype=np.int32)],
            255
        )

        # Extract ROI
        roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Show masked frame
        cv2.imshow("Masked Frame", roi_frame)

    # Always show selector
    cv2.imshow("ROI Selector", temp)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(points)