import cv2
import numpy as np

all_polygons = []
current_polygon = []


def click_event(event, x, y, flags, params):
    global current_polygon, all_polygons
    if event == cv2.EVENT_LBUTTONDOWN:
        # Scale the click coordinates back to the original 640x480 frame space
        # because the window displays a smaller 320x240 version of the frame
        scaled_x = int(x * (640 / 320))
        scaled_y = int(y * (480 / 240))

        current_polygon.append((scaled_x, scaled_y))
        if len(current_polygon) == 5:
            all_polygons.append(current_polygon.copy())
            current_polygon.clear()


cap = cv2.VideoCapture("../ASSETS/Final_year_datavideo.mp4")
ret, frame = cap.read()
if not ret:
    print("Could not read video.")
    exit()

frame = cv2.resize(frame, (640, 480))
cap.release()

# Define  window name
window_name = "Traffic System ROI Setup (Press Q to quit)"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, click_event)

while True:
    temp = frame.copy()

    # Draw active clicks (Red)
    for p in current_polygon:
        cv2.circle(temp, p, 5, (0, 0, 255), -1)


    # Draw completed polygons (Green)
    for i,poly in enumerate(all_polygons):
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(temp, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        # Compute centroid of polygon
        M = cv2.moments(pts)
        if M["m00"] != 0: # ensure area of the polygon is not 0
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Draw text at centroid
            cv2.putText(temp, f"ROI {i + 1}", (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    # Create the mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Fill the polygons in white
    for poly in all_polygons:
        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    # Apply mask to get the ROI
    roi = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert mask to BGR
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


    # Downsize all 3 panels to 320x240 before stacking so they fit on your desktop
    temp_small = cv2.resize(temp, (320, 240))
    mask_small = cv2.resize(mask_bgr, (320, 240))
    roi_small = cv2.resize(roi, (320, 240))

    # Stack them horizontally (Total width: 960 pixels)
    combined_display = np.hstack((temp_small, mask_small, roi_small))

    # Show everything in our single interactive window!
    cv2.imshow(window_name, combined_display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
print("lanes = [")
for i, poly in enumerate(all_polygons):
    print(f"    {poly}, # Lane {i+1}")
print("]"),