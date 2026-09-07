import cv2
import numpy as np

all_polygons = []
current_polygon = []

# Original frame size and displayed panel size — change DISPLAY_W/H to resize the window
FRAME_W, FRAME_H = 640, 480
DISPLAY_W, DISPLAY_H = 320, 480  # width unchanged, height enlarged (was 320x240)  # was 320x240 — now larger


def click_event(event, x, y, flags, params):
    global current_polygon, all_polygons
    if event == cv2.EVENT_LBUTTONDOWN:
        # Only accept clicks on the FIRST panel (the frame view).
        # combined_display is 3 panels side by side: [frame | mask | roi]
        if x >= DISPLAY_W:
            return

        # Scale click coords back to the original frame space
        scaled_x = int(x * (FRAME_W / DISPLAY_W))
        scaled_y = int(y * (FRAME_H / DISPLAY_H))

        current_polygon.append((scaled_x, scaled_y))
        if len(current_polygon) == 5:
            all_polygons.append(current_polygon.copy())
            current_polygon.clear()


cap = cv2.VideoCapture(r"C:\Users\User\Videos\Screen Recordings\top-view2.mp4")
ret, frame = cap.read()
if not ret:
    print("Could not read video.")
    exit()

frame = cv2.resize(frame, (FRAME_W, FRAME_H))
cap.release()

window_name = "Traffic System ROI Setup (Press Q to quit)"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, click_event)

while True:
    temp = frame.copy()

    for p in current_polygon:
        cv2.circle(temp, p, 5, (0, 0, 255), -1)

    for i, poly in enumerate(all_polygons):
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(temp, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        M = cv2.moments(pts)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(temp, f"ROI {i + 1}", (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for poly in all_polygons:
        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    roi = cv2.bitwise_and(frame, frame, mask=mask)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Resize all 3 panels to DISPLAY_W x DISPLAY_H before stacking
    temp_small = cv2.resize(temp, (DISPLAY_W, DISPLAY_H))
    mask_small = cv2.resize(mask_bgr, (DISPLAY_W, DISPLAY_H))
    roi_small = cv2.resize(roi, (DISPLAY_W, DISPLAY_H))

    combined_display = np.hstack((temp_small, mask_small, roi_small))

    cv2.imshow(window_name, combined_display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
print("lanes = [")
for i, poly in enumerate(all_polygons):
    print(f"    {poly}, # Lane {i+1}")
print("]")