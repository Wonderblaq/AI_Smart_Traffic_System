import cv2
import numpy as np



def draw_roi_from_points(frame, polygons, display_size=(320, 240), base_size=(640, 480)):
    """
    Draw ROI polygons and return masked frame + mask, with resizing/rescaling.

    Parameters:
        frame (np.ndarray): The video frame (BGR image).
        polygons (list): List of polygons, each polygon is a list of (x,y) tuples
                         defined in the base_size coordinate system.
        display_size (tuple): Size to downscale for display (default 320x240).
        base_size (tuple): The coordinate system used for polygons (default 640x480).

    Returns:
        roi (np.ndarray): Frame with ROI applied (only polygons visible).
        mask_bgr (np.ndarray): Mask image in BGR format (white polygons, black background).
        annotated_small (np.ndarray): Downscaled annotated frame for display.
    """
    # Resize frame to base size (640x480)
    frame_resized = cv2.resize(frame, base_size)
    temp = frame_resized.copy()
    mask = np.zeros(frame_resized.shape[:2], dtype=np.uint8)

    for i, poly in enumerate(polygons):
        pts = np.array(poly, dtype=np.int32)
        # Draw polygon outline
        cv2.polylines(temp, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        # Fill mask
        cv2.fillPoly(mask, [pts], 255)
        # Compute centroid for text
        M = cv2.moments(pts)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(temp, f"ROI {i+1}", (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Apply mask
    roi = cv2.bitwise_and(frame_resized, frame_resized, mask=mask)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Downsize for display
    temp_small = cv2.resize(temp, display_size)
    roi_small = cv2.resize(roi, display_size)

    # Stack horizontally
    annotated_small = np.hstack((temp_small, roi_small))

    return roi, mask_bgr, annotated_small
