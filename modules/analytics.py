import numpy as np
import cv2


class LaneAnalyzer:
    def __init__(self, lane_polygons):

        # Convert every polygon to list of numpy array
        self.lane_polygons = [np.array(poly, dtype=np.int32) for poly in lane_polygons]

        # Create a Python set for each lane to store unique vehicle IDs
        # Key: Lane Index (0, 1, 2...) -> Value: set of track_ids seen
        self.cumulative_counts = {i : set() for i in range(len(lane_polygons))}

    def process_frame(self, tracked_detections):
        # Reset live density counts for THIS frame
        live_density = {i: 0 for i in range(len(self.lane_polygons))}

        # Loop through every vehicle active in this frame
        for det in tracked_detections:
            x1, y1, x2, y2, track_id = det[:5]

            # Calculate tire contact point at bottom-center of bounding box
            cx = int((x1 + x2) / 2)
            cy = int(y2)

            # Test which lane polygon contains (cx, cy)
            for lane_idx, poly_pts in enumerate(self.lane_polygons):
                # returns >= 0 if point is inside polygon or on boundary
                is_inside = cv2.pointPolygonTest(poly_pts, (cx, cy), False)

                if is_inside >= 0:
                    # Vehicle is inside this lane!
                    self.cumulative_counts[lane_idx].add(int(track_id))
                    live_density[lane_idx] += 1
                    break  # Stop checking other lanes once matched

        #
        # Calculate total unique count per lane by getting length of each set
        cumulative_totals = {lane_idx: len(ids) for lane_idx, ids in self.cumulative_counts.items()}

        return live_density, cumulative_totals