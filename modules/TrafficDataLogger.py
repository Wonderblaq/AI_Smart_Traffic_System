import csv
import datetime


class TrafficDataLogger:
    """
    Aggregates live frame-by-frame lane density and volume stats
    over fixed time windows and saves them to CSV for LSTM training.
    """

    def __init__(self, num_lanes, csv_filename="traffic_data.csv", window_seconds=10, fps=30):
        self.num_lanes = num_lanes
        self.csv_filename = csv_filename
        self.window_frames = window_seconds * fps  # e.g., 10s * 30fps = 300 frames

        self.frame_counter = 0
        self.density_sums = {i: 0 for i in range(num_lanes)}
        self.last_cumulative_counts = {i: 0 for i in range(num_lanes)}

        self._init_csv()

    def _init_csv(self):
        headers = ["timestamp"]
        for i in range(self.num_lanes):
            headers.append(f"lane_{i + 1}_avg_density")
            headers.append(f"lane_{i + 1}_volume")

        with open(self.csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    def log_frame(self, live_density, cumulative_totals):
        self.frame_counter += 1

        for lane_idx in range(self.num_lanes):
            self.density_sums[lane_idx] += live_density[lane_idx]

        if self.frame_counter >= self.window_frames:
            self._write_window_to_csv(cumulative_totals)
            self._reset_window_state(cumulative_totals)

    def _write_window_to_csv(self, current_cumulative_totals):
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [now_str]

        for lane_idx in range(self.num_lanes):
            avg_density = self.density_sums[lane_idx] / self.frame_counter
            volume = current_cumulative_totals[lane_idx] - self.last_cumulative_counts[lane_idx]

            row.append(f"{avg_density:.2f}")
            row.append(volume)

        with open(self.csv_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)

        print(f"[DATA LOGGER] Saved row to {self.csv_filename}: {row}")

    def _reset_window_state(self, current_cumulative_totals):
        self.frame_counter = 0
        self.density_sums = {i: 0 for i in range(self.num_lanes)}
        self.last_cumulative_counts = current_cumulative_totals.copy()