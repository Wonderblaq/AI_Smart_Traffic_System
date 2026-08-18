# AI-Powered Traffic Congestion Prediction & Route Optimization System

An end-to-end intelligent transportation system that combines real-time computer vision,
lane-level spatial analytics, and time-series forecasting. 
The pipeline detects and tracks vehicles from live/recorded video, aggregates lane-specific density and flow metrics, and logs structured temporal data for deep learning-based congestion prediction and signal optimization.

---

## Architecture Overview
```
+---------------------------+
                            |     Input Video Stream    |
                            +-------------+-------------+
                                          |
                                          v
                            +---------------------------+
                            |  Region of Interest (ROI) |
                            |      Poly-Masking         |
                            +-------------+-------------+
                                          |
                                          v
                            +---------------------------+
                            |   YOLOv8 + ByteTrack      |
                            |  Detection & ID Tracking  |
                            +-------------+-------------+
                                          |
                                          v
                            +---------------------------+
                            |      LaneAnalytics        |
                            | (Point-in-Poly / Sets)    |
                            +-------------+-------------+
                                          |
                                          v
                            +---------------------------+
                            |    TrafficDataLogger      |
                            | (CSV Time-Series Export)  |
                            +-------------+-------------+
                                          |
                                          v
                            +---------------------------+
                            |    LSTM Neural Network    |
                            |  (Congestion Prediction)  |
                            +---------------------------+
```

## Key Features

* **Multi-Lane Spatial Analytics:** Uses custom polygon ROIs to monitor individual lanes.
* **Tire-Ground Point Tracking:** Computes bottom-center coordinates $(c_x, y_2)$ to accurately assign cars to lanes.
* **Deduplicated Flow Counting:** Uses Python sets to log unique vehicle IDs per lane without double-counting.
* **Time-Series Aggregation:** Aggregates 30 FPS video telemetry into average density ($\bar{D}$) and traffic volume ($V$) metrics.
* **LSTM-Ready Data Output:** Automatically exports structured CSV logs for neural network training.

---

## Project Structure

```text
AI_smart_traffic_system/
├── ASSETS/                         # Video files
├── modules/
│   ├── models.py                   # YOLO + ByteTrack engine
│   ├── analytics.py                # LaneAnalytics spatial engine
│   ├── data_logger.py              # Time-series CSV logger
│   └── main.py                     # Execution pipeline
├── roi.py                          # ROI polygon utilities
├── traffic_data.csv                # Generated dataset
└── requirements.txt                # Dependencies
