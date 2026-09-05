                    **AI-Powered Traffic Congestion Prediction & Route Optimization System**

An end-to-end intelligent traffic monitoring platform designed for the Sunyani COCOBOD intersection. The system processes camera feeds, monitors lane-specific vehicle counts, logs time-series data, and uses an LSTM model to forecast traffic bottlenecks—all displayed on an interactive Streamlit web dashboard.

**SYSTEM WORKFLOW**

                [ Video / Camera Feed ]
                          │
                          ▼
            [ Spatial ROI Masking (Polygons) ]
                          │
                          ▼
             [ YOLOv8 + ByteTrack Detection ]
                          │
                          ▼
             [ Lane Analytics & Deduplication ]
                          │
                          ▼
               [ Traffic Data Logger ]
                          │
           ┌──────────────┴──────────────┐
           ▼                             ▼
    [ LSTM Model Predictions ]    [ Streamlit + Folium UI ]



**What It Does**

Lane-by-Lane Monitoring: Maps four distinct approach roads (Kumasi Rd, CBD/Township, Berekum Rd, and Atronie Rd) using custom polygon coordinates.

Accurate Tracking: Tracks vehicle base coordinates rather than bounding box centers to prevent overlap errors when cars cross lane borders.

Unique Flow Counts: Tracks vehicle IDs to count each car once per lane interval.

Time-Series Logging: Converts high-framerate video data into aggregated interval logs for deep learning.

LSTM Forecasting: Processes historical sequence logs to predict incoming traffic spikes across all approaches.

Interactive Dashboard: Embeds an annotated video stream and an interactive Folium map that dynamically displays approach traffic states using colored markers and directional vectors.
