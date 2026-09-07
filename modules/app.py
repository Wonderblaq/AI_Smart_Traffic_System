import folium
import joblib
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from tensorflow.keras.models import load_model

# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CITY AI SMART TRAFFIC CONTROLLER",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

TRAFFIC_DATA_PATH = (
    r"C:\Users\User\Downloads\final_year_project_mock_traffic_data_v2.csv"
)

LSTM_MODEL_PATH = (
    r"C:\Users\User\PycharmProjects\AI_smart_traffic_system"
    r"\models\traffic_lstm.keras"
)

LSTM_SCALER_PATH = (
    r"C:\Users\User\PycharmProjects\AI_smart_traffic_system"
    r"\models\traffic_scaler.pkl"
)

VIDEO_PATH = (
    r"C:\Users\User\Videos\Screen Recordings\top-view2.mp4"
)


# ============================================================
# CONSTANTS
# ============================================================

ROUNDABOUT_CENTER = [7.3327756, -2.3278144]

LANE_IDS = [
    "Lane 1",
    "Lane 2",
    "Lane 3",
    "Lane 4"
]

FEATURE_COLUMNS = [
    "lane_1_avg_density",
    "lane_1_volume",
    "lane_2_avg_density",
    "lane_2_volume",
    "lane_3_avg_density",
    "lane_3_volume",
    "lane_4_avg_density",
    "lane_4_volume",
]


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_traffic_data():
    """
    Loads traffic data from CSV, converts timestamps,
    and sorts observations chronologically.
    """
    df = pd.read_csv(TRAFFIC_DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp", ascending=True).reset_index(drop=True)
    return df


def get_latest_traffic(df):
    """
    Returns the most recent traffic observation.
    """
    return df.iloc[-1]


# ============================================================
# TRAFFIC DENSITY
# ============================================================

def get_lane_densities(latest_traffic):
    """
    Extracts current density for each lane.
    """
    return {
        "Lane 1": float(latest_traffic["lane_1_avg_density"]),
        "Lane 2": float(latest_traffic["lane_2_avg_density"]),
        "Lane 3": float(latest_traffic["lane_3_avg_density"]),
        "Lane 4": float(latest_traffic["lane_4_avg_density"]),
    }


def get_lane_density_changes(latest_traffic, previous_traffic):
    """
    Calculates the change in traffic density
    between the latest and previous observation.
    """
    return {
        "Lane 1": round(latest_traffic["lane_1_avg_density"] - previous_traffic["lane_1_avg_density"], 2),
        "Lane 2": round(latest_traffic["lane_2_avg_density"] - previous_traffic["lane_2_avg_density"], 2),
        "Lane 3": round(latest_traffic["lane_3_avg_density"] - previous_traffic["lane_3_avg_density"], 2),
        "Lane 4": round(latest_traffic["lane_4_avg_density"] - previous_traffic["lane_4_avg_density"], 2),
    }


def format_density_change(change):
    """
    Formats density change with an explicit + or - sign.
    """
    return f"{change:+.2f}"


# ============================================================
# CONGESTION CLASSIFICATION
# ============================================================

def classify_congestion(density):
    """
    Converts traffic density into a congestion state.
    """
    if density < 10.0:
        return "Low"
    elif density < 15.0:
        return "Moderate"
    else:
        return "High Congestion"


def get_congestion_color(state):
    """
    Returns a color associated with congestion state.
    """
    if state == "Low":
        return "#28a745"
    elif state == "Moderate":
        return "#ffc107"
    elif state == "High Congestion":
        return "#dc3545"
    return "#6c757d"


# ============================================================
# LANE / MAP DATA
# ============================================================

def build_lanes_data(lane_densities):
    lanes = {
        "Lane 1": {
            "name": "Kumasi Road Approach",
            "coords": [7.3326701, -2.3276250],
            "corridor_polygon": [
                [7.33220, -2.32710],
                [7.33235, -2.32695],
                [7.33275, -2.32755],
                [7.33260, -2.32770],
            ],
            "direction": "Southeast → Roundabout",
            "density": lane_densities["Lane 1"],
            "state": classify_congestion(lane_densities["Lane 1"]),
        },
        "Lane 2": {
            "name": "Sunyani CBD / Township Approach",
            "coords": [7.3329172, -2.3276572],
            "corridor_polygon": [
                [7.33280, -2.32680],
                [7.33305, -2.32680],
                [7.33305, -2.32760],
                [7.33280, -2.32760],
            ],
            "direction": "East → Roundabout",
            "density": lane_densities["Lane 2"],
            "state": classify_congestion(lane_densities["Lane 2"]),
        },
        "Lane 3": {
            "name": "Berekum Road Approach",
            "coords": [7.3339371, -2.3291834],
            "corridor_polygon": [
                [7.33380, -2.32930],
                [7.33405, -2.32910],
                [7.33310, -2.32800],
                [7.33290, -2.32815],
            ],
            "direction": "Northwest → Roundabout",
            "density": lane_densities["Lane 3"],
            "state": classify_congestion(lane_densities["Lane 3"]),
        },
        "Lane 4": {
            "name": "Atronie Road Approach",
            "coords": [7.3325883, -2.3279298],
            "corridor_polygon": [
                [7.33220, -2.32830],
                [7.33235, -2.32845],
                [7.33270, -2.32800],
                [7.33250, -2.32785],
            ],
            "direction": "Southwest → Roundabout",
            "density": lane_densities["Lane 4"],
            "state": classify_congestion(lane_densities["Lane 4"]),
        },
    }

    for lane in lanes:
        lanes[lane]["color"] = get_congestion_color(lanes[lane]["state"])

    return lanes


# ============================================================
# LSTM MODEL
# ============================================================

@st.cache_resource
def load_lstm_model():
    return load_model(LSTM_MODEL_PATH)


@st.cache_resource
def load_lstm_scaler():
    return joblib.load(LSTM_SCALER_PATH)


def prepare_prediction_sequence(df, scaler):
    recent_data = df[FEATURE_COLUMNS].tail(10)
    scaled_data = scaler.transform(recent_data)
    return scaled_data.reshape(1, 10, 8)


def predict_next_traffic(sequence, model, scaler):
    prediction_scaled = model.predict(sequence, verbose=0)
    return scaler.inverse_transform(prediction_scaled)


def get_predicted_lane_densities(predicted_traffic):
    return {
        "Lane 1": float(predicted_traffic[0][0]),
        "Lane 2": float(predicted_traffic[0][2]),
        "Lane 3": float(predicted_traffic[0][4]),
        "Lane 4": float(predicted_traffic[0][6]),
    }


def get_predicted_congestions(predicted_lane_densities):
    return {
        lane: classify_congestion(density)
        for lane, density in predicted_lane_densities.items()
    }


# ============================================================
# ROUTE RECOMMENDATION
# ============================================================

def recommended_route(predicted_lane_densities):
    recommended_lane = min(predicted_lane_densities, key=predicted_lane_densities.get)
    recommended_density = predicted_lane_densities[recommended_lane]
    return recommended_lane, recommended_density


# ============================================================
# ADAPTIVE TRAFFIC SIGNAL CONTROLLER
# ============================================================

def calculate_3color_adaptive_signals(lane_densities, predicted_lane_densities):
    effective_densities = {
        lane: max(lane_densities[lane], predicted_lane_densities[lane])
        for lane in lane_densities
    }

    highest_lane = max(effective_densities, key=effective_densities.get)
    max_density = effective_densities[highest_lane]

    signals = {}

    if max_density < 10.0:
        for lane in lane_densities:
            signals[lane] = {
                "status": "🟢 GREEN (Normal Cycle)",
                "duration": "20 Seconds",
                "color": "green",
                "mode": "Normal Operation",
                "effective_density": effective_densities[lane],
            }
        return signals

    if 10.0 <= max_density < 15.0:
        green_time = 40
    else:
        green_time = 60

    for lane in lane_densities:
        if lane == highest_lane:
            signals[lane] = {
                "status": "🟢 GREEN (Priority Extended)",
                "duration": f"{green_time}s Green + 3s Yellow",
                "color": "green",
                "mode": "Adaptive Priority",
                "effective_density": effective_densities[lane],
            }
        else:
            signals[lane] = {
                "status": "🔴 RED (Waiting for Phase)",
                "duration": f"Next cycle after {green_time}s",
                "color": "red",
                "mode": "Normal Cycle",
                "effective_density": effective_densities[lane],
            }

    return signals


# ============================================================
# SIGNAL UI
# ============================================================

def render_glowing_signal(signal_text, duration, color="green"):
    color_map = {
        "green": ("#00FF66", "rgba(0, 255, 102, 0.6)"),
        "red": ("#FF3333", "rgba(255, 51, 51, 0.6)"),
        "yellow": ("#FFCC00", "rgba(255, 204, 0, 0.6)"),
    }

    hex_color, glow_color = color_map.get(color, ("#00FF66", "rgba(0, 255, 102, 0.6)"))

    html = f"""
    <style>
        .signal-card {{
            background-color: #121212;
            color: {hex_color};
            border: 2px solid {hex_color};
            padding: 14px;
            border-radius: 10px;
            font-weight: bold;
            text-align: center;
            box-shadow: 0 0 8px {glow_color}, 0 0 18px {glow_color};
            margin-top: 10px;
            margin-bottom: 10px;
        }}
        .signal-status {{
            font-size: 16px;
            margin-bottom: 6px;
        }}
        .signal-duration {{
            color: #CCCCCC;
            font-size: 13px;
            font-weight: 600;
        }}
    </style>

    <div class="signal-card">
        <div class="signal-status">{signal_text}</div>
        <div class="signal-duration">{duration}</div>
    </div>
    """
    st.html(html)


# ============================================================
# LOAD AND PROCESS TRAFFIC DATA
# ============================================================

traffic_df = load_traffic_data()
latest_traffic = get_latest_traffic(traffic_df)
previous_traffic = traffic_df.iloc[-2]

lane_densities = get_lane_densities(latest_traffic)
lane_changes = get_lane_density_changes(latest_traffic, previous_traffic)

# LOAD LSTM & GENERATE FORECAST
traffic_scaler = load_lstm_scaler()
lstm_model = load_lstm_model()

prediction_sequence = prepare_prediction_sequence(traffic_df, traffic_scaler)
predicted_traffic = predict_next_traffic(prediction_sequence, lstm_model, traffic_scaler)

predicted_lane_densities = get_predicted_lane_densities(predicted_traffic)
predicted_congestion = get_predicted_congestions(predicted_lane_densities)

# ROUTE & DECISIONS
recommended_lane, recommended_density = recommended_route(predicted_lane_densities)
LANES_DATA = build_lanes_data(lane_densities)
SYSTEM_SIGNALS = calculate_3color_adaptive_signals(lane_densities, predicted_lane_densities)

PREDICTION_DATA = {
    lane: {
        "density": predicted_lane_densities[lane],
        "state": predicted_congestion[lane],
    }
    for lane in LANE_IDS
}


# ============================================================
# HEADER & SYSTEM STATUS BADGE
# ============================================================

st.markdown(
    "<h1 style='text-align: center;'>AI SMART TRAFFIC CONTROLLER</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #00FF66; font-weight: bold; font-size: 16px; margin-top: -10px;'>"
    "🟢 SYSTEM LIVE & ONLINE | Node: Sunyani COCOBOD Intersection (30 FPS Inferred)"
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("---")


# ============================================================
# TOP-LEVEL KPI OVERVIEW BAR
# ============================================================

top_kpi1, top_kpi2, top_kpi3, top_kpi4 = st.columns(4)

highest_current_demand_lane = max(lane_densities, key=lane_densities.get)

with top_kpi1:
    st.metric(
        label="Peak Demand Lane",
        value=highest_current_demand_lane,
        delta=f"{lane_densities[highest_current_demand_lane]:.2f} density"
    )

with top_kpi2:
    priority_signal_lane = max(
        SYSTEM_SIGNALS,
        key=lambda l: SYSTEM_SIGNALS[l]["effective_density"]
    )
    st.metric(
        label="Signal Override Target",
        value=priority_signal_lane,
        delta=SYSTEM_SIGNALS[priority_signal_lane]["mode"]
    )

with top_kpi3:
    st.metric(
        label="Recommended Clear Corridor",
        value=recommended_lane,
        delta=f"{recommended_density:.2f} forecast density",
        delta_color="normal"
    )

with top_kpi4:
    st.metric(
        label="LSTM Forecast Horizon",
        value="~1 Minute",
        delta="10-step Sequence Active"
    )

st.markdown("---")


# ============================================================
# CAMERA + MAP (SIDE BY SIDE)
# ============================================================

col1, col2 = st.columns([1, 1])

# CAMERA FEED
with col1:
    st.subheader("📹 AI-Analyzed Traffic Camera Feed")
    st.video(VIDEO_PATH)

# INTERACTIVE MAP
with col2:
    st.subheader("🗺️ Interactive COCOBOD Intersection Map")

    map_objects = folium.Map(
        location=ROUNDABOUT_CENTER,
        zoom_start=18,
        tiles="OpenStreetMap"
    )

    # Roundabout center marker
    folium.Marker(
        location=ROUNDABOUT_CENTER,
        popup="<b>Sunyani COCOBOD Roundabout</b><br>Intersection Control Center",
        tooltip="COCOBOD Roundabout Center",
        icon=folium.Icon(color="black", icon="info-sign")
    ).add_to(map_objects)

    # LANE POLYGONS & MARKERS
    for lane_id, data in LANES_DATA.items():
        is_recommended = (lane_id == recommended_lane)

        if is_recommended:
            border_color = "#007BFF"
            border_weight = 5
            route_label = "★ RECOMMENDED ROUTE"
        else:
            border_color = data["color"]
            border_weight = 2
            route_label = data["state"]

        signal_info = SYSTEM_SIGNALS[lane_id]
        predicted_density = predicted_lane_densities[lane_id]
        predicted_state = predicted_congestion[lane_id]

        popup_html = f"""
        <div style="width: 260px; font-family: Arial;">
            <h4 style="margin-bottom: 8px;">{lane_id}</h4>
            <b>{data['name']}</b>
            <hr>
            <b>Current Traffic</b><br>
            Density: {data['density']:.2f}<br>
            State: {data['state']}<br><br>
            <b>LSTM Forecast</b><br>
            Next ~1 minute: {predicted_density:.2f}<br>
            Predicted State: {predicted_state}<br><br>
            <b>Adaptive Signal</b><br>
            {signal_info['status']}<br>
            Duration: {signal_info['duration']}<br>
            Mode: {signal_info['mode']}<br><br>
            <b>Direction</b><br>
            {data['direction']}<br><br>
            <b>{route_label}</b>
        </div>
        """

        folium.Polygon(
            locations=data["corridor_polygon"],
            color=border_color,
            weight=border_weight,
            fill=True,
            fill_color=data["color"],
            fill_opacity=0.45,
            tooltip=f"{lane_id} | {data['name']} | {route_label}",
            popup=folium.Popup(popup_html, max_width=320)
        ).add_to(map_objects)

        folium.Marker(
            location=data["coords"],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{lane_id} - {data['name']}",
            icon=folium.Icon(
                color="blue" if is_recommended else "gray",
                icon="car",
                prefix="fa"
            ),
        ).add_to(map_objects)

    # Rendered map cleanly without forcing reloads
    st_folium(map_objects, width=650, height=360)


# ============================================================
# CURRENT TRAFFIC ANALYTICS
# ============================================================

st.markdown("---")
st.markdown("<h2 style='text-align: center;'>Real-Time Lane Analytics</h2>", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

for column, lane in zip([m1, m2, m3, m4], LANE_IDS):
    with column:
        st.metric(
            label=f"{lane} ({LANES_DATA[lane]['name']})",
            delta=format_density_change(lane_changes[lane]),
            value=f"{lane_densities[lane]:.2f}",
            delta_color="inverse",
        )
        st.caption(f"Status: {LANES_DATA[lane]['state']}")


# ============================================================
# LSTM FORECAST + ADAPTIVE SIGNAL CONTROLLER
# ============================================================

st.markdown("---")
st.markdown(
    "<h2 style='text-align: center;'>Next ~1 Minute LSTM Forecast & Adaptive Signal Controller</h2>",
    unsafe_allow_html=True,
)
st.caption("The LSTM model evaluates the previous 10 historical time steps to dynamically assign signal priorities.")

prediction_cols = st.columns(4)

for idx, lane in enumerate(LANE_IDS):
    with prediction_cols[idx]:
        st.metric(
            label=f"{lane} FORECAST",
            value=f"{PREDICTION_DATA[lane]['density']:.2f}"
        )
        st.caption(f"Predicted State: {PREDICTION_DATA[lane]['state']}")

        signal_info = SYSTEM_SIGNALS[lane]
        render_glowing_signal(
            signal_info["status"],
            signal_info["duration"],
            color=signal_info["color"]
        )


# ============================================================
# ROUTE RECOMMENDATION
# ============================================================

st.markdown("---")
st.subheader("Dynamic Route Recommendation Engine")

recommended_name = LANES_DATA[recommended_lane]["name"]
recommended_direction = LANES_DATA[recommended_lane]["direction"]

st.success(f"Optimal Corridor Selected: **{recommended_lane}** ({recommended_name})")

res_col1, res_col2 = st.columns(2)
with res_col1:
    st.write(f"**Travel Direction:** {recommended_direction}")
with res_col2:
    st.write(f"**Predicted Minimum Density:** {recommended_density:.2f}")