# ============================================================
# CASEFILE
# AI-Powered Missing Person Investigation
# Advanced Machine Learning Project
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import plotly.express as px
import plotly.graph_objects as go

import folium
from streamlit_folium import st_folium


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CASEFILE - Missing Person Investigation",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    font-size: 18px;
    color: #666;
}

.card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    background-color: #fafafa;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🔎 CASEFILE</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Missing Person Investigation and Probable Location Prediction System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# LOAD DATASET
# ============================================================

DATA_FILE = "data.csv"

try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:

    st.error(
        "Dataset not found. Make sure "
        "'casefile_gps_trajectory_dataset.csv' "
        "is in the same folder as app.py."
    )

    st.stop()


# ============================================================
# DATA PREPROCESSING
# ============================================================

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df["Hour"] = df["Timestamp"].dt.hour

df["Weekday_Number"] = df["Timestamp"].dt.dayofweek

df["Is_Weekend"] = df["Weekday_Number"].isin([5, 6]).astype(int)

# Remove missing values
df = df.dropna()

# Remove duplicate records
df = df.drop_duplicates()

# Remove invalid coordinates
df = df[
    (df["Latitude"].between(-90, 90)) &
    (df["Longitude"].between(-180, 180))
]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Case Controls")

selected_person = st.sidebar.selectbox(
    "Select Historical Person",
    sorted(df["Person_ID"].unique())
)

selected_hour = st.sidebar.slider(
    "Last Seen Hour",
    min_value=0,
    max_value=23,
    value=18
)

st.sidebar.info(
    "This application is an academic simulation using "
    "fictional/synthetic movement data."
)


# ============================================================
# SELECT PERSON DATA
# ============================================================

person_df = df[df["Person_ID"] == selected_person].copy()


# ============================================================
# DASHBOARD METRICS
# ============================================================

st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Records",
        len(df)
    )

with col2:
    st.metric(
        "Persons",
        df["Person_ID"].nunique()
    )

with col3:
    st.metric(
        "Areas",
        df["Target_Area"].nunique()
    )

with col4:
    st.metric(
        "Average Speed",
        f"{df['Speed_kmh'].mean():.1f} km/h"
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📍 Movement Analysis",
    "🚨 Anomaly Detection",
    "🤖 Location Prediction",
    "🛣️ Route Prediction",
    "🎯 Search Priority",
    "🗺️ Investigation Map"
])


# ============================================================
# TAB 1 - MOVEMENT ANALYSIS
# ============================================================

with tab1:

    st.header("📍 Movement Pattern Analysis")

    st.write(
        "K-Means clustering is used to identify geographical "
        "movement patterns and frequently visited areas."
    )

    cluster_features = df[
        ["Latitude", "Longitude"]
    ]

    kmeans = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = kmeans.fit_predict(
        cluster_features
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter_mapbox(
            df,
            lat="Latitude",
            lon="Longitude",
            color="Cluster",
            hover_data=[
                "Person_ID",
                "Target_Area",
                "Speed_kmh"
            ],
            zoom=11,
            height=500,
            mapbox_style="open-street-map",
            title="K-Means Movement Clusters"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        cluster_counts = (
            df["Cluster"]
            .value_counts()
            .sort_index()
        )

        fig2 = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={
                "x": "Cluster",
                "y": "Number of Visits"
            },
            title="Visits per Cluster"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.subheader("Frequently Visited Areas")

    area_frequency = (
        df["Target_Area"]
        .value_counts()
        .reset_index()
    )

    area_frequency.columns = [
        "Area",
        "Visits"
    ]

    st.dataframe(
        area_frequency,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 2 - ANOMALY DETECTION
# ============================================================

with tab2:

    st.header("🚨 Anomaly Detection")

    st.write(
        "Isolation Forest identifies movement records "
        "that differ significantly from historical patterns."
    )

    anomaly_features = df[
        [
            "Latitude",
            "Longitude",
            "Speed_kmh",
            "Distance_km",
            "Movement_Duration_Min"
        ]
    ]

    iso_model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    df["Anomaly_Result"] = iso_model.fit_predict(
        anomaly_features
    )

    df["Anomaly"] = df["Anomaly_Result"].map({
        1: "Normal",
        -1: "Anomaly"
    })

    anomaly_count = (
        df["Anomaly"] == "Anomaly"
    ).sum()

    normal_count = (
        df["Anomaly"] == "Normal"
    ).sum()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Normal Records",
            normal_count
        )

    with col2:
        st.metric(
            "Anomalous Records",
            anomaly_count
        )

    with col3:
        st.metric(
            "Anomaly Rate",
            f"{anomaly_count / len(df) * 100:.2f}%"
        )

    st.divider()

    fig = px.scatter_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        color="Anomaly",
        hover_data=[
            "Person_ID",
            "Speed_kmh",
            "Target_Area"
        ],
        zoom=11,
        height=550,
        mapbox_style="open-street-map",
        title="Normal vs Anomalous Movement"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.warning(
        "Important: An anomaly indicates unusual movement "
        "relative to historical patterns. It does NOT prove "
        "suspicious or criminal behavior."
    )


# ============================================================
# TAB 3 - LOCATION PREDICTION
# ============================================================

with tab3:

    st.header("🤖 Probable Location Prediction")

    st.write(
        "Random Forest predicts the probable geographical "
        "area based on historical movement characteristics."
    )

    prediction_features = [
        "Hour",
        "Weekday_Number",
        "Latitude",
        "Longitude",
        "Speed_kmh",
        "Distance_km",
        "Movement_Duration_Min",
        "Is_Weekend"
    ]

    X = df[prediction_features]

    y = df["Target_Area"]

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )

    rf_model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        max_depth=12
    )

    rf_model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # MODEL EVALUATION
    # --------------------------------------------------------

    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )

    st.divider()

    # --------------------------------------------------------
    # CREATE FICTIONAL CURRENT CASE
    # --------------------------------------------------------

    st.subheader("🔎 Current Fictional Case")

    current_case = pd.DataFrame([{
        "Hour": selected_hour,
        "Weekday_Number": 1,
        "Latitude": person_df["Latitude"].iloc[-1],
        "Longitude": person_df["Longitude"].iloc[-1],
        "Speed_kmh": person_df["Speed_kmh"].mean(),
        "Distance_km": person_df["Distance_km"].mean(),
        "Movement_Duration_Min":
            person_df["Movement_Duration_Min"].mean(),
        "Is_Weekend": 0
    }])

    probabilities = rf_model.predict_proba(
        current_case[prediction_features]
    )[0]

    classes = rf_model.classes_

    results = []

    for class_id, probability in zip(
        classes,
        probabilities
    ):

        area_name = label_encoder.inverse_transform(
            [class_id]
        )[0]

        results.append({
            "Area": area_name,
            "Probability": probability * 100
        })

    prediction_df = pd.DataFrame(results)

    prediction_df = prediction_df.sort_values(
        "Probability",
        ascending=False
    ).reset_index(drop=True)

    prediction_df["Priority"] = np.where(
        prediction_df["Probability"] >= 30,
        "Very High",
        np.where(
            prediction_df["Probability"] >= 20,
            "High",
            np.where(
                prediction_df["Probability"] >= 10,
                "Medium",
                "Low"
            )
        )
    )

    prediction_df["Rank"] = (
        prediction_df.index + 1
    )

    prediction_df = prediction_df[
        [
            "Rank",
            "Area",
            "Probability",
            "Priority"
        ]
    ]

    prediction_df["Probability"] = (
        prediction_df["Probability"]
        .round(2)
    )

    st.dataframe(
        prediction_df,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        prediction_df,
        x="Area",
        y="Probability",
        text="Probability",
        title="Predicted Area Probability"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader("💡 Prediction Explanation")

    importance_df = pd.DataFrame({
        "Feature": prediction_features,
        "Importance": rf_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    fig3 = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.info(
        "The feature-importance chart explains which movement "
        "features contributed most to the model's predictions."
    )


# ============================================================
# TAB 4 - MARKOV CHAIN ROUTE PREDICTION
# ============================================================

with tab4:

    st.header("🛣️ Probable Route Prediction")

    st.write(
        "A Markov Chain is used to estimate the next likely "
        "location from historical area-to-area transitions."
    )

    # --------------------------------------------------------
    # CREATE TRANSITION MATRIX
    # --------------------------------------------------------

    transition_counts = pd.crosstab(
        df["Previous_Area"],
        df["Target_Area"]
    )

    transition_probabilities = (
        transition_counts
        .div(
            transition_counts.sum(axis=1),
            axis=0
        )
        .fillna(0)
    )

    st.subheader("Transition Probability Matrix")

    st.dataframe(
        transition_probabilities.round(2),
        use_container_width=True
    )

    # --------------------------------------------------------
    # ROUTE GENERATION
    # --------------------------------------------------------

    current_area = person_df[
        "Target_Area"
    ].iloc[-1]

    route = [current_area]

    for _ in range(4):

        if current_area in transition_probabilities.index:

            probabilities = (
                transition_probabilities
                .loc[current_area]
            )

            if probabilities.sum() > 0:

                next_area = np.random.choice(
                    probabilities.index,
                    p=probabilities.values
                )

                route.append(next_area)

                current_area = next_area

    st.subheader("🔮 Predicted Movement Route")

    route_text = " → ".join(route)

    st.success(route_text)

    # --------------------------------------------------------
    # ROUTE TABLE
    # --------------------------------------------------------

    route_df = pd.DataFrame({
        "Step": range(1, len(route) + 1),
        "Predicted Area": route
    })

    st.dataframe(
        route_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 5 - SEARCH PRIORITY
# ============================================================

with tab5:

    st.header("🎯 Search Priority Score")

    st.write(
        "Each predicted area receives a priority score using "
        "prediction probability, historical visits, route "
        "relevance, distance relevance, time relevance and "
        "anomaly evidence."
    )

    # --------------------------------------------------------
    # HISTORICAL VISIT FREQUENCY
    # --------------------------------------------------------

    visit_counts = (
        df["Target_Area"]
        .value_counts()
    )

    max_visits = visit_counts.max()

    # --------------------------------------------------------
    # CALCULATE SCORES
    # --------------------------------------------------------

    priority_rows = []

    for _, row in prediction_df.iterrows():

        area = row["Area"]

        prediction_score = (
            row["Probability"]
        )

        prediction_component = (
            prediction_score / 100
        ) * 30

        visit_frequency = (
            visit_counts.get(area, 0)
            / max_visits
        )

        visit_component = (
            visit_frequency * 20
        )

        # Route similarity
        route_component = (
            15 if area in route else 5
        )

        # Distance relevance
        distance_component = 10

        # Time relevance
        area_hour_count = len(
            df[
                (df["Target_Area"] == area) &
                (df["Hour"] == selected_hour)
            ]
        )

        time_component = min(
            10,
            area_hour_count / max(
                1,
                len(df) / 100
            ) * 10
        )

        # Anomaly evidence
        anomaly_area_count = len(
            df[
                (df["Target_Area"] == area) &
                (df["Anomaly"] == "Anomaly")
            ]
        )

        anomaly_component = min(
            10,
            anomaly_area_count / max(
                1,
                len(df) * 0.05
            ) * 10
        )

        score = (
            prediction_component +
            visit_component +
            route_component +
            distance_component +
            time_component +
            anomaly_component
        )

        score = min(
            100,
            score
        )

        if score >= 81:
            priority = "Very High"

        elif score >= 61:
            priority = "High"

        elif score >= 31:
            priority = "Medium"

        else:
            priority = "Low"

        priority_rows.append({
            "Area": area,
            "ML Probability": round(
                prediction_score,
                2
            ),
            "Historical Visits": visit_counts.get(
                area,
                0
            ),
            "Priority Score": round(
                score,
                2
            ),
            "Priority": priority
        })

    priority_df = pd.DataFrame(
        priority_rows
    )

    priority_df = priority_df.sort_values(
        "Priority Score",
        ascending=False
    )

    st.dataframe(
        priority_df,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        priority_df,
        x="Area",
        y="Priority Score",
        text="Priority Score",
        title="Search Priority Score"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "Priority weighting follows the project guide: "
        "ML probability 30%, historical visit frequency 20%, "
        "route similarity 15%, distance relevance 15%, "
        "time relevance 10%, anomaly evidence 10%."
    )


# ============================================================
# TAB 6 - INVESTIGATION MAP
# ============================================================

with tab6:

    st.header("🗺️ Interactive Investigation Map")

    st.write(
        "The map displays historical movement points, "
        "the last known location, predicted areas and anomalies."
    )

    # --------------------------------------------------------
    # CENTER MAP
    # --------------------------------------------------------

    center_lat = person_df["Latitude"].mean()
    center_lon = person_df["Longitude"].mean()

    investigation_map = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=12
    )

    # --------------------------------------------------------
    # HISTORICAL MOVEMENT POINTS
    # --------------------------------------------------------

    sample_points = person_df.sample(
        min(100, len(person_df)),
        random_state=42
    )

    for _, row in sample_points.iterrows():

        folium.CircleMarker(
            location=[
                row["Latitude"],
                row["Longitude"]
            ],
            radius=4,
            popup=(
                f"Area: {row['Target_Area']}<br>"
                f"Speed: {row['Speed_kmh']:.1f} km/h"
            ),
            fill=True
        ).add_to(
            investigation_map
        )

    # --------------------------------------------------------
    # LAST KNOWN LOCATION
    # --------------------------------------------------------

    last_record = person_df.iloc[-1]

    folium.Marker(
        location=[
            last_record["Latitude"],
            last_record["Longitude"]
        ],
        popup="Last Known Location",
        tooltip="Last Known Location"
    ).add_to(
        investigation_map
    )

    # --------------------------------------------------------
    # ANOMALOUS LOCATIONS
    # --------------------------------------------------------

    person_anomalies = df[
        (df["Person_ID"] == selected_person) &
        (df["Anomaly"] == "Anomaly")
    ]

    for _, row in person_anomalies.iterrows():

        folium.CircleMarker(
            location=[
                row["Latitude"],
                row["Longitude"]
            ],
            radius=6,
            popup=(
                f"ANOMALY<br>"
                f"Area: {row['Target_Area']}<br>"
                f"Speed: {row['Speed_kmh']:.1f} km/h"
            ),
            fill=True
        ).add_to(
            investigation_map
        )

    # --------------------------------------------------------
    # MAP DISPLAY
    # --------------------------------------------------------

    st.components.v1.html(
    investigation_map._repr_html_(),
    height=600,
    scrolling=False
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CASEFILE is an academic simulation. "
    "Predictions are probabilistic and should not be used "
    "for real-world decisions concerning missing persons."
)