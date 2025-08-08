import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="Energy Demand Predictor", layout="wide")
st.title("Energy Demand Predictor")

# Load model
model_path = "models/demand_predictor.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

model = load_model()

if model is None:
    st.error("No trained model found. Please train a model first.")
    st.stop()

# Load historical dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df.sort_values("settlement_date", inplace=True)
    df.fillna(method="ffill", inplace=True)

    # Feature engineering
    df["year"] = df["settlement_date"].dt.year
    df["month"] = df["settlement_date"].dt.month
    df["day"] = df["settlement_date"].dt.day
    df["weekday"] = df["settlement_date"].dt.weekday
    df["hour"] = df["settlement_date"].dt.hour

    df["demand_lag_1h"] = df["england_wales_demand"].shift(1)
    df["demand_lag_24h"] = df["england_wales_demand"].shift(24)
    df["demand_rolling_24h"] = df["england_wales_demand"].rolling(24).mean()

    df.dropna(inplace=True)
    return df

df = load_data()
cola, colb = st.columns(2)

with cola:
    st.subheader("Enter Input Features")
    with st.form("prediction_form"):
        
        col1, col2 = st.columns(2)
        with col1:
            year = st.number_input("Year", min_value=2000, max_value=2100, value=datetime.now().year)
            day = st.number_input("Day", min_value=1, max_value=31, value=datetime.now().day)
            hour = st.slider("Hour (0–23)", min_value=0, max_value=23, value=datetime.now().hour)
            embedded_wind_capacity = st.number_input("Embedded Wind Capacity (MW)", min_value=0.0, value=1200.0)
        with col2:
            month = st.number_input("Month", min_value=1, max_value=12, value=datetime.now().month)
            weekday = st.selectbox("Weekday (0 = Monday, 6 = Sunday)", list(range(7)))
            embedded_wind_generation = st.number_input("Embedded Wind Generation (MW)", min_value=0.0, value=600.0)

        submitted = st.form_submit_button("Predict Demand")
        
with colb:
    if submitted:
        input_datetime = pd.Timestamp(datetime(year, month, day, hour))

        # Load model features
        model_features = model.get_booster().feature_names if hasattr(model, "get_booster") else model.feature_names_in_

        input_data = {
            "year": year,
            "month": month,
            "day": day,
            "weekday": weekday,
            "hour": hour,
            "embedded_wind_capacity": embedded_wind_capacity,
            "embedded_wind_generation": embedded_wind_generation
        }

        row = df[df["settlement_date"] == input_datetime]

        if not row.empty:
            st.info("Using exact match from historical data for lag and flow values.")
            for feat in model_features:
                if feat not in input_data and feat in row.columns:
                    input_data[feat] = row.iloc[0][feat]
        else:
            # Get last known row for lag and flow values
            latest_row = df[df["settlement_date"] < input_datetime].iloc[-1] if not df[df["settlement_date"] < input_datetime].empty else None

            for feat in model_features:
                if feat in input_data:
                    continue  # already filled by user

                if latest_row is not None and feat in latest_row:
                    input_data[feat] = latest_row[feat]
                else:
                    # Default fallback values
                    if "lag" in feat or "rolling" in feat:
                        input_data[feat] = df["england_wales_demand"].mean()
                    else:
                        input_data[feat] = 0.0  # for flows or other numerical features

        # Final input
        input_df = pd.DataFrame([input_data])[model_features]

        prediction = model.predict(input_df)[0]
                
        with st.container():
            st.markdown(
            f"""
            <div style='
                margin-top: 60px;
                background-color: #f0f8ff; 
                padding: 20px; 
                border-radius: 10px; 
                border: 1px solid #cce; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                font-size: 30px;
                font-weight: bold;
                color: #003366;
            '>
                Predicted Energy Demand: {prediction:.2f} MW
            </div>
            """,
            unsafe_allow_html=True
    )

    # with st.expander("Feature Inputs Used"):
    #    st.dataframe(input_df.T.rename(columns={0: "Value"}))
