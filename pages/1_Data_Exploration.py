import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Model Training", layout="wide")
st.title("Data exploration and Feature engineering")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df['settlement_date'] = pd.to_datetime(df['settlement_date'])
    df.fillna(method="ffill", inplace=True)
    return df

df = load_data()
col1, col2 = st.columns(2)

with col1:
    # Distribution BEFORE
    st.subheader("Raw Data")
    fig, ax = plt.subplots(figsize=(8, 2))
    sns.boxplot(x=df["england_wales_demand"], ax=ax)
    st.pyplot(fig)
    #st.write("Min:", df["england_wales_demand"].min())
    #st.write("Max:", df["england_wales_demand"].max())

    # Remove zero or unrealistic demand
    df = df[df["england_wales_demand"] > 100]

    # Remove extreme outliers using IQR
    Q1 = df["england_wales_demand"].quantile(0.25)
    Q3 = df["england_wales_demand"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df["england_wales_demand"] >= lower_bound) & (df["england_wales_demand"] <= upper_bound)]

    # Distribution AFTER
    #st.subheader("Target Distribution (After Cleaning)")
    #fig2, ax2 = plt.subplots(figsize=(8, 2))
    #sns.boxplot(x=df["england_wales_demand"], ax=ax2)
    #st.pyplot(fig2)

    # Required columns
    required_cols = ['embedded_wind_capacity', 'embedded_wind_generation']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f" Missing required columns: {', '.join(missing)}")
        st.stop()

    # Feature engineering
    df = df.sort_values(by="settlement_date")
    df['year'] = df['settlement_date'].dt.year
    df['month'] = df['settlement_date'].dt.month
    df['day'] = df['settlement_date'].dt.day
    df['weekday'] = df['settlement_date'].dt.weekday
    df['hour'] = df['settlement_date'].dt.hour

    # Optional time-based lag features
    df["demand_lag_1h"] = df["england_wales_demand"].shift(1)
    df["demand_lag_24h"] = df["england_wales_demand"].shift(24)
    df["demand_rolling_24h"] = df["england_wales_demand"].rolling(24).mean()

    # Clean up
    df.dropna(inplace=True)
    df.drop(columns=["settlement_date"], inplace=True)

    # Define features and target
    target_col = "england_wales_demand"
    feature_cols = [
        'year', 'month', 'day', 'weekday', 'hour',
        'embedded_wind_capacity', 'embedded_wind_generation',
        'demand_lag_1h', 'demand_lag_24h', 'demand_rolling_24h'
    ]

    # Optional: add other available useful columns if needed
    available_extra = [
        'ifa_flow', 'nemo_flow', 'ifa2_flow', 'britned_flow', 'moyle_flow',
        'east_west_flow', 'scottish_transfer', 'eleclink_flow'
    ]

    # Correlation Matrix
    st.subheader("📊 Correlation Matrix")

    corr_features = feature_cols + [target_col]
    corr_matrix = df[corr_features].corr()

    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5, ax=ax_corr)
    ax_corr.set_title("Correlation Matrix")
    st.pyplot(fig_corr)

    feature_cols += [col for col in available_extra if col in df.columns]

    X = df[feature_cols]
    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with tuned parameters
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#st.success("✅ Model Trained!")

with col2:
    # Sample Predictions
    st.subheader("Actual vs Predicted")
    st.line_chart(pd.DataFrame({
        "Actual": y_test.values[:100],
        "Predicted": y_pred[:100]
    }))
    
    with st.container():
            st.markdown(
            f"""
            <div style='
                margin-top: 60px;
                margin-left: 60px;
                margin-right: 60px;
                background-color: #f0f8ff; 
                padding: 20px; 
                border-radius: 10px; 
                border: 1px solid #cce; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                font-size: 30px;
                font-weight: bold;
                color: #003366;
            '>
                Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f} MW
               <br/> R² Score: {r2_score(y_test, y_pred):.4f}
            </div>
            """,
            unsafe_allow_html=True
    )
    
# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/demand_predictor.pkl")
#st.success(" Model saved to `models/demand_predictor.pkl`")
