import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance
import shap
import joblib
import os

# Set up
st.set_page_config(page_title="Model Training + Explainability", layout="wide")
st.title("💡 Model Training & Explainability")

@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df['settlement_date'] = pd.to_datetime(df['settlement_date'])
    df = df[df["england_wales_demand"] > 100]
    df.fillna(method="ffill", inplace=True)
    df['year'] = df['settlement_date'].dt.year
    df['month'] = df['settlement_date'].dt.month
    df['day'] = df['settlement_date'].dt.day
    df['weekday'] = df['settlement_date'].dt.weekday
    df['hour'] = df['settlement_date'].dt.hour
    df["demand_lag_1h"] = df["england_wales_demand"].shift(1)
    df["demand_lag_24h"] = df["england_wales_demand"].shift(24)
    df["demand_rolling_24h"] = df["england_wales_demand"].rolling(24).mean()
    df.dropna(inplace=True)
    df.drop(columns=["settlement_date"], inplace=True)
    return df

df = load_data()
target_col = "england_wales_demand"
feature_cols = [
    'year', 'month', 'day', 'weekday', 'hour',
    'embedded_wind_capacity', 'embedded_wind_generation',
    'demand_lag_1h', 'demand_lag_24h', 'demand_rolling_24h'
]
X = df[feature_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar: Hyperparameter tuning
st.sidebar.header("Hyperparameters")
n_estimators = st.sidebar.slider("n_estimators", 100, 1000, 300, step=50)
max_depth = st.sidebar.slider("max_depth", 2, 12, 6)
learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.05, step=0.01)
subsample = st.sidebar.slider("subsample", 0.5, 1.0, 0.8, step=0.05)
colsample_bytree = st.sidebar.slider("colsample_bytree", 0.5, 1.0, 0.8, step=0.05)

# Train model
model = XGBRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Actual vs Predicted")
    st.line_chart(pd.DataFrame({"Actual": y_test.values[:100], "Predicted": y_pred[:100]}))
     #  Feature Importance - XGBoost Built-In
    st.subheader(" XGBoost Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_importance(model, ax=ax, importance_type='gain')
    st.pyplot(fig)
    
with col2:
    st.markdown(f"""
    <div style='background-color:#f0f8ff;padding:90px;border-radius:10px;font-size:40px'>
        <b>Mean Squared Error:</b> {mean_squared_error(y_test, y_pred):.2f} MW<br/>
        <b>R² Score:</b> {r2_score(y_test, y_pred):.4f}
    </div>
    """, unsafe_allow_html=True)
   
    #  SHAP Explainability
    st.subheader(" SHAP Explainability (Top Features)")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test.sample(300, random_state=42))
    shap.summary_plot(shap_values, X_test.sample(300, random_state=42), show=False)
    fig2 = plt.gcf()
    st.pyplot(fig2)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/demand_predictor.pkl")


# 🌟 SHAP Force Plot for Individual Prediction
st.subheader("SHAP Force Plot (Single Prediction)")

# Choose a specific row
index_to_explain = st.number_input(
    "Select Row Index to Explain (from the test set):",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0,
    step=1,
    help="Pick an index to visualize SHAP contributions for an individual prediction."
)

# Extract the specific row
X_instance = X_test.iloc[[index_to_explain]]

# Compute SHAP values for this instance
shap_values_single = explainer(X_instance)

# Display prediction details
pred_value = model.predict(X_instance)[0]
true_value = y_test.iloc[index_to_explain]

st.markdown(
    f"""
    <div style='background-color:#eef;padding:10px;border-radius:6px;'>
        <b>Prediction:</b> {pred_value:.2f} MW<br/>
        <b>True Value:</b> {true_value:.2f} MW
    </div>
    """,
    unsafe_allow_html=True
)

# Generate force plot
#st.set_option('deprecation.showPyplotGlobalUse', False)
shap.plots.force(shap_values_single[0], matplotlib=True)
fig_force = plt.gcf()
st.pyplot(fig_force)
