import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.optimizers import Adam
from tcn import TCN
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# Load and preprocess data
df = pd.read_csv("dataset.csv")
df['settlement_date'] = pd.to_datetime(df['settlement_date'])
df = df[df["england_wales_demand"] > 100]
df.fillna(method="ffill", inplace=True)

Q1 = df["england_wales_demand"].quantile(0.25)
Q3 = df["england_wales_demand"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df["england_wales_demand"] >= lower_bound) & (df["england_wales_demand"] <= upper_bound)]

df['year'] = df['settlement_date'].dt.year
df['month'] = df['settlement_date'].dt.month
df['day'] = df['settlement_date'].dt.day
df['weekday'] = df['settlement_date'].dt.weekday
df['hour'] = df['settlement_date'].dt.hour
df["demand_lag_1h"] = df["england_wales_demand"].shift(1)
df["demand_lag_24h"] = df["england_wales_demand"].shift(24)
df["demand_rolling_24h"] = df["england_wales_demand"].rolling(24).mean()
df.dropna(inplace=True)
df.set_index("settlement_date", inplace=True)

# Features and target
target_col = "england_wales_demand"
feature_cols = [
    'year', 'month', 'day', 'weekday', 'hour',
    'embedded_wind_capacity', 'embedded_wind_generation',
    'demand_lag_1h', 'demand_lag_24h', 'demand_rolling_24h'
]
X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}

# ✅ Model: XGBoost
model = XGBRegressor()
model.fit(X_train, y_train)
pred = model.predict(X_test)
results["XGBoost"] = (mean_squared_error(y_test, pred), r2_score(y_test, pred))

# ✅ Model: Random Forest
model = RandomForestRegressor()
model.fit(X_train, y_train)
pred = model.predict(X_test)
results["Random Forest"] = (mean_squared_error(y_test, pred), r2_score(y_test, pred))

# ✅ Model: LightGBM
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)
pred = model.predict(X_test)
results["LightGBM"] = (mean_squared_error(y_test, pred), r2_score(y_test, pred))

# ✅ Model: CatBoost
model = CatBoostRegressor(verbose=0)
model.fit(X_train, y_train)
pred = model.predict(X_test)
results["CatBoost"] = (mean_squared_error(y_test, pred), r2_score(y_test, pred))



# ============================
# Deep Learning Models (LSTM, GRU, TCN)
# ============================

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

def create_sequences(X, y, seq_len=24):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X.iloc[i:i+seq_len].values)
        ys.append(y.iloc[i+seq_len])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y, 24)
split = int(0.8 * len(X_seq))
X_train_dl, X_test_dl = X_seq[:split], X_seq[split:]
y_train_dl, y_test_dl = y_seq[:split], y_seq[split:]

def train_rnn(cell="LSTM"):
    model = Sequential()
    if cell == "LSTM":
        model.add(LSTM(64, input_shape=X_train_dl.shape[1:]))
    elif cell == "GRU":
        model.add(GRU(64, input_shape=X_train_dl.shape[1:]))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer=Adam(0.001))
    model.fit(X_train_dl, y_train_dl, epochs=5, batch_size=32, verbose=0)
    pred = model.predict(X_test_dl).flatten()
    return mean_squared_error(y_test_dl, pred), r2_score(y_test_dl, pred)

def train_tcn():
    model = Sequential()
    model.add(TCN(input_shape=X_train_dl.shape[1:]))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer=Adam(0.001))
    model.fit(X_train_dl, y_train_dl, epochs=5, batch_size=32, verbose=0)
    pred = model.predict(X_test_dl).flatten()
    return mean_squared_error(y_test_dl, pred), r2_score(y_test_dl, pred)

results["LSTM"] = train_rnn("LSTM")
results["GRU"] = train_rnn("GRU")
results["TCN"] = train_tcn()

# ============================
# Final Output
# ============================

print("\n🔍 Model Performance (MSE, R2):\n")
for model_name, (mse, r2) in results.items():
    print(f"{model_name:<15} | MSE: {mse:.2f} | R²: {r2:.4f}")
    df_results = pd.DataFrame([
    {"Model": model_name, "MSE": mse, "R²": r2}
    for model_name, (mse, r2) in results.items()
])

st.subheader("🔍 Model Performance (MSE, R²):")
st.table(df_results)
