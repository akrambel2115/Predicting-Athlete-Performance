#!/usr/bin/env python3
# train_delta_p_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ─── 1) LOAD & PREPARE DATA ──────────────────────────────────────────────────
df = pd.read_csv("../data/soccerData/soccer_data_cleaned.csv", parse_dates=["date"])
df = df.sort_values(["player_id", "date"]).reset_index(drop=True)

# map action → intensity
action_intensity_map = {
    "Train_Low":  0.3,
    "Train_Med":  0.6,
    "Train_High": 0.9
}
df["action_intensity"] = df["action"].map(action_intensity_map).fillna(0.0)

# flag game days where performance_metric is non-null
df = df[df["performance_metric"].notna()]

# ─── 2) COMPUTE TARGET: ΔP ─────────────────────────────────────────────────────
# lag performance within each player
df["performance_lag"] = (
    df.groupby("player_id")["performance_metric"]
      .shift(1)
)
# drop first game-day per player (no lag)
df["delta_p"] = df["performance_metric"] - df["performance_lag"]
df = df.dropna(subset=["delta_p"])  # only rows with a valid delta

# ─── 3) DEFINE FEATURES ──────────────────────────────────────────────────────
FEATURES_P = [
    "load",
    "action_intensity",
    "fatigue_post",
    "sleep_duration",
    "sleep_quality",
    "stress",
    "is_rest_day",
    "load_rolling_7",
    "fatigue_post_rolling_7",
    "sleep_duration_rolling_7",
    "sleep_quality_rolling_7",
    "stress_rolling_7",
    "total_duration",
    "load_lag_1"
]

# ensure all those exist
missing = set(FEATURES_P) - set(df.columns)
if missing:
    raise KeyError(f"Missing columns in data for ΔP features: {missing}")

# drop any rows with NaN in our chosen features
df_p = df.dropna(subset=FEATURES_P)

X = df_p[FEATURES_P]
y = df_p["delta_p"]

# ─── 4) SPLIT & TRAIN ─────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_p = RandomForestRegressor(
    n_estimators=250,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)
model_p.fit(X_train, y_train)

# ─── 5) EVALUATE ──────────────────────────────────────────────────────────────
y_pred = model_p.predict(X_val)
print(f"ΔP samples = {len(y):6d}")
print(f"MSE       = {mean_squared_error(y_val, y_pred):.4f}")
print(f"R²        = {r2_score(y_val, y_pred):.4f}")

# ─── 6) SAVE MODEL ────────────────────────────────────────────────────────────
joblib.dump(model_p, "delta_p_model.pkl")
print("✅ ΔP model trained and saved as delta_p_model.pkl")
