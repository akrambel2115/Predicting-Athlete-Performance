#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ─── 1) LOAD & PREPARE ──────────────────────────────────────────────────────────
df = pd.read_csv("../data/soccerData/soccer_data_cleaned.csv", parse_dates=["date"])
df = df.sort_values(["player_id", "date"]).reset_index(drop=True)

# Use the filled fatigue, drop the original
# df["fatigue_post"] = df["fatigue_post_filled"]

# Recompute a 7‑day rolling average on fatigue_post
df["fatigue_post_rolling_7"] = (
    df
    .groupby("player_id")["fatigue_post"]
    .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
)

# ─── 2) INJECT INTENSITY & COMPUTE ΔF ───────────────────────────────────────────
# Map action → intensity
action_intensity_map = {
    "Train_Low":  0.3,
    "Train_Med":  0.6,
    "Train_High": 0.9,
}
df["action_intensity"] = df["action"].map(action_intensity_map).fillna(0.0)

# Compute next‑day delta_f
df["delta_f"] = (
    df.groupby("player_id")["fatigue_post"].shift(-1)
    - df["fatigue_post"]
)

# ─── 3) FEATURES & FILTER ───────────────────────────────────────────────────────
FEATURES_F = [
    "load",
    "action_intensity",
    "fatigue_post",       # ← now use the filled values
    "sleep_duration",
    "sleep_quality",
    "stress",
    "is_rest_day",
    "load_rolling_7",
    "fatigue_post_rolling_7",    # ensure you compute this from fatigue_post_filled as well
    "stress_rolling_7",
    "sleep_duration_rolling_7",
    "sleep_quality_rolling_7",
    "total_duration",
    "load_lag_1",
]


# Keep only rows where all features + target are present
df_f = df.dropna(subset=["delta_f"] + FEATURES_F)

Xf = df_f[FEATURES_F]
yf = df_f["delta_f"]

# ─── 4) TRAIN / EVAL ────────────────────────────────────────────────────────────
Xf_train, Xf_val, yf_train, yf_val = train_test_split(
    Xf, yf, test_size=0.2, random_state=42
)

model_f = RandomForestRegressor(n_estimators=100, random_state=42)
model_f.fit(Xf_train, yf_train)

yf_pred = model_f.predict(Xf_val)
mse = mean_squared_error(yf_val, yf_pred)

print(f"ΔF samples={len(Xf):,}  MSE={mse:.4f}")

# ─── 5) SAVE MODEL ─────────────────────────────────────────────────────────────
joblib.dump(model_f, "delta_f_model.pkl")
print("✅ ΔF model trained & saved as delta_f_model.pkl")
