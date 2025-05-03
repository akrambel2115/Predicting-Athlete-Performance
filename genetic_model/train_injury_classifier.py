import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
import joblib

# Load and sort dataset
df = pd.read_csv("../data/soccerData/soccer_data_cleaned.csv", parse_dates=["date"])
df = df.sort_values(["player_id", "date"]).reset_index(drop=True)

# Ensure injury_flag is integer for logical ops
df["injury_flag"] = df["injury_flag"].fillna(0).astype(int)

# Create future injury proxy (injury in next 3 days)
grouped = df.groupby("player_id")["injury_flag"]
df["injury_within_3"] = (
    grouped.shift(-1).fillna(0).astype(int) |
    grouped.shift(-2).fillna(0).astype(int) |
    grouped.shift(-3).fillna(0).astype(int)
).astype(int)

# Drop rows with null target
df = df.dropna(subset=["injury_within_3"])

# Select features
FEATURES = [
    "load",
    "fatigue_post",
    "performance_lag_1",
    "sleep_duration",
    "sleep_quality",
    "stress",
    "is_rest_day",
    "injury_flag_lag_1",
    "load_rolling_7",
    "fatigue_post_rolling_7"
]

X = df[FEATURES]
y = df["injury_within_3"]

# Handle missing values
imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=FEATURES)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X_imp, y, test_size=0.2, stratify=y, random_state=42
)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_val)
y_prob = clf.predict_proba(X_val)[:, 1]
print("ROC AUC:", roc_auc_score(y_val, y_prob))
print(classification_report(y_val, y_pred))

# Save pipeline
# train_injury_classifier.py (fix this section if needed)
joblib.dump({
    "model": clf,               # your trained classifier
    "features": list(X.columns) # save the feature list used
}, "delta_r_classifier.pkl")

print("✅ Risk classifier saved as delta_r_classifier.pkl")
