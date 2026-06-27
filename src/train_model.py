import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# 1. Load dataset
# -----------------------------
data_file = "data/raw/synthetic_telemetry.csv"

df = pd.read_csv(data_file)

print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", list(df.columns))

# -----------------------------
# 2. Select features and label
# -----------------------------
features = [
    "request_interval_seconds",
    "user_agent_category",
    "has_favicon_request",
    "requested_robots_txt",
    "pages_per_session",
    "error_rate",
    "tls_version",
    "cipher_suite_count",
    "extension_count",
    "alpn",
    "sni_present"
]

target = "label"

X = df[features].copy()
y = df[target].copy()

# -----------------------------
# 3. Encode text columns
# -----------------------------
encoders = {}

for column in X.columns:
    if X[column].dtype == "object":
        encoder = LabelEncoder()
        X[column] = encoder.fit_transform(X[column])
        encoders[column] = encoder

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -----------------------------
# 4. Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# -----------------------------
# 5. Train model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 6. Evaluate model
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel training complete.")
print("Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# 7. Save model
# -----------------------------
os.makedirs("models", exist_ok=True)

model_package = {
    "model": model,
    "feature_columns": features,
    "encoders": encoders,
    "label_encoder": label_encoder
}

with open("models/bot_detection_model.pkl", "wb") as file:
    pickle.dump(model_package, file)

print("\nSaved model to models/bot_detection_model.pkl")