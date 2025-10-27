import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os
from app.drift import detect_data_drift

#Folder to save model artifacts
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "train_data.csv")

def ensure_artifacts_dir():

    """Ensure Artifacts folder exists.."""

    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)


def train_model(data_path=TRAIN_DATA_PATH):

    """Train and save the linearRegression Model...""" 

    ensure_artifacts_dir()

    df = pd.read_csv(data_path)
    print("📋 Available columns:", df.columns.tolist())
    x = df[["size","bedrooms"]]
    y = df["price"]

    model = LinearRegression()
    model.fit(x,y)

    joblib.dump(model, MODEL_PATH)
    return model


def load_model():

    """Load model from artifacts folder or retrain the model using training data"""

    ensure_artifacts_dir()

    if not os.path.exists(MODEL_PATH):
        return train_model()
    else:
        return joblib.load(MODEL_PATH)


def predict(input_data: dict):

    """Doing prediction using saved model"""   

    model = load_model()
    current_data = pd.DataFrame([input_data])
    pred = model.predict(current_data)[0]

    # --- Drift Detection ---
    if os.path.exists(TRAIN_DATA_PATH):
        reference_data = pd.read_csv(TRAIN_DATA_PATH)
        reference_data = reference_data[["size", "bedrooms"]]  # match model features
        drift_detected, drift_score = detect_data_drift(reference_data, current_data)
        if drift_detected:
            print(f"Data drift detected! (Score={drift_score:.2f})")
        else:
            print(f"No drift detected. (Score={drift_score:.2f})")

    else:
        print("No training data available for drift detection.")

    return round(float(pred),4)


def retrain_model():

    """Retrain model using new data"""

    return train_model("data/new_data.csv")