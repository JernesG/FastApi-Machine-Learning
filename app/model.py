import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

#Folder to save model artifacts
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "/model.pkl")

def ensure_artifacts_dir():

    """Ensure Artifacts folder exists.."""

    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)


def train_model(data_path="data/train_data.csv"):

    """Train and save the linearRegression Model...""" 

    ensure_artifacts_dir()

    df = pd.read_csv(data_path)
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
    x_pred = pd.DataFrame([input_data])
    pred = model.predict(x_pred)[0]
    return float(pred)


def retrain_model():

    """Retrain model using new data"""

    return train_model("data/new_data.csv")



