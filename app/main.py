from fastapi import FastAPI
from pydantic import BaseModel
from app.model import predict, retrain_model, load_model

app = FastAPI(title="FastAPI with ML")

#Input schema for prediction
class HouseData(BaseModel):
    size : float
    bedrooms : int


@app.on_event("startup")
def startup_event():
    
    """Load model at startup (train if model is not available)"""

    load_model()


@app.post("/predict")
def make_prediction(data: HouseData):

    """Make prediction..."""

    result = predict(data.dict())
    return {"prediction":result}


@app.post("/retrain")
def retrain():
    """Retrain model on new data."""
    retrain_model()
