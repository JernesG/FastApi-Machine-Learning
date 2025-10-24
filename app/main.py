from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from app.model import predict, retrain_model, load_model

app = FastAPI(title="FastAPI with ML")

#Input schema for prediction
class HouseData(BaseModel):
    size : float
    bedrooms : int


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup phase ---
    print("🚀 App starting... loading or training model.")
    load_model()  # ensure model exists at startup

    # Yield control to the app
    yield

    # --- Shutdown phase ---
    print("🛑 App shutting down... cleanup if needed.")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="FastAPI ML Retraining Demo with Artifacts",
    lifespan=lifespan
)


@app.post("/predict")
def make_prediction(data: HouseData):

    """Make prediction..."""

    result = predict(data.dict())
    return {"prediction":result}


@app.post("/retrain")
def retrain():
    """Retrain model on new data."""
    retrain_model()
