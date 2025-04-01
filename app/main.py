from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib

# FastAPI app to serve predictions for the Iris classifier


# Define expected input format: a list of 4 float features
class InputData(BaseModel):
    features: List[float]

# Load pre-trained model once at app startup
app = FastAPI()
model = joblib.load('app/iris_model.pkl')

@app.post("/predict")
# Make prediction using input features
# Return predicted class label as JSON

def predict(data: InputData):
    prediction = model.predict([data.features])
    return {"prediction": prediction.tolist()}

