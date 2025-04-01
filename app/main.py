from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib

class InputData(BaseModel):
    features: List[float]

app = FastAPI()
model = joblib.load('app/iris_model.pkl')

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([data.features])
    return {"prediction": prediction.tolist()}

