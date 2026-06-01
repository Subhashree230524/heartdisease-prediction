from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

model = pickle.load(open("trained_model.sav","rb"))

app = FastAPI()

class HeartInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.get("/")
def home():
    return {"message":"Heart Disease Prediction API"}

@app.post("/predict")
def predict(data: HeartInput):

    input_data = np.array([
        data.age,
        data.sex,
        data.cp,
        data.trestbps,
        data.chol,
        data.fbs,
        data.restecg,
        data.thalach,
        data.exang,
        data.oldpeak,
        data.slope,
        data.ca,
        data.thal
    ]).reshape(1,-1)

    prediction = model.predict(input_data)

    return {
        "prediction": int(prediction[0])
    }