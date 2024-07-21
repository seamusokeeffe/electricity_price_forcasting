import datetime
import pandas as pd
from fastapi import FastAPI, Response
from pydantic import BaseModel

# Load the Predictions and ground truth
filename_pred = "predictions_test.csv"
df = pd.read_csv(filename_pred)
df["datetime"] = pd.to_datetime(df["datetime"])

class user_input(BaseModel):
    date: str

app = FastAPI(title="Netherlands Electricity Price Prediction",
              description='''Obtain forecasts of Dutch electricity prices for given date.
                           Visit this URL at port 8501 for the streamlit interface.''',
              version="0.1.0",)

@app.get("/")
def read_root():
    return {"message": "Welcome to the model API!"}

@app.post("/predict")
def get_date(data: dict):
    temp_df = df[df["datetime"].dt.date == datetime.datetime.strptime(data["date"],'%Y-%m-%d').date()].copy()
    return Response(temp_df.to_json(orient="records"), media_type="application/json")