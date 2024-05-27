import requests
import datetime
import matplotlib.pyplot as plt
import json
import pandas as pd
import streamlit as st

st.title('Netherlands Electricity Price Prediction')

d = st.date_input("Select a date to predict prices",
                  value=None,
                  min_value=datetime.date(2023,5,23),
                  max_value=datetime.date(2024,3,23))

def plot_predictions(df, _date):
    fig, ax = plt.subplots(figsize=(12, 5))
    df.plot(ax=ax,
            title=f'{_date} Day Ahead Electricity Price Prediction',
            ylabel='Price (€/MWh)')
    ax.legend()
    st.pyplot(fig)

# displays a button
if st.button(f"Generate Predictions"):
    if d is not None:
        input_data = {"date": str(d)}
        # response = requests.post('http://127.0.0.1:8000/predict', json=input_data)
        response = requests.post('http://backend:8000/predict', json=input_data)
        df = pd.json_normalize(response.json())
        df["datetime"] = pd.to_datetime(df["datetime"],unit='ms')
        df = df.set_index("datetime")
        plot_predictions(df, d)
        st.write(df.round(2))
    else:
        st.write("Date not selected!")