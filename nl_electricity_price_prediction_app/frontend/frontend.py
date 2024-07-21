import requests
import datetime
import pandas as pd
import streamlit as st
import altair as alt

st.title('Netherlands Electricity Price Prediction')

# Define CSS for adding space between radio buttons and setting max-width of the main container
st.markdown(
    """
    <style>
    .vega-bindings label {
        margin-right: 10px;  /* Adjust this value to increase the spacing */
    }
    .vega-bindings {
        margin-bottom: 40px;  /* Add margin below the radio buttons */
    }
    </style>
    """,
    unsafe_allow_html=True
)

d = st.date_input("Select a date to predict prices",
                  value=None,
                  min_value=datetime.date(2023,5,19),
                  max_value=datetime.date(2024,3,21))

# def plot_predictions(df, _date):
#     fig, ax = plt.subplots(figsize=(12, 5))
#     df.plot(ax=ax,
#             title=f'{_date} Day Ahead Electricity Price Prediction',
#             ylabel='Price (€/MWh)')
#     ax.legend()
#     st.pyplot(fig)

def generate_predictions_plot(df, _date):
    # Melt the dataframe to have a long format suitable for Altair
    dfm = df.melt(ignore_index=False).reset_index()
    dfm.columns = ['datetime', 'model', 'price']
    df_pred_gt = dfm[dfm['model'] == 'ground_truth'].copy()
    dfm = dfm[dfm['model'] != 'ground_truth'].copy()

    # Define the model names excluding ground_truth
    model_names = ['sarimax', 'lasso', 'ridge', 'elasticnet', 'linear_gam', 'light_gbm']

    # Create radio button input for model selection
    input_dropdown = alt.binding_radio(
        options=model_names + [None],
        labels=[name + ' ' for name in model_names] + ['All'],
        name='Model: '
    )
    selection = alt.selection_point(
        fields=['model'],
        bind=input_dropdown,
    )

    # Configuration for axes
    xaxis_config = alt.Axis(
        grid=False,
        labelFontSize=14,
        titleFontSize=16,
        titleX=0,
        titleAlign='left',
        titlePadding=10
    )

    yaxis_config = alt.Axis(
        grid=False,
        labelFontSize=14,
        titleFontSize=16
    )

    # Chart for ground truth values
    ground_truth = alt.Chart(df_pred_gt).mark_point(color='blue', size=30, opacity=0.4).encode(
        x=alt.X('datetime:T', title='Datetime', axis=xaxis_config),
        y=alt.Y('price:Q', title='Price (€/MWh)', axis=yaxis_config),
        tooltip=['datetime:T', 'price:Q', 'model:N']
    )

    # Chart for model predictions
    predictions = alt.Chart(dfm).mark_line().encode(
        x=alt.X('datetime:T', title='Datetime', axis=xaxis_config),
        y=alt.Y('price:Q', title='Price (€/MWh)', axis=yaxis_config),
        color=alt.condition(
            selection,
            alt.Color('model:N', legend=alt.Legend(title="Model"),
                      scale=alt.Scale(domain=['ground_truth'] + model_names)),
            alt.value('grey')
        ),
        opacity=alt.condition(
            selection,
            alt.value(1),
            alt.value(0)
        ),
        tooltip=['datetime:T', 'price:Q', 'model:N']
    ).add_params(
        selection
    )

    # Combine the charts
    chart = alt.layer(
        ground_truth,
        predictions
    ).properties(
        title=alt.TitleParams(
            text=f'{_date} Day Ahead Electricity Price Prediction',
            anchor='middle',
            fontSize=18
        ),
        width=900,
        height=500
    ).interactive()

    return chart

# displays a button
if st.button(f"Generate Predictions"):
    if d is not None:
        input_data = {"date": str(d)}
        # response = requests.post('http://127.0.0.1:8000/predict', json=input_data)
        response = requests.post('http://backend:8000/predict', json=input_data)
        df = pd.json_normalize(response.json())
        df["datetime"] = pd.to_datetime(df["datetime"],unit='ms')
        df = df.set_index("datetime")

        # Display the chart in the adjusted container
        chart = generate_predictions_plot(df, d)
        st.altair_chart(chart, use_container_width=True)

        st.write(df.round(2))
    else:
        st.write("Date not selected!")