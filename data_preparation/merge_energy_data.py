import pandas as pd

from os import listdir
from os.path import isfile, join

data_dir = "..\\data"
filenames = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

energy_data_features = [
    "day_ahead_prices",
    "load",
    "load_forecast",
    "generation",
    "generation_forecast",
    "wind_solar_forecast",
    "crossborder_flow_net",
    "imports",
    "unavailability_of_generation_units",
]

country_code = "NL"

for feature in features:
    temp_filename = f"..\\data\\{feature}_{country_code}_2.csv"
    temp_data = []

    for filename in filenames:
        if f"{country_code}_{feature}.csv" in filename:
            temp_data.append(pd.read_csv(join(data_dir, filename)))

    temp_df = pd.concat(temp_data)
    temp_df["datetime"] = pd.to_datetime(temp_df["datetime"], utc=True)
    temp_df = temp_df.sort_values("datetime")
    temp_df.to_csv(temp_filename, index=False)









