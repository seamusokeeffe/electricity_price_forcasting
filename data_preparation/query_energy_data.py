import pandas as pd
from entsoe import EntsoePandasClient
import os
from entsoe.exceptions import NoMatchingDataError
entsoe_key = os.environ["ENTSOE_API_KEY"]

client = EntsoePandasClient(api_key=entsoe_key)

start_date = pd.Timestamp('20160101', tz='UTC')
end_date = pd.Timestamp('20200101', tz='UTC')

batch_size_days = 10

country_codes = ["NL"]

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

for country_code in country_codes:
    print(f"Running for {country_code}")

    country_code_1 = country_code

    def physical_crossborder_allborders(country_code, start, end):
        crossborder_flows_1 = client.query_physical_crossborder_allborders(country_code, start=start, end=end,
                                                                           export=True, per_hour=True)
        crossborder_flows_2 = client.query_physical_crossborder_allborders(country_code, start=start, end=end,
                                                                           export=False, per_hour=True)
        return crossborder_flows_1['sum'] - crossborder_flows_2['sum']

    queries = {}
    queries["day_ahead_prices"] = lambda start, end:client.query_day_ahead_prices(country_code_1, start=start, end=end)
    queries["load"] = lambda start, end :client.query_load(country_code_1, start=start, end=end)
    queries["load_forecast"] = lambda start, end: client.query_load_forecast(country_code_1, start=start, end=end)
    queries["generation"] = lambda start, end: client.query_generation(country_code_1, start=start, end=end, nett=True)
    queries["generation_forecast"] = lambda start, end: client.query_generation_forecast(country_code_1, start=start, end=end)
    queries["wind_solar_forecast"] = lambda start, end: client.query_wind_and_solar_forecast(country_code_1, start=start, end=end)
    queries["crossborder_flow_net"] = lambda start, end: physical_crossborder_allborders(country_code_1, start, end)
    queries["imports"] = lambda start, end: client.query_import(country_code_1, start=start, end=end)['sum']
    queries["unavailability_of_generation_units"] = lambda start, end: client.query_unavailability_of_generation_units(country_code_1, start=start, end=end)

    for feature in energy_data_features:
        print(f"Running for {feature}")

        temp_start_date = start_date
        temp_end_date = end_date
        temp_batch_end_date = temp_start_date + pd.Timedelta(days=batch_size_days) - pd.Timedelta(seconds=1)

        while temp_start_date < temp_end_date:

            temp_filename = f"..\\data\\{str(temp_start_date).split(' ')[0]}_{str(temp_batch_end_date).split(' ')[0]}_{country_code_1}_{feature}.csv"

            if not os.path.exists(temp_filename):
                print(f"File doesn't exist for {feature}: {str(temp_start_date).split(' ')[0]} - {str(temp_batch_end_date).split(' ')[0]}, generating")

                while True:
                    try:
                        temp_df = queries[feature](temp_start_date, temp_batch_end_date)
                        temp_df.index.name = 'datetime'

                        if isinstance(temp_df, pd.Series):
                            temp_df.name = feature
                            temp_df = temp_df.to_frame()

                        temp_df = temp_df.reset_index()
                        if temp_batch_end_date == (temp_end_date + pd.Timedelta(hours=1)):
                            temp_df = temp_df[temp_df["datetime"] <= temp_end_date]

                        temp_df.to_csv(temp_filename, index=False)
                        print(f"Writing file for {feature}: {str(temp_start_date).split(' ')[0]} - {str(temp_batch_end_date).split(' ')[0]}, generating")
                    except NoMatchingDataError:
                        print('NoMatchingDataError - skipping')
                        break
                    except Exception as e:
                        print(e)
                        print('retrying')
                        continue
                    break
            else:
                print(f"File exists for {feature}: {str(temp_start_date).split(' ')[0]}-{str(temp_batch_end_date).split(' ')[0]}, skippng")

            temp_start_date = temp_batch_end_date + pd.Timedelta(seconds=1)
            temp_batch_end_date = temp_start_date + pd.Timedelta(days=batch_size_days) - pd.Timedelta(seconds=1)
            if (temp_batch_end_date + pd.Timedelta(seconds=1)) >= temp_end_date:
                temp_batch_end_date = temp_end_date + pd.Timedelta(hours=1)







