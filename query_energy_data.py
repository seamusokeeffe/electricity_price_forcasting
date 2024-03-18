import time
import pandas as pd
from entsoe import EntsoePandasClient
import os
from collections import defaultdict
from entsoe.exceptions import NoMatchingDataError
entsoe_key = os.environ["ENTSOE_API_KEY"]

client = EntsoePandasClient(api_key=entsoe_key)

start_date = pd.Timestamp('20200101', tz='UTC')
end_date = pd.Timestamp('20240318', tz='UTC')
batch_size_days = 365

### Energy Data

country_code_1 = 'IE_SEM'
country_code_2 = 'GB'

energy_data_features = [
    "day_ahead_prices",
    "load",
    "load_forecast",
    "generation",
]

# energy_data_features = [
#     "day_ahead_prices",
#     # "aggregated_bids",
#     "load",
#     "load_forecast",
#     "generation",
#     # "generation_per_plant",
#     "generation_forecast",
#     "wind_solar_forecast",
#     "intraday_wind_and_solar_forecast"
#     # "installed_generation_capacity",
#     # "installed_generation_capacity_per_unit",
#     # "aggregate_water_reservoirs_and_hydro_storage",
#     "scheduled_exchanges",
#     "crossborder_flow_net",
#     "imports",
#     "net_transfer_capacity_dayahead",
#     "net_transfer_capacity_monthahead",
#     # "net_transfer_capacity_weekahead",
#     # "net_transfer_capacity_yearahead",
#     # "offered_capacity",
#     # "intraday_offered_capacity",
#     # "activated_balancing_energy_prices",
#     "imbalance_prices",
#     "imbalance_volumes",
#     # "procured_balancing_capacity",
#     # "activated_balancing_energy",
#     "unavailability_of_generation_units",
#     "unavailability_of_production_units",
#     "unavailability_of_transmission_units",
#     "withdrawn_unavailability_of_generation",
#
# ]

def physical_crossborder_allborders(country_code, start, end):
    crossborder_flows_1 = client.query_physical_crossborder_allborders(country_code, export=True,
                                                                       per_hour=True, start=start, end=end)
    crossborder_flows_2 = client.query_physical_crossborder_allborders(country_code, export=False,
                                                                       per_hour=True, start=start, end=end)
    return crossborder_flows_1['sum'] - crossborder_flows_2['sum']


# MARKETAGREEMENTTYPE = {'A01': 'Daily',
#                        'A02': 'Weekly',
#                        'A03': 'Monthly',
#                        'A04': 'Yearly',
#                        'A05': 'Total',
#                        'A06': 'Long term',
#                        'A07': 'Intraday',
#                        'A13': 'Hourly'}

# PROCESSTYPE = {
#     'A01': 'Day ahead',
#     'A02': 'Intra day incremental',
#     'A16': 'Realised',
#     'A18': 'Intraday total',
#     'A31': 'Week ahead',
#     'A32': 'Month ahead',
#     'A33': 'Year ahead',
#     'A39': 'Synchronisation process',
#     'A40': 'Intraday process',
#     'A46': 'Replacement reserve',
#     'A47': 'Manual frequency restoration reserve',
#     'A51': 'Automatic frequency restoration reserve',
#     'A52': 'Frequency containment reserve',
#     'A56': 'Frequency restoration reserve'
# }

queries = {}
queries["day_ahead_prices"] = lambda start, end:client.query_day_ahead_prices(country_code_1, start=start, end=end)
queries["aggregated_bids"] = lambda start, end:client.query_aggregated_bids(country_code_1, process_type='A47', start=start, end=end)
queries["load"] = lambda start, end :client.query_load(country_code_1, start=start, end=end)
queries["load_forecast"] = lambda start, end: client.query_load_forecast(country_code_1, start=start, end=end)
queries["generation"] = lambda start, end: client.query_generation(country_code_1, start=start, end=end, nett=True)
queries["generation_per_plant"] = lambda start, end: client.query_generation_per_plant(country_code_1, start=start, end=end)
queries["generation_forecast"] = lambda start, end: client.query_generation_forecast(country_code_1, start=start, end=end)
queries["wind_solar_forecast"] = lambda start, end: client.query_wind_and_solar_forecast(country_code_1, start=start, end=end)
queries["intraday_wind_and_solar_forecast"] = lambda start, end: client.query_intraday_wind_and_solar_forecast(country_code_1, start=start, end=end)
queries["installed_generation_capacity"] = lambda start, end: client.query_installed_generation_capacity(country_code_1, start=start, end=end)
queries["installed_generation_capacity_per_unit"] = lambda start, end: lambda start, end: client.installed_generation_capacity_per_unit(country_code_1, start=start, end=end)
queries["aggregate_water_reservoirs_and_hydro_storage"] = lambda start, end: client.query_aggregate_water_reservoirs_and_hydro_storage(country_code_1, start=start, end=end)
queries["scheduled_exchanges"] = lambda start, end: client.query_scheduled_exchanges(country_code_1, country_code_2, start=start, end=end)
queries["crossborder_flow_net"] = lambda start, end: physical_crossborder_allborders(country_code_1, start, end)
queries["imports"] = lambda start, end: client.query_import(country_code_1, start=start, end=end)['sum']
queries["net_transfer_capacity_dayahead"] = lambda start, end: client.query_net_transfer_capacity_dayahead(country_code_1, country_code_2, start=start, end=end)
queries["net_transfer_capacity_monthahead"] = lambda start, end: client.query_net_transfer_capacity_monthahead(country_code_1, country_code_2, start=start, end=end)
queries["net_transfer_capacity_weekahead"] = lambda start, end: client.query_net_transfer_capacity_weekahead(country_code_1, country_code_2, start=start, end=end)
queries["net_transfer_capacity_yearahead"] = lambda start, end: client.query_net_transfer_capacity_yearahead(country_code_1, country_code_2, start=start, end=end)
queries["offered_capacity"] = lambda start, end: client.query_offered_capacity(country_code_1, country_code_2, contract_marketagreement_type='A01', start=start, end=end)
queries["intraday_offered_capacity"] = lambda start, end: client.query_intraday_offered_capacity(country_code_1, country_code_2, start=start, end=end)
queries["activated_balancing_energy_prices"] = lambda start, end: client.query_activated_balancing_energy_prices(country_code_1, start=start, end=end)
queries["imbalance_prices"] = lambda start, end: client.query_imbalance_prices(country_code_1, start=start, end=end)
queries["imbalance_volumes"] = lambda start, end: client.query_imbalance_volumes(country_code_1, start=start, end=end)
queries["procured_balancing_capacity"] = lambda start, end: client.query_procured_balancing_capacity(country_code_1, process_type='A47', start=start, end=end)
queries["activated_balancing_energy"] = lambda start, end: client.query_activated_balancing_energy(country_code_1, country_code_2, start=start, end=end)
queries["unavailability_of_generation_units"] = lambda start, end: client.query_unavailability_of_generation_units(country_code_1, start=start, end=end)
queries["unavailability_of_production_units"] = lambda start, end: client.query_unavailability_of_production_units(country_code_1, start=start, end=end)
queries["unavailability_of_transmission_units"] = lambda start, end: client.query_unavailability_transmission(country_code_1, country_code_2, start=start, end=end)
queries["withdrawn_unavailability_of_generation"] = lambda start, end: client.query_withdrawn_unavailability_of_generation_units(country_code_1, start=start, end=end)


for feature in energy_data_features:
    print(f"Running for {feature}")

    temp_start_date = start_date
    temp_end_date = end_date
    temp_batch_end_date = temp_start_date + pd.Timedelta(days=batch_size_days) - pd.Timedelta(seconds=1)

    while temp_start_date < temp_end_date:

        temp_filename = f".\\data\\{str(temp_start_date).split(' ')[0]}_{str(temp_batch_end_date).split(' ')[0]}_{country_code_1}_{feature}.csv"

        if not os.path.exists(temp_filename):
            print(f"File doesn't exist for {feature}: {str(temp_start_date).split(' ')[0]} - {str(temp_batch_end_date).split(' ')[0]}, generating")
            try:
                temp_df = queries[feature](temp_start_date, temp_batch_end_date)
                temp_df.index.name = 'datetime'

                if isinstance(temp_df, pd.Series):
                    temp_df.name = feature
                    temp_df = temp_df.to_frame()

                temp_df = temp_df.reset_index()
                temp_df.to_csv(temp_filename, index=False)
            except NoMatchingDataError:
                pass
        else:
            print(f"File exists for {feature}: {str(temp_start_date).split(' ')[0]}-{str(temp_batch_end_date).split(' ')[0]}, skippng")

        temp_start_date = temp_batch_end_date + pd.Timedelta(seconds=1)
        temp_batch_end_date = temp_start_date + pd.Timedelta(days=batch_size_days) - pd.Timedelta(seconds=1)
        if temp_batch_end_date > temp_end_date:
            temp_batch_end_date = temp_end_date







