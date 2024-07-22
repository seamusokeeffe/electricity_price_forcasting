import pandas as pd
import os

country_code = "NL"
df_unav_gen = pd.read_csv(f"..\\data\\unavailability_of_generation_units_{country_code}.csv")
df_unav_gen["start"] = pd.to_datetime(df_unav_gen["start"], utc=True).dt.tz_localize(None)
df_unav_gen["end"] = pd.to_datetime(df_unav_gen["end"], utc=True).dt.tz_localize(None)
df_unav_gen = df_unav_gen.drop_duplicates()

check_start_lt_end = (df_unav_gen["start"] < df_unav_gen["end"]).sum() == len(df_unav_gen)
check_aq_lte_np = (df_unav_gen["avail_qty"] <= df_unav_gen["nominal_power"]).sum() == len(df_unav_gen)
print(f"Are all 'start' entries strictly less than 'end' entries, {check_start_lt_end}")
print(f"Are all 'nominal_power' entries greater than or equal to 'avail_qty', {check_aq_lte_np }")

df_unav_gen["missing_qty"] = df_unav_gen["nominal_power"].sub(df_unav_gen["avail_qty"])
print("missing_qty distribution")
print(df_unav_gen["missing_qty"].describe())

plant_types = df_unav_gen["plant_type"].unique()

for plant_type in plant_types:
    plant_type_edit = plant_type.lower().replace(' ', '_')
    print(f"Generating {plant_type} unavailability_of_generation_units timeseries")

    temp_filename = f"..\\data\\unavailability_of_generation_units_{country_code}_{plant_type_edit}.csv"

    if not os.path.exists(temp_filename):

        df_pt_tmp = df_unav_gen[df_unav_gen["plant_type"] == plant_type]
        df_pt_tmp = df_pt_tmp.sort_values(["start", "end"])
        start_tmp = df_pt_tmp["start"].values
        end_tmp = df_pt_tmp["end"].values
        missing_quantity_tmp = df_pt_tmp["missing_qty"].values
        n_tmp = len(df_pt_tmp)
        tmp_data = []
        for i in range(n_tmp):
            df_pt_tmp_sub = pd.DataFrame()
            df_pt_tmp_sub["datetime"] = pd.date_range(start_tmp[i], end_tmp[i], freq="1h")
            df_pt_tmp_sub[f"mq_{i}"] = missing_quantity_tmp[i]
            df_pt_tmp_sub = df_pt_tmp_sub.set_index("datetime")
            tmp_data.append(df_pt_tmp_sub)

        df_pt_tmp_full = pd.concat(tmp_data, axis=1).fillna(0)
        df_pt_tmp_full = df_pt_tmp_full.sum(axis=1)
        df_pt_tmp_full.name = f"generation_{plant_type_edit}_missing_qty"
        df_pt_tmp_full = df_pt_tmp_full.to_frame().reset_index()
        df_pt_tmp_full.to_csv(temp_filename, index=False)
