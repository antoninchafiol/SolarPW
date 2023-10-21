import numpy as np
import pandas as pd
import datetime


def load_sensor(path, save=False) -> [pd.DataFrame]:
    # Setting up the date_time dataset
    sdate=datetime.date(2020, 5, 15)
    edate = datetime.date(2020, 6, 18)
    timeset = pd.DataFrame({'DATE_TIME' : pd.date_range(start=sdate, end=edate, freq='15min')})
    timeset['date'] = timeset['DATE_TIME'].dt.date
    timeset['time'] = timeset['DATE_TIME'].dt.time
    timetemp = pd.DataFrame({'time': timeset['time'].unique()})
    timetemp['time_id'] = [x for x in range(timetemp.shape[0])]
    timeset= timeset.merge(timetemp, on='time')
    
    # Loading csv
    df = pd.read_csv(path)
    df = df.drop(columns=['SOURCE_KEY','PLANT_ID'])
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

    df_nan = df.merge(timeset, how='right', on='DATE_TIME')
    df['date'] = df['DATE_TIME'].dt.date
    df['time'] = df['DATE_TIME'].dt.time
    temp = pd.DataFrame({'time': df['time'].unique()})
    temp['time_id'] = [x for x in range(temp.shape[0])]
    df = df.merge(temp, on='time')
    if save:
        parquet_path  =  "/".join(path.split("/")[:-1] + ["parquet_dir"] + [path.split("/")[-1][:-4]])
        df.to_parquet(parquet_path)
        df_nan.to_parquet(parquet_path + "_with_nan")        
    return [df, df_nan]


def write_df(models, dest_dataframe):
    '''
    Apply models predictions to dataframe

    Parameters
    ----------
    models: dict of models
        Dict of the differents trained models:
        - 1st: Irradiation
        - 2nd: Ambient temp
        - 3rd: Module temp
    dest_dataframe: DataFrame
        Dataframe to update
    '''
    
    for k, v in models.items():
        if len(dest_dataframe[k]) != 0:
            dest_dataframe = apply_model(v, dest_dataframe, k)
        else:
            print(f"{k} is not a right key (not found)")

    return dest_dataframe 


    # After fixing the date_time, noticed that we only have the aall temp and irradiation NaN, which will be a mess for predicting, as all I need to use then is the time_id to somwewhat predict each of those.
    # GB algo might be performing the best on it so keeping these bad boys to testing time!
    # As it could be useful to train for 1 input, 3 output regressors values.


def apply_model(trained_model, src_dataframe, column):
    na_df = src_dataframe[src_dataframe[column].isna()]
    ndf_cols = na_df.columns.values.tolist()
    ndf_cols.remove('time_id')
    temp_df = na_df.drop(columns=ndf_cols)
    # print(np.round(trained_model.predict(temp_df), 3))
    na_df[column] = np.round(trained_model.predict(temp_df), 3)
    src_dataframe[src_dataframe[column].isna()] = na_df
    return src_dataframe

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from feature_engineering import *

dfs = load_sensor("dataset/Plant_1_Weather_Sensor_Data.csv", save=False)
model_irr = GradientBoostingRegressor()
model_at = GradientBoostingRegressor()
model_am = GradientBoostingRegressor()

irr_X, irr_y = lone_sensor_for_irradiation(dfs[0])
iX_train, iX_dev, iy_train, iy_dev = train_test_split(irr_X, irr_y, test_size=0.2, random_state=42)
at_X, at_y = lone_sensor_for_ambient_temp(dfs[0])
atX_train, atX_dev, aty_train, aty_dev = train_test_split(at_X, at_y, test_size=0.2, random_state=42)
# print(iX_train.values)
amX, amy = lone_sensor_for_module_temp(dfs[0])
amX_train, amX_dev, amy_train, amy_dev =  train_test_split(amX, amy, test_size=0.2, random_state=42)
model_irr.fit(iX_train, iy_train)
model_at.fit(atX_train, aty_train)

models = {
    "IRRADIATION": model_irr, 
    "AMBIENT_TEMPERATURE": model_at,
    }

df_irr = write_df(models, dfs[1])
print(df_irr)
# print(len(dfs[0]))
# print(len(dfs[1]))

# print(dfs[1][dfs[1]['IRRADIATION'].isna()])

# dfs[1] = write_df(0, dfs[1], 'IRRADIATION')
# dfs[1] = write_df(0, dfs[1], 'MODULE_TEMPERATURE')
# dfs[1] = write_df(0, dfs[1], 'AMBIENT_TEMPERATURE')

# print(apply_model(0, dfs[1], 'IRRADIATION'))

