import numpy as np
import pandas as pd
import datetime


def load_sensor(path, save=False) -> [pd.DataFrame]:
    # Setting up the date_time dataset
    sdate=datetime.date(2020, 5, 15)
    edate = datetime.date(2020, 6, 17)
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

    df_nan = df.merge(timeset, left_index=True, right_index=True, how='left')

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
