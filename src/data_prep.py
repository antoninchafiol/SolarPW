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

