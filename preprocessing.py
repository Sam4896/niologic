import pandas as pd
from sodapy import Socrata
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def add_time_features(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df["year"] = df['datetime'].apply(lambda x: x.year)
#     df['year'] = df['year']-df['year'].iloc[0]
    df["month"] = df["datetime"].apply(lambda x: x.month)
    df["dayofweek"] = df["datetime"].apply(lambda x: x.dayofweek)
#     df['weekofyear'] = df['datetime'].apply(lambda x: x.weekofyear)
    df["hour"] = df["datetime"].apply(lambda x: x.hour)
    df['day'] = df['datetime'].apply(lambda x: x.day)
    df['date'] = df['datetime'].apply(lambda x: x.date())
    return df

def group_df_func(df, datetime_list, date_col):
    temp = df.groupby(datetime_list).size().reset_index(name='num_calls')
#     temp['year'] = temp.year+df.iloc[0].datetime.year
    if date_col == True:
        extra = set(datetime_list)-set(['year', 'month', 'day','hour','minute'])
        datetime_list = list(set(datetime_list) - extra)
        if 'hour' not in datetime_list:
            temp['date'] = pd.to_datetime(temp[datetime_list]).apply(lambda x: x.date())
        elif 'hour' in datetime_list:
            temp['date'] = pd.to_datetime(temp[datetime_list]).apply(lambda x: x.ctime())
        temp = temp.set_index(['date']).reset_index()
    return temp