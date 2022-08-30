from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def convert_results_dataframe(test_target, prediction, test_features, pred_type):
    results_df = pd.DataFrame()
    results_df['target'] = test_target
    results_df['predicted'] = prediction
    if pred_type == 'daily':
        results_df[['year','month','day']] = test_features[['year','month','day']]
#         results_df['year'] = results_df.year+current_year
        results_df['date'] = pd.to_datetime(results_df[['year','month','day']]).apply(lambda x: x.date())
    elif pred_type == 'hourly':
        results_df = results_df.reset_index(drop = True)
        test_features = test_features.reset_index(drop = True)
        results_df[['year','month','day','hour']] = test_features[['year','month','day','hour']]
#         display(results_df)
#         results_df['year'] = results_df.year+current_year
        results_df['date'] = pd.to_datetime(results_df[['year','month','day','hour']]).apply(lambda x: x.ctime())
    return results_df

def plot_results(results_df):
    results_df = results_df.reset_index(drop=True)
    fig, ax = plt.subplots()
    ax.plot(results_df['target'], ls= '-.', marker='o', label = 'True', linewidth = 2)
    ax.plot(results_df['predicted'], ls= '-.', marker='o', label = 'Predicted', linewidth = 2)
    ax.set_ylabel('Number of calls')
    return fig, ax

def test_model(model, test_df, pred_type):
    
#     test_features, test_target = test_df.drop(['num_calls','difference'],axis=1), test_df['difference']
    test_features, test_target = test_df.drop('num_calls',axis=1), test_df['num_calls']
    
    pred = model.predict(test_features)
    
    ## making a dataframe for the predictions
    results_df = convert_results_dataframe(test_target, pred, test_features, pred_type = pred_type)
    return results_df

def evaluate_predictions(results_df, time_period, pred_type, title):
    ## Metric to quantify the error
    print(f'The mean squared error is: {mean_squared_error(results_df.target,results_df.predicted)}')
    print(f'The mean absolute error is: {mean_absolute_error(results_df.target,results_df.predicted)}')

    ## Plotting
    if pred_type == 'daily':
        fig, ax = plot_results(results_df.iloc[-time_period:-1]) #omitting the last prediction
        ax.set_title(title)
        ax.set_xticks(np.arange(len(results_df.iloc[-time_period:-1].date))[0::int(time_period/15)])
        _ = ax.set_xticklabels(results_df.iloc[-time_period:-1].date[0::int(time_period/15)], rotation=45)
        plt.legend()
        plt.show()
    elif pred_type == 'hourly':
        fig, ax = plot_results(results_df.iloc[-time_period:-1]) #omitting the last prediction
        ax.set_xticks(np.arange(len(results_df.iloc[-time_period:-1].date))[0::int(time_period/14)])
        _ = ax.set_xticklabels(results_df.iloc[-time_period:-1].date[0::int(time_period/14)], rotation=45)
        ax.set_title(title)
        plt.legend()
        plt.show()
    return results_df