from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def train_model(model, train_df, test_df, pred_type, current_year):
    ## feature and target split
    train_features, train_target = train_df.drop('num_calls',axis=1), train_df['num_calls']
    test_features, test_target = test_df.drop('num_calls',axis=1), test_df['num_calls']
    
#     train_features, train_target = train_df.drop(['num_calls','difference'],axis=1), train_df['difference']
#     test_features, test_target = test_df.drop(['num_calls','difference'],axis=1), test_df['difference']

    model.fit(train_features,train_target)
    return model