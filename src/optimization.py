import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor

import numpy as np

def main_optimization(X, y, type='normal'):
    x_train, x_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=42)
    metric = {}
    model = 0
    if type=='normal':
        metric, model = normgb_opt(x_train, y_train)
    elif type=='light':
        metric, model = lightgb_opt(x_train, y_train)
    return metric, model

def normgb_opt(x_train, y_train):
    param_dist = {
        'n_estimators': np.arange(50, 200, 5),
        'max_depth': np.arange(2, 10),
        'learning_rate': np.arange(0.01, 1, 0.05),
        'subsample': np.arange(0.1, 1.0, 0.1), 
        'max_features': np.arange(0.1, 1.0, 0.1),
        'criterion': ['squared_error', 'friedman_mse'],
        'loss': ['squared_error', 'absolute_error'],
    }

    search = RandomizedSearchCV(
        GradientBoostingRegressor(), 
        param_dist,
        n_iter=5000, 
        cv=2, 
        n_jobs=-1, 
        verbose=1
    )
    search.fit(x_train, y_train)

    return search.best_params_, search.best_estimator_

def lightgb_opt(x_train, y_train):
    param_dist = {
        'n_estimators': np.arange(50, 200, 5),
        'max_depth': np.arange(2, 10, 1),
        'learning_rate': np.linspace(0.01, 1, 50),
        'subsample': np.arange(0.1, 1.0, 0.1), 
        'max_features': np.arange(0.1, 1.0, 0.15),
        'lambda_l2': np.linspace(0.01, 0.5, 10),
    }
    search = RandomizedSearchCV(
        lgb.LGBMRegressor(), 
        param_dist,
        n_iter=50000, 
        cv=2, 
        n_jobs=-1, 
        verbose=1
    )
    search.fit(x_train, y_train)

    return search.best_params_, search.best_estimator_


# from sklearn.model_selection import train_test_split
# from data_prep import *
# from feature_engineering import *

# dfs = load_sensor("dataset/Plant_1_Weather_Sensor_Data.csv", save=False)
# irr_X, irr_y = sensor_for_irradiation(dfs[0])

# x_train, x_dev, y_train, y_dev = train_test_split(irr_X, irr_y, test_size=0.2, random_state=42)
# params, model = normgb_opt(x_train, y_train)
# print(params)