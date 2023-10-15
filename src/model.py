from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

import numpy as np

def model_testing(X,y, mtype):
    # Quick test of models, without any tunings.
    res = []
    x_train, x_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=42)
    if mtype=='SVM':
        res= model_testing_SVR(x_train, x_dev, y_train, y_dev)
    elif mtype=='DT':
        res= model_testing_DT(x_train, x_dev, y_train, y_dev)
    elif mtype=='RF':
        res= model_testing_RF(x_train, x_dev, y_train, y_dev)
    elif mtype=='GB':
        res= model_testing_GB(x_train, x_dev, y_train, y_dev)
    return res

def plain_testing(x_train, x_dev, y_train, y_dev, model):
    model.fit(x_train, y_train)
    preds = model.predict(x_dev)
    return model, {
        'r2': r2_score(preds, y_dev),
        'MAE': mean_absolute_error(preds, y_dev), 
        'MSE': mean_squared_error(preds, y_dev, squared=True),
        'RMSE': mean_squared_error(preds, y_dev, squared=False)
    }

# For future hyperparameters tuning
def model_testing_SVR(x_train, x_dev, y_train, y_dev):
    model = SVR()
    param_dist = {
        'kernel': ['linear','rbf'],
        'C': np.arange(1,21,5),
        'gamma': np.arange(0.1, 1, 0.2),
    }
    tuning = GridSearchCV(model, param_dist, n_jobs=-1, cv=2, verbose=2)
    tuning.fit(x_train, y_train)
    print(tuning.best_params_)
    preds = tuning.best_estimator_.predict(x_dev)
    return tuning.best_estimator_, {
        'r2': r2_score(preds, y_dev),
        'MAE': mean_absolute_error(preds, y_dev), 
        'MSE': mean_squared_error(preds, y_dev, squared=True),
        'RMSE': mean_squared_error(preds, y_dev, squared=False)
    }

def model_testing_DT(x_train, x_dev, y_train, y_dev):
    model = DecisionTreeRegressor()
    param_dist = {
        'criterion': ['squared_error', 'absolute_error'], 
        'splitter': ['best', 'random'],
        'max_depth': np.arange(1,21,10), 
        'min_samples_split': np.arange(2, 22,10), 
    }
    tuning = GridSearchCV(model, param_dist, n_jobs=-1, cv=2, verbose=2)
    tuning.fit(x_train, y_train)
    print(tuning.best_params_)
    preds = tuning.best_estimator_.predict(x_dev)
    return tuning.best_estimator_, {
        'r2': r2_score(preds, y_dev),
        'MAE': mean_absolute_error(preds, y_dev), 
        'MSE': mean_squared_error(preds, y_dev, squared=True),
        'RMSE': mean_squared_error(preds, y_dev, squared=False)
    }

def model_testing_RF(x_train, x_dev, y_train, y_dev):
    model = RandomForestRegressor()
    param_dist = {
        'n_estimators': np.arange(10,210, 75),
        'criterion': ['squared_error', 'absolute_error'],
        'max_depth': np.arange(2,22, 20), 
        'min_samples_split': np.arange(2, 22, 20), 
        'max_features': ['sqrt', 'log2'],
    }
    tuning = GridSearchCV(model, param_dist, n_jobs=-1, cv=2, verbose=2)
    tuning.fit(x_train, y_train)
    print(tuning.best_params_)
    preds = tuning.best_estimator_.predict(x_dev)
    return tuning.best_estimator_, {
        'r2': r2_score(preds, y_dev),
        'MAE': mean_absolute_error(preds, y_dev), 
        'MSE': mean_squared_error(preds, y_dev, squared=True),
        'RMSE': mean_squared_error(preds, y_dev, squared=False)
    }

def model_testing_GB(x_train, x_dev, y_train, y_dev):
    model = GradientBoostingRegressor()
    param_dist = {
        'n_estimators': np.arange(10,210, 75),
        'learning_rate': np.arange(0.1, 0.5, 0.25),
        'max_depth': np.arange(1,11, 5), 
        'subsample': np.arange(0.1,1.1,0.5),
        'min_samples_split': np.arange(2, 21, 20), 
    }
    tuning = GridSearchCV(model, param_dist, n_jobs=-1, cv=2, verbose=2)
    tuning.fit(x_train, y_train)
    print(tuning.best_params_)
    preds = tuning.best_estimator_.predict(x_dev)
    return tuning.best_estimator_, {
        'r2': r2_score(preds, y_dev),
        'MAE': mean_absolute_error(preds, y_dev), 
        'MSE': mean_squared_error(preds, y_dev, squared=True),
        'RMSE': mean_squared_error(preds, y_dev, squared=False)
    }






