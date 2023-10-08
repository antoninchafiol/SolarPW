from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV

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
    elif mtype=='NN':
        res= model_testing_NN(x_train, x_dev, y_train, y_dev)
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
        'kernel': ['linear','rbf', 'poly'],
        'C': np.arange(1,100, 10),
        'gamma': np.arange(1,10,0.1),
    }
    tuning = RandomizedSearchCV(model, param_dist, n_iter=50 , n_jobs=-1, random_state=42)
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
        'criterion': ['mse', 'mae'], 
        'splitter': ['best', 'random'],
        'max_depth': np.arange(1,20), 
        'min_samples_split': np.arange(2, 21), 
        'max_samples_leaf': np.arange(1,20),
    }
    tuning = RandomizedSearchCV(model, param_dist, n_iter=50 , n_jobs=-1, random_state=42)
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
        'n_estimators': np.arange(10,200),
        'criterion': ['mse', 'mae'],
        'max_depth': np.arange(1,20), 
        'min_samples_split': np.arange(2, 21), 
        'max_samples_leaf': np.arange(1,20),
        'max_features': ['auto', 'sqrt', 'log2', None],
    }
    tuning = RandomizedSearchCV(model, param_dist, n_iter=50 , n_jobs=-1, random_state=42)
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
        'n_estimators': np.arange(10,200),
        'learning_rate': np.arange(0.01, 0.5, 0.05),
        'max_depth': np.arange(1,10), 
        'subsample': np.arange(0.1,1.0,0.1),
        'min_samples_split': np.arange(2, 21), 
        'max_samples_leaf': np.arange(1,20),
    }
    tuning = RandomizedSearchCV(model, param_dist, n_iter=50 , n_jobs=-1, random_state=42)
    tuning.fit(x_train, y_train)
    print(tuning.best_params_)
    preds = tuning.best_estimator_.predict(x_dev)
    return tuning.best_estimator_, {
        'r2': r2_score(preds, y_dev),
        'MAE': mean_absolute_error(preds, y_dev), 
        'MSE': mean_squared_error(preds, y_dev, squared=True),
        'RMSE': mean_squared_error(preds, y_dev, squared=False)
    }
def model_testing_NN(x_train, x_dev, y_train, y_dev):
    return 0



