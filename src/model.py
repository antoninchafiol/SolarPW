from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

def model_testing(X,y, mtype):
    # Quick test of models, without any tunings.
    res = []
    x_train, x_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=42)
    if mtype=='SVM':
        res= plain_testing(x_train, x_dev, y_train, y_dev, SVR())
    elif mtype=='DT':
        res= plain_testing(x_train, x_dev, y_train, y_dev, DecisionTreeRegressor())
    elif mtype=='RF':
        res= plain_testing(x_train, x_dev, y_train, y_dev, RandomForestRegressor())
    elif mtype=='GB':
        res= plain_testing(x_train, x_dev, y_train, y_dev, GradientBoostingRegressor())
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
    return 0

def model_testing_DT(x_train, x_dev, y_train, y_dev):
    return 0

def model_testing_RF(x_train, x_dev, y_train, y_dev):
    return 0

def model_testing_GB(x_train, x_dev, y_train, y_dev):
    return 0

def model_testing_NN(x_train, x_dev, y_train, y_dev):
    return 0
