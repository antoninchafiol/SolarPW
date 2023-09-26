from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

basic_randForest = RandomForestRegressor(
    n_estimators=100, 
    criterion='squared_error',
    max_depth=None, 
    min_samples_split=2, 
    min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, 
    max_features=1.0, 
    max_leaf_nodes=None, 
    min_impurity_decrease=0.0, 
    bootstrap=True,
    oob_score=False, 
    n_jobs=None, 
    random_state=None, 
    verbose=0, 
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None
) 

best_svr = SVR(
    C=10, 
    gamma=0.1, 
    kernel='linear'
)

best_GBR = GradientBoostingRegressor(
    max_features='sqrt', 
    n_estimators=100,
    max_depth=3,
    subsample=0.7999999999999999
)