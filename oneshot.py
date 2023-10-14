import numpy as np
import pandas as pd
import seaborn as sns
import datetime
from src.data_prep import *
from src.feature_engineering import *
# Load both training and dataset used for end completion. 
dfs = load_sensor("dataset/Plant_1_Weather_Sensor_Data.csv", save=False)
# Get the X and y for all expectedd values
irr_X, irr_y = sensor_for_irradiation(dfs[0])
atemp_X, atemp_y = sensor_for_ambient_temp(dfs[0])
mtemp_X, mtemp_y = sensor_for_module_temp(dfs[0])

# Now that we have all starting values we take a quick glance at one of them
from sklearn.model_selection import train_test_split
from src.model import * 
x_train, x_dev, y_train, y_dev = train_test_split(irr_X, irr_y, test_size=0.2, random_state=42)
mod, met = model_testing_SVR(x_train, x_dev, y_train, y_dev)