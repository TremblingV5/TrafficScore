import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error


def read_file(path):
    return pd.read_csv(path)

def validate(predicted, YTest):
    MAE = mean_absolute_error(YTest, predicted)
    MSE = mean_squared_error(YTest, predicted)
    RMSE = np.sqrt(MSE)
    print(MAE, MSE, RMSE)

