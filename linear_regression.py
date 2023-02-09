from sklearn.linear_model import LinearRegression
import warnings
import utils
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


data = utils.read_file("train_2.csv")
height, width = data.shape
x, y = data.iloc[:, 1:width - 1].values, data.iloc[:, width - 1].values

XTrain, XTest, YTrain, YTest = train_test_split(x, y, test_size=0.3)
print(XTrain.shape, XTest.shape, YTrain.shape, YTest.shape)


model = LinearRegression(
    fit_intercept=True,
    normalize="deprecated",
    copy_X=True,
    n_jobs=None,
    positive=False
)
model.fit(XTrain, YTrain)
utils.validate(model.predict(XTest), YTest)
