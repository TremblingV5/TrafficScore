from lightgbm.sklearn import LGBMRegressor
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

# 固定参数
learning_rate = 0.1
n_estimators = 1000          # 100~1000，设计较大值配合early_stopping_round自动选择，设置过大会过拟合
min_split_gain = 0          # 不需要调整
min_child_samples = 20      # 数据量大时，提升此数值以提升泛化能力
min_child_weight = 1        # 一般设置为1，一个叶子上的最小hessian和

# 调节参数
max_depth = 4               # 重要参数，一般3~5，对模型性能和泛化能力有决定性作用
num_leaves = 15             # 一棵树上的叶子节点数量，一般设置为(0,2^max_depth-1]之间的一个值
subsample = 0.7            # 一般设置到0.8和1之间，防止过拟合。若此参数小于1，将会在每次迭代时不进行重采样并随机选择数据
colsample_bytree = 0.7      # 小于1时选择部分特征，一般是0.8到1以防止过拟合
reg_alpha = 0.              # 调节控制过拟合
reg_lambda = 0.             # 调大使各个特征对模型的影响力趋于平均

model = LGBMRegressor(
    boosting_type='gbdt',
    metric="rmse",
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    min_split_gain=min_split_gain,
    min_child_samples=min_child_samples,
    min_child_weight=min_child_weight,
    max_depth=max_depth,
    num_leaves=num_leaves,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    reg_alpha=reg_alpha,
    reg_lambda=reg_lambda
)

model.fit(XTrain, YTrain)
utils.validate(model.predict(XTest), YTest)
