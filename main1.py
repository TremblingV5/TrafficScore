from catboost import CatBoostRegressor
import warnings
import utils
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from bayes_opt import BayesianOptimization

warnings.filterwarnings("ignore")


data = utils.read_file("train_2.csv")
height, width = data.shape
x, y = data.iloc[:, 1:width - 1].values, data.iloc[:, width - 1].values

XTrain_src, XTest_src, YTrain_src, YTest_src = train_test_split(
    x, y, test_size=0.3)
# print(XTrain.shape, XTest.shape, YTrain.shape, YTest.shape)

# XTrain = utils.MinMaxPre(XTrain_src)
# XTest = utils.MinMaxPre(XTest_src)
# YTrain = utils.MinMaxPre(YTrain_src.reshape(-1, 1))
# YTest = utils.MinMaxPre(YTest_src.reshape(-1, 1))
# print(XTrain.shape, XTest.shape, YTrain.shape, YTest.shape)

# 固定参数
learning_rate = 0.02
n_estimators = 1000
min_split_gain = 0          # 不需要调整
min_child_samples = 20      # 数据量大时，提升此数值以提升泛化能力
min_child_weight = 1        # 一般设置为1，一个叶子上的最小hessian和

# 调节参数
max_depth = 4               # 重要参数，一般3~5，对模型性能和泛化能力有决定性作用
num_leaves = 15             # 一棵树上的叶子节点数量，一般设置为(0,2^max_depth-1]之间的一个值
subsample = 0.6            # 一般设置到0.8和1之间，防止过拟合。若此参数小于1，将会在每次迭代时不进行重采样并随机选择数据
colsample_bytree = 0.7      # 小于1时选择部分特征，一般是0.8到1以防止过拟合
reg_alpha = 0.              # 调节控制过拟合
reg_lambda = 0.             # 调大使各个特征对模型的影响力趋于平均


def train(iterations, learning_rate, depth, subsample, rsm):
    model = CatBoostRegressor(
        loss_function="RMSE",
        learning_rate=learning_rate,
        iterations=int(iterations),
        depth=int(depth),
        eval_metric="RMSE",
        # random_seed=42,
        l2_leaf_reg=5,
        # bootstrap_type="Bayesian",
        subsample=subsample,
        rsm=rsm,
        min_data_in_leaf=500,
        boosting_type="Plain",
        metric_period=100,
        # task_type="GPU"
        verbose=True
    )

    model.fit(XTrain_src, YTrain_src)

    # predicted = model.predict(XTest)
    # inversed = utils.MinMaxInverse(
    #     XTest_src.reshape(-1, 1), predicted.reshape(-1, 1))

    # inversed = predicted * 100
    # print(YTest_src)

    return utils.validate(model.predict(XTest_src), YTest_src)


params_value_dicts = {
    "iterations": (50, 10000),
    "learning_rate": (0.01, 0.2),
    "depth": (3, 7),
    "subsample": (0.6, 1),
    "rsm": (0.6, 1)
}
# cat_bayes = BayesianOptimization(train, params_value_dicts)
# cat_bayes.maximize(init_points=1, n_iter=20)

# print(cat_bayes.max.get("params"))

model = CatBoostRegressor(
    loss_function="RMSE",
    learning_rate=0.02,
    iterations=int(2000),
    depth=int(3),
    eval_metric="RMSE",
    # random_seed=42,
    l2_leaf_reg=15,
    # bootstrap_type="Bayesian",
    subsample=0.9,
    rsm=0.9,
    min_data_in_leaf=1,
    metric_period=1000,
    # task_type="GPU"
    verbose=True
)

model.fit(XTrain_src, YTrain_src)

# predicted = model.predict(XTest)
# inversed = utils.MinMaxInverse(
#     XTest_src.reshape(-1, 1), predicted.reshape(-1, 1))

# inversed = predicted * 100
# print(YTest_src)

utils.validate(model.predict(XTest_src), YTest_src)
