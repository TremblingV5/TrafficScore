import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import utils
from sklearn.model_selection import cross_val_score

data = utils.read_file("train_2.csv")
height, width = data.shape
x, y = data.iloc[:, 1:width - 1].values, data.iloc[:, width - 1].values

XTrain, XTest, YTrain, YTest = train_test_split(x, y, test_size=0.3)
print(XTrain.shape, XTest.shape, YTrain.shape, YTest.shape)

# 各个方法的参数以键值对的形式写在对应算法的args参数之后
methods = {
    "随机森林": {
        "class": RandomForestClassifier,
        "args": {
            "n_estimators": 100,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_features": "sqrt",
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
            "bootstrap": True,
            "oob_score": False,
            "n_jobs": None,
            "random_state": None,
            "verbose": 0,
            "warm_start": False,
            "class_weight": None,
            "ccp_alpha": 0.0,
            "max_samples": None
        }
    },
    "SVM-LinearSVC": {
        "class": LinearSVC,
        "args": {
            "penalty": "l2",
            "loss": "squared_hinge",
            "dual": True,
            "tol": 1e-4,
            "C": 1.0,
            "multi_class": "ovr",
            "fit_intercept": True,
            "intercept_scaling": 1,
            "class_weight": None,
            "verbose": 0,
            "random_state": None,
            "max_iter": 1000
        }
    },
    "K近邻": {
        "class": KNeighborsClassifier,
        "args": {
            "n_neighbors": 5,
            "weights": "uniform",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": 2,
            "metric": "minkowski",
            "metric_params": None,
            "n_jobs": None
        }
    },
    "Naive Bayes": {
        "class": MultinomialNB,
        "args": {
            "alpha": 1.0,
            "fit_prior": True,
            "class_prior": None
        }
    },
    "随机梯度下降": {
        "class": SGDClassifier,
        "args": {
            "loss": "hinge",
            "penalty": "l2",
            "l1_ratio": 0.15,
            "fit_intercept": True,
            "max_iter": 1000,
            "tol": 1e-3,
            "shuffle": True,
            "verbose": 0,
            "epsilon": 0.1,
            "n_jobs": None,
            "random_state": None,
            "learning_rate": "optimal",
            "eta0": 0.0,
            "power_t": 0.5,
            "early_stopping": False,
            "validation_fraction": 0.1,
            "n_iter_no_change": 5,
            "class_weight": None,
            "warm_start": False,
            "average": False
        }
    },
    "多层感知器": {
        "class": MLPClassifier,
        "args": {
            "hidden_layer_sizes": (100,),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "batch_size": "auto",
            "learning_rate": "constant",
            "learning_rate_init": 0.001,
            "power_t": 0.5,
            "max_iter": 200,
            "shuffle": True,
            "random_state": None,
            "tol": 1e-4,
            "verbose": False,
            "warm_start": False,
            "momentum": 0.9,
            "nesterovs_momentum": True,
            "early_stopping": False,
            "validation_fraction": 0.1,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-8,
            "n_iter_no_change": 10,
            "max_fun": 15000
        }
    },
    "逻辑回归": {
        "class": LogisticRegression,
        "args": {
            "penalty": "l2",
            "dual": False,
            "tol": 1e-4,
            "C": 1.0,
            "fit_intercept": True,
            "intercept_scaling": 1,
            "class_weight": None,
            "random_state": None,
            "solver": "lbfgs",
            "max_iter": 100,
            "multi_class": "auto",
            "verbose": 0,
            "warm_start": False,
            "n_jobs": None,
            "l1_ratio": None
        }
    },
    "AdaBoost": {
        "class": AdaBoostClassifier,
        "args": {
            "base_estimator": None,
            "n_estimators": 50,
            "learning_rate": 1.0,
            "algorithm": "SAMME.R",
            "random_state": None
        }
    },
    "多元函数回归": {
        "class": LinearRegression,
        "args": {
            "fit_intercept": True,
            "normalize": "deprecated",
            "copy_X": True,
            "n_jobs": None,
            "positive": False
        }
    }
    # "梯度提升": {
    #     "class": GradientBoostingClassifier,
    #     "args": {
    #         "loss": "log_loss",
    #         "learning_rate": 0.1,
    #         "n_estimators": 100,
    #         "subsample": 1.0,
    #         "criterion": "friedman_mse",
    #         "min_samples_split": 2,
    #         "min_samples_leaf": 1,
    #         "min_weight_fraction_leaf": 0.0,
    #         "init": None,
    #         "random_state": None,
    #         "max_features": None,
    #         "verbose": 0,
    #         "max_leaf_nodes": None,
    #         "warm_start": False,
    #         "validation_fraction": 0.1,
    #         "n_iter_no_change": None,
    #         "tol": 1e-4,
    #         "ccp_alpha": 0.0
    #     }
    # }
}

for item in methods:
    print("算法：", item)
    if methods[item]["args"] == {}:
        model = methods[item]["class"]()
    else:
        model = methods[item]["class"](**methods[item]["args"])
    # model.fit(XTrain, YTrain)

    valScore = cross_val_score(model, XTrain, YTrain, cv=100, n_jobs=-1, scoring="neg_mean_squared_error")
    print(max(valScore) * -1)

    model.fit(XTrain, YTrain)
    utils.validate(model.predict(XTest), YTest)
