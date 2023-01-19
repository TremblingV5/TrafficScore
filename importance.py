from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


data = pd.read_csv("train_1.csv")

col_min_max = {
    np.inf: data[np.isfinite(data)].max()
}
data = data.replace({
    col: col_min_max for col in data.columns
})

data.to_csv("train_2.csv")

height, width = data.shape
x, y = data.iloc[:, 1:width - 1].values, data.iloc[:, width - 1].values

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

labels = data.columns[1:width - 1]

model = RandomForestClassifier()
model.fit(x_train, y_train)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, labels[indices[f]], importances[indices[f]]))
