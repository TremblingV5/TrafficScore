import utils
import numpy as np
import pandas as pd


def get(range_string):
    range_string = range_string.replace(" ", "")
    range_string = range_string.replace("[", "")
    range_string = range_string.replace(")", "")

    data = range_string.split(",")

    value1 = int(data[0])
    value2 = int(data[1]) if data[1] != "inf" else np.inf

    return value1, value2

df = utils.read_file("train.csv")

print(df.shape)
height, width = df.shape

data = list()

for i in range(height):
    temp = list()
    for j in range(width):
        if j != 0 and j != width - 1:
            b, r = get(df.iloc[i, j])
            temp.append(b)
            temp.append(r)
        else:
            if j == 0:
                continue
            temp.append(df.iloc[i, j])
    data.append(temp)

newData = pd.DataFrame(data)
print(newData.head())

newData.to_csv("train_1.csv")
