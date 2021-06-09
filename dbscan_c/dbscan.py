#!/user/bin/python3
from collections import defaultdict
from random import uniform
from math import sqrt
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np

# 数据集，半径，簇中的最小点数


def MyDBSCAN(D, eps, MinPts):

    #    -1 - 表示离群点
    #     0 - 未使用
    labels = [0]*len(D)

    # 标识类号
    C = 0

    # 找到核心点--分配簇
    for P in range(0, len(D)):

        # 只有标识为0可以选为种子点
        if not (labels[P] == 0):
            continue

        # 找到点P的邻居点
        NeighborPts = regionQuery(D, P, eps)

        # 邻居点小于最小数，标记为噪声
        if len(NeighborPts) < MinPts:
            labels[P] = -1
        # 反之，找到一个类
        else:
            C += 1
            growCluster(D, labels, P, NeighborPts, C, eps, MinPts)

    return labels

# 用P点增加新类，也就是找到所有属于新类的点，只应用于未分配的点


def growCluster(D, labels, P, NeighborPts, C, eps, MinPts):
    labels[P] = C

    i = 0
    while i < len(NeighborPts):
        Pn = NeighborPts[i]

        if labels[Pn] == -1:
            labels[Pn] = C
        elif labels[Pn] == 0:
            labels[Pn] = C

            PnNeighborPts = regionQuery(D, Pn, eps)

            # 添加新的点
            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts
            # 如果没有足够点，就是边缘节点
        i += 1


# 找到点p半径eps的所有点
def regionQuery(D, P, eps):
    neighbors = []

    for Pn in range(0, len(D)):
        # 距离小于半径，添加索引
        if np.linalg.norm(D[P] - D[Pn]) < eps:
            neighbors.append(Pn)

    return neighbors


data = pd.read_csv("../iris.data", header=None)  # header 不把第一行做为列属性
#data.columns=['sepal length','sepal width','petal length','petal width','class']

data = data.to_numpy()
# 各取出10条，共30条
train = np.vstack((data[0:10, :], data[50:60, :], data[100:110, :]))

# print("训练数据：\n",train)
X = train[:, 0:4]  # data
# print("训练数据：\n", X)
Y = train[:, 4]  # target
# print("训练数据：\n", Y)
# assign = k_means(X, 3)
my_labels = MyDBSCAN(X, eps=1.8, MinPts=10)

print('\n')
results = {"[sepal length]": train[:, 0],
           "[sepal width]": train[:, 1],
           "[petal length]": train[:, 2],
           "[petal width]": train[:, 3],
           "[Class]": train[:, 4],
           "[聚类结果]": my_labels}

results = DataFrame(results)
print(results)
