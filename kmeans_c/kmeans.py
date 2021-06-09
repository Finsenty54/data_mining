from collections import defaultdict
from random import uniform
from math import sqrt
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np


def point_avg(points):
    # 每个维度的中心值
    dimensions = len(points[0])

    new_center = []

    for dimension in range(dimensions):
        dim_sum = 0
        for p in points:
            dim_sum += p[dimension]

        # 每个维度的平均值
        new_center.append(dim_sum / float(len(points)))

    return new_center

# 更新中心值，即每个维度的中心值


def update_centers(data_set, assignments):
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)

    for points in new_means.values():
        centers.append(point_avg(points))

    return centers


def assign_points(data_points, centers):
    # 数据距离哪个质心最近 标识
    assignments = []
    for point in data_points:
        shortest = float("inf")  # 正无穷
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    """
    dimensions = len(a)

    _sum = 0
    for dimension in range(dimensions):
        difference_sq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_sq
    return sqrt(_sum)

# 选择个k个初始质心


def generate_k(data_set, k):
    centers = []
    dimensions = len(data_set[0])
    min_max = defaultdict(int)

    # 找到每个维度所有座标中的最大值和最小值
    for point in data_set:
        for i in range(dimensions):
            val = point[i]
            min_key = 'min_%d' % i
            max_key = 'max_%d' % i
            if min_key not in min_max or val < min_max[min_key]:
                min_max[min_key] = val
            if max_key not in min_max or val > min_max[max_key]:
                min_max[max_key] = val

    # 在每个维度的最大值和最小值这个范围中 随机选择值
    for _k in range(k):
        rand_point = []
        for i in range(dimensions):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]

            rand_point.append(uniform(min_val, max_val))

        centers.append(rand_point)

    return centers


def k_means(dataset, k):
    k_points = generate_k(dataset, k)
    print("初始点：\n",k_points)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    print("迭代终止后数据属于的簇：\n", assignments)
    return assignments


data = pd.read_csv("../iris.data", header=None)  # header 不把第一行做为列属性
#data.columns=['sepal length','sepal width','petal length','petal width','class']

data = data.to_numpy()
# 各取出10条，共30条
train = np.vstack((data[0:10, :], data[50:60, :], data[100:110, :]))

# print("训练数据：\n",train)
X = train[:, 0:4]  # data
print("训练数据：\n", X)
Y = train[:, 4]  # target
print("训练数据：\n", Y)
assign = k_means(X, 3)

print('\n')
results={"[sepal length]":train[:,0],
"[sepal width]":train[:,1],
"[petal length]":train[:,2],
"[petal width]":train[:,3],
"[Class]":train[:,4],
"[聚类结果]":assign}

results=DataFrame(results)
print(results)